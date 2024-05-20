import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchsummary import summary
import numpy as np
from model import PositionalEncoding, get_bou_features, find_best_shift, get_img_tokens
from model import IterWholeFirst
from einops import rearrange
import torchvision.transforms as T
from PIL import Image
from featup.util import norm, unnorm
from featup.plotting import plot_feats
from collections import OrderedDict
import time


def add_extra_channels(conv2d: nn.Conv2d, extra_chan=1):
    """
    Add extra channels to a Conv2d layer.
    """
    device = conv2d.weight.device
    new_conv2d = nn.Conv2d(
        conv2d.in_channels + extra_chan,
        conv2d.out_channels,
        conv2d.kernel_size,
        conv2d.stride,
        conv2d.padding,
        conv2d.dilation,
        conv2d.groups,
        conv2d.bias is not None,
        conv2d.padding_mode,
    ).to(device)
    new_dict = OrderedDict()
    for name, param in new_conv2d.state_dict().items():
        new_param = conv2d.state_dict()[name]
        if new_param.shape != param.shape:
            c, _, w, h = param.shape
            pads = torch.zeros((c, extra_chan, w, h)).to(device)
            nn.init.kaiming_normal_(pads)
            new_param = torch.cat((new_param, pads), dim=1)
        new_dict[name] = new_param
    new_conv2d.load_state_dict(new_dict)
    return new_conv2d


def get_dino4(device="cuda") -> nn.Module:
    featup = torch.hub.load(
        "mhamilton723/FeatUp",
        "dino16",
        use_norm=True,
    ).to(device)
    dino4 = featup.model
    new_proj = add_extra_channels(dino4[0].model.patch_embed.proj, 1)
    dino4[0].model.patch_embed.proj = new_proj
    return dino4


def get_raw_dino(device="cuda") -> nn.Module:
    featup = torch.hub.load(
        "mhamilton723/FeatUp",
        "dino16",
        use_norm=True,
    ).to(device)
    dino = featup.model
    return dino


def get_res4(device="cuda") -> nn.Module:
    res4 = resnet50(pretrained=True).to(device)
    new_conv1 = add_extra_channels(res4.conv1, 1)
    res4.conv1 = new_conv1
    res4 = nn.Sequential(*list(res4.children())[:-3])
    return res4

def get_raw_res(device="cuda") -> nn.Module:
    res = resnet50(pretrained=True).to(device)
    res = nn.Sequential(*list(res.children())[:-3])
    return res


class ResDetr(nn.Module):
    def __init__(self, boundary_num=80, device="cuda"):
        super(ResDetr, self).__init__()
        self.res4 = get_res4()
        # freeze res4
        for param in self.res4.parameters():
            param.requires_grad = False
        # unfreeze the first layer in res4
        for param in self.res4[0].parameters():
            param.requires_grad = True
        self.raw_res = get_raw_res()
        # freeze the raw res
        for param in self.raw_res.parameters():
            param.requires_grad = False
        self.hidden_dim = 1024
        self.mem_img_fc = nn.Linear(1024, 1024)
        self.cur_img_fc = nn.Linear(1024, 1024)
        self.layernorm = nn.LayerNorm(self.hidden_dim).to(device)
        self.pos_enc = PositionalEncoding(self.hidden_dim).to(device)
        self.boundary_num = boundary_num
        self.query_embed = nn.Embedding(
            self.boundary_num,
            self.hidden_dim,
        ).to(device)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=1,
                batch_first=True,
            ),
            num_layers=1,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.hidden_dim,
                nhead=1,
                batch_first=True,
            ),
            num_layers=1,
        )
        self.xy_fc = nn.Linear(self.hidden_dim, 2).to(device)

    def forward(
        self,
        fir_img: torch.Tensor,
        fir_mask: torch.Tensor,
        pre_img: torch.Tensor,
        pre_mask: torch.Tensor,
        cur_img: torch.Tensor,
    ) -> torch.Tensor:
        fir_mask = fir_mask.unsqueeze(1)
        pre_mask = pre_mask.unsqueeze(1)
        fir_con = torch.cat([fir_img, fir_mask], dim=1)
        pre_con = torch.cat([pre_img, pre_mask], dim=1)
        fir_con_feats = self.res4(fir_con)
        pre_con_feats = self.res4(pre_con)
        fir_con_tokens = get_img_tokens(fir_con_feats)
        pre_con_tokens = get_img_tokens(pre_con_feats)
        mem_img_tokens = torch.cat((fir_con_tokens, pre_con_tokens), dim=1)

        cur_img_feats = self.raw_res(cur_img)
        cur_img_tokens = get_img_tokens(cur_img_feats)

        mem_img_tokens = self.mem_img_fc(mem_img_tokens)
        cur_img_tokens = self.cur_img_fc(cur_img_tokens)
        img_tokens = torch.cat((mem_img_tokens, cur_img_tokens), dim=1)
        img_tokens = self.layernorm(img_tokens)
        img_tokens = self.pos_enc(img_tokens)
        img_tokens = self.transformer_encoder(img_tokens)

        B, S, D = mem_img_tokens.shape
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        queries = self.transformer_decoder(queries, img_tokens)

        xy = self.xy_fc(queries).sigmoid() * 224
        return xy

class DinoDETR(nn.Module):
    def __init__(self, boundary_num=80, device="cuda"):
        super(DinoDETR, self).__init__()
        self.raw_dino = get_raw_dino()
        # freeze raw_dino
        for param in self.raw_dino.parameters():
            param.requires_grad = False

        self.con_to_img_conv = nn.Conv2d(4, 3, 1)
        self.mem_img_fc = nn.Linear(384, 384)
        self.cur_img_fc = nn.Linear(384, 384)
        self.hidden_dim = 384
        self.layernorm = nn.LayerNorm(self.hidden_dim).to(device)
        self.pos_enc = PositionalEncoding(self.hidden_dim).to(device)
        self.boundary_num = boundary_num
        self.query_embed = nn.Embedding(
            self.boundary_num,
            self.hidden_dim,
        ).to(device)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=1,
                batch_first=True,
            ),
            num_layers=1,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.hidden_dim,
                nhead=1,
                batch_first=True,
            ),
            num_layers=1,
        )
        self.xy_fc = nn.Linear(self.hidden_dim, 2).to(device)

    def forward(
        self,
        fir_img: torch.Tensor,
        fir_mask: torch.Tensor,
        pre_img: torch.Tensor,
        pre_mask: torch.Tensor,
        cur_img: torch.Tensor,
    ) -> torch.Tensor:
        fir_mask = fir_mask.unsqueeze(1)
        pre_mask = pre_mask.unsqueeze(1)
        fir_con = self.con_to_img_conv(
            torch.cat(
                [fir_img, fir_mask],
                dim=1,
            ),
        )
        pre_con = self.con_to_img_conv(
            torch.cat(
                [pre_img, pre_mask],
                dim=1,
            ),
        )
        fir_con_feats = self.raw_dino(fir_con)
        pre_con_feats = self.raw_dino(pre_con)
        fir_con_tokens = get_img_tokens(fir_con_feats)
        pre_con_tokens = get_img_tokens(pre_con_feats)
        mem_img_tokens = torch.cat((fir_con_tokens, pre_con_tokens), dim=1)

        cur_img_feats = self.raw_dino(cur_img)
        cur_img_tokens = get_img_tokens(cur_img_feats)

        mem_img_tokens = self.mem_img_fc(mem_img_tokens)
        cur_img_tokens = self.cur_img_fc(cur_img_tokens)
        img_tokens = torch.cat((mem_img_tokens, cur_img_tokens), dim=1)
        img_tokens = self.layernorm(img_tokens)
        img_tokens = self.pos_enc(img_tokens)
        img_tokens = self.transformer_encoder(img_tokens)

        B, S, D = mem_img_tokens.shape
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        queries = self.transformer_decoder(queries, img_tokens)

        xy = self.xy_fc(queries).sigmoid() * 224
        return xy


class DinoDetrSimpleFuse(nn.Module):
    def __init__(self, boundary_num=80, device="cuda"):
        super(DinoDetrSimpleFuse, self).__init__()
        self.raw_dino = get_raw_dino()
        # freeze raw_dino
        for param in self.raw_dino.parameters():
            param.requires_grad = False

        self.mem_mask_fc = nn.Linear(384, 384)
        self.mem_img_fc = nn.Linear(384, 384)
        self.cur_img_fc = nn.Linear(384, 384)
        self.hidden_dim = 384
        self.layernorm = nn.LayerNorm(self.hidden_dim).to(device)
        self.pos_enc = PositionalEncoding(self.hidden_dim).to(device)
        self.boundary_num = boundary_num
        self.query_embed = nn.Embedding(
            self.boundary_num,
            self.hidden_dim,
        ).to(device)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=1,
                batch_first=True,
            ),
            num_layers=1,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.hidden_dim,
                nhead=1,
                batch_first=True,
            ),
            num_layers=1,
        )
        self.xy_fc = nn.Linear(self.hidden_dim, 2).to(device)

    def forward(
        self,
        fir_img: torch.Tensor,
        fir_mask: torch.Tensor,
        pre_img: torch.Tensor,
        pre_mask: torch.Tensor,
        cur_img: torch.Tensor,
    ) -> torch.Tensor:
        fir_mask[fir_mask > 0] = 255
        pre_mask[pre_mask > 0] = 255
        fir_mask = fir_mask.unsqueeze(1).expand(-1, 3, -1, -1)
        pre_mask = pre_mask.unsqueeze(1).expand(-1, 3, -1, -1)
        fir_mask_feats = self.raw_dino(fir_mask)
        pre_mask_feats = self.raw_dino(pre_mask)
        fir_mask_tokens = get_img_tokens(fir_mask_feats)
        pre_mask_tokens = get_img_tokens(pre_mask_feats)

        fir_img_feats = self.raw_dino(fir_img)
        pre_img_feats = self.raw_dino(pre_img)
        cur_img_feats = self.raw_dino(cur_img)
        fir_img_tokens = get_img_tokens(fir_img_feats)
        pre_img_tokens = get_img_tokens(pre_img_feats)
        cur_img_tokens = get_img_tokens(cur_img_feats)

        mem_mask_tokens = torch.cat((fir_mask_tokens, pre_mask_tokens), dim=1)
        mem_img_tokens = torch.cat((fir_img_tokens, pre_img_tokens), dim=1)
        cur_img_tokens = cur_img_tokens
        mem_mask_tokens = self.mem_mask_fc(mem_mask_tokens)
        mem_img_tokens = self.mem_img_fc(mem_img_tokens)
        cur_img_tokens = self.cur_img_fc(cur_img_tokens)
        img_tokens = torch.cat((mem_mask_tokens, mem_img_tokens, cur_img_tokens), dim=1)
        img_tokens = self.layernorm(img_tokens)
        img_tokens = self.pos_enc(img_tokens)
        img_tokens = self.transformer_encoder(img_tokens)

        B, S, D = mem_img_tokens.shape
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        queries = self.transformer_decoder(queries, img_tokens)

        xy = self.xy_fc(queries).sigmoid() * 224
        return xy


class DinoDetrMaskMul(nn.Module):
    def __init__(self, mul_before: bool, boundary_num=80, device="cuda"):
        super(DinoDetrMaskMul, self).__init__()
        self.mul_before = mul_before
        self.raw_dino = get_raw_dino()
        # freeze raw_dino
        for param in self.raw_dino.parameters():
            param.requires_grad = False

        self.mem_img_fc = nn.Linear(384, 384)
        self.cur_img_fc = nn.Linear(384, 384)
        self.hidden_dim = 384
        self.layernorm = nn.LayerNorm(self.hidden_dim).to(device)
        self.pos_enc = PositionalEncoding(self.hidden_dim).to(device)
        self.boundary_num = boundary_num
        self.query_embed = nn.Embedding(
            self.boundary_num,
            self.hidden_dim,
        ).to(device)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=1,
                batch_first=True,
            ),
            num_layers=1,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.hidden_dim,
                nhead=1,
                batch_first=True,
            ),
            num_layers=1,
        )
        self.xy_fc = nn.Linear(self.hidden_dim, 2).to(device)

    def forward(
        self,
        fir_img: torch.Tensor,
        fir_mask: torch.Tensor,
        pre_img: torch.Tensor,
        pre_mask: torch.Tensor,
        cur_img: torch.Tensor,
    ) -> torch.Tensor:
        fir_mask = fir_mask.unsqueeze(1)
        pre_mask = pre_mask.unsqueeze(1)
        if self.mul_before:
            fir_img = fir_img * fir_mask
            pre_img = pre_img * pre_mask
        fir_img_feats = self.raw_dino(fir_img)
        pre_img_feats = self.raw_dino(pre_img)
        if not self.mul_before:
            fir_mask = F.interpolate(
                fir_mask,
                size=(14, 14),
                mode="bilinear",
            )
            pre_mask = F.interpolate(
                pre_mask,
                size=(14, 14),
                mode="bilinear",
            )
            fir_img_feats = fir_img_feats * fir_mask
            pre_img_feats = pre_img_feats * pre_mask

        cur_img_feats = self.raw_dino(cur_img)
        fir_img_tokens = get_img_tokens(fir_img_feats)
        pre_img_tokens = get_img_tokens(pre_img_feats)
        cur_img_tokens = get_img_tokens(cur_img_feats)

        mem_img_tokens = torch.cat((fir_img_tokens, pre_img_tokens), dim=1)
        mem_img_tokens = self.mem_img_fc(mem_img_tokens)
        cur_img_tokens = self.cur_img_fc(cur_img_tokens)
        img_tokens = torch.cat((mem_img_tokens, cur_img_tokens), dim=1)
        img_tokens = self.layernorm(img_tokens)
        img_tokens = self.pos_enc(img_tokens)
        img_tokens = self.transformer_encoder(img_tokens)

        B, S, D = mem_img_tokens.shape
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        queries = self.transformer_decoder(queries, img_tokens)

        xy = self.xy_fc(queries).sigmoid() * 224
        return xy
