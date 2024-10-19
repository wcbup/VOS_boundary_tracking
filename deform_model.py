import torch
from Loader_17 import DAVIS_Rawset, DAVIS_Infer, DAVIS_Dataset, normalize
from polygon import RasLoss, SoftPolygon
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import copy
import matplotlib.pyplot as plt
from einops import rearrange
import torch.nn.functional as F
from MyLoss import deviation_loss, total_len_loss
from torch.nn.init import xavier_uniform_
from ms_deform_attn import MSDeformAttn
from torch.utils.data import Dataset
import json
import random


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def get_valid_ratio(mask):
    _, H, W = mask.shape
    valid_H = torch.sum(~mask[:, :, 0], 1)
    valid_W = torch.sum(~mask[:, 0, :], 1)
    valid_ratio_h = valid_H.float() / H
    valid_ratio_w = valid_W.float() / W
    valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
    return valid_ratio


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_heads=8,
        n_points=4,
    ):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(
        self,
        src,
        pos,
        reference_points,
        spatial_shapes,
        level_start_index,
        padding_mask=None,
    ):
        # self attention
        src2 = self.self_attn(
            self.with_pos_embed(src, pos),
            reference_points,
            src,
            spatial_shapes,
            level_start_index,
            padding_mask,
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(
        self,
        src,
        spatial_shapes,
        level_start_index,
        valid_ratios,
        pos=None,
        padding_mask=None,
    ):
        output = src
        reference_points = self.get_reference_points(
            spatial_shapes, valid_ratios, device=src.device
        )
        for _, layer in enumerate(self.layers):
            output = layer(
                output,
                pos,
                reference_points,
                spatial_shapes,
                level_start_index,
                padding_mask,
            )

        return output


def get_bou_feats(img_feats: torch.Tensor, boundary: torch.Tensor) -> torch.Tensor:
    return img_feats[
        torch.arange(boundary.shape[0]).unsqueeze(1),
        :,
        boundary[:, :, 1],
        boundary[:, :, 0],
    ]


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, dropout=0.1, max_seq_len=102400) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-np.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


def get_img_tokens(img_features: torch.Tensor) -> torch.Tensor:
    img_tokens = rearrange(
        img_features,
        "b c h w -> b (h w) c",
    )
    return img_tokens


def get_extrame_4_points(batch_indices: torch.Tensor):
    if batch_indices.shape[0] == 0:
        mid = 224 // 2
        return torch.tensor([[mid, mid], [mid, mid], [mid, mid], [mid, mid]])
    x_min = batch_indices[:, 1].min()
    x_max = batch_indices[:, 1].max()
    y_min = batch_indices[:, 0].min()
    y_max = batch_indices[:, 0].max()
    return torch.tensor(
        [[x_min, y_min], [x_min, y_max], [x_max, y_max], [x_max, y_min]]
    )


def get_bounding_box(sgm: torch.Tensor):
    indices = torch.nonzero(sgm)
    batch_size = sgm.shape[0]
    bounding_boxes = []
    for i in range(batch_size):
        batch_indices = indices[indices[:, 0] == i][:, 1:]
        bounding_boxes.append(get_extrame_4_points(batch_indices))
    return torch.stack(bounding_boxes)


def add_mid_points(points: torch.Tensor) -> torch.Tensor:
    points_shift = torch.roll(points, 1, 1)
    mid_points = (points + points_shift) / 2
    new_points = torch.zeros((points.shape[0], points.shape[1] * 2, 2)).to(
        points.device
    )
    new_points[:, ::2] = mid_points
    new_points[:, 1::2] = points
    return new_points


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_heads=8,
        n_points=4,
    ):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(
        self,
        tgt,
        query_pos,
        reference_points,
        src,
        src_spatial_shapes,
        level_start_index,
        src_padding_mask=None,
    ):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1)
        )[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, query_pos),
            reference_points,
            src,
            src_spatial_shapes,
            level_start_index,
            src_padding_mask,
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt,
        reference_points,
        src,
        src_spatial_shapes,
        src_level_start_index,
        src_valid_ratios,
        query_pos=None,
        src_padding_mask=None,
    ):
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = (
                    reference_points[:, :, None]
                    * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
                )
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = (
                    reference_points[:, :, None] * src_valid_ratios[:, None]
                )
            output = layer(
                output,
                query_pos,
                reference_points_input,
                src,
                src_spatial_shapes,
                src_level_start_index,
                src_padding_mask,
            )

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class DeformLearnImage(nn.Module):
    def __init__(
        self,
        layer_num=1,
        up_scale_num=4,
        head_num=6,
        medium_level_size=[14, 28, 56, 112],
        offset_limit=56,
        n_points=4,
        freeze_backbone=True,
    ) -> None:
        super(DeformLearnImage, self).__init__()
        self.up_scale_num = up_scale_num
        self.offset_limit = offset_limit
        self.medium_level_size = medium_level_size
        self.featup = torch.hub.load(
            "mhamilton723/FeatUp",
            "dino16",
            use_norm=True,
        ).cuda()
        if freeze_backbone:
            for param in self.featup.parameters():
                param.requires_grad = False
        d_model = 384
        d_ffn = 1024
        n_levels = len(medium_level_size) + 1
        self.pos_enoc = PositionalEncoding(d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        deform_encoder_layer = DeformableTransformerEncoderLayer(
            d_model=d_model,
            d_ffn=d_ffn,
            n_levels=n_levels,
            n_heads=head_num,
            n_points=n_points,
        )
        self.deform_encoder = DeformableTransformerEncoder(
            deform_encoder_layer,
            num_layers=layer_num,
        )
        query_num = 4
        self.query_num = query_num
        for i in range(up_scale_num):
            query_num *= 2
        self.query_embed = nn.Embedding(query_num, d_model)
        # init the query embedding
        xavier_uniform_(self.query_embed.weight)
        deform_decoder_layer = DeformableTransformerDecoderLayer(
            d_model=d_model,
            d_ffn=d_ffn,
            n_levels=n_levels,
            n_heads=head_num,
            n_points=n_points,
        )
        deform_decoder = DeformableTransformerDecoder(
            deform_decoder_layer,
            num_layers=layer_num,
        )
        self.deform_decoders = _get_clones(deform_decoder, up_scale_num + 1)
        xy_fc = MLP(d_model, d_model, 2, 3).cuda()
        # xy_fc = nn.Linear(d_model, 2).cuda()
        self.xy_fc = _get_clones(xy_fc, up_scale_num + 1)

    def get_valid_ratio(self, mask: torch.Tensor):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(
        self,
        img: torch.Tensor,
        mask: torch.Tensor,
    ):
        feats = self.featup(img)
        # prepare the input for the MSDeformAttn module
        srcs = []
        padding_masks = []
        for low_res in self.medium_level_size:
            srcs.append(
                F.interpolate(
                    feats,
                    size=(low_res, low_res),
                    mode="bilinear",
                ),
            )
        srcs.append(feats)
        for src in srcs:
            padding_masks.append(torch.zeros_like(src[:, 0:1, :, :]).squeeze(1).bool())
        src_flatten = []
        spatial_shapes = []
        for src in srcs:
            src_flatten.append(
                rearrange(src, "b c h w -> b (h w) c"),
            )
            spatial_shapes.append(src.shape[-2:])
        level_start_index = torch.cat(
            (
                torch.tensor([0]),
                torch.cumsum(
                    torch.tensor([x.shape[1] for x in src_flatten]),
                    0,
                )[:-1],
            )
        ).cuda()
        src_flatten = torch.cat(src_flatten, 1).cuda()
        valid_ratios = torch.stack(
            [self.get_valid_ratio(mask) for mask in padding_masks],
            1,
        ).cuda()
        spatial_shapes = torch.as_tensor(
            spatial_shapes,
            dtype=torch.long,
            device=src_flatten.device,
        )
        src_flatten = self.layer_norm(src_flatten)
        src_flatten = self.pos_enoc(src_flatten)
        src_flatten = self.deform_encoder(
            src=src_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
        )

        B, S, C = src_flatten.shape
        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1).cuda()
        init_bou = get_bounding_box(mask).cuda() / 224
        current_query_num = init_bou.shape[1]
        current_query = queries[:, :current_query_num]
        decode_output, _ = self.deform_decoders[0](
            current_query,
            init_bou,
            src_flatten,
            spatial_shapes,
            level_start_index,
            valid_ratios,
        )

        xy_offset = (
            (self.xy_fc[0](decode_output).sigmoid() - 0.5) * self.offset_limit / 224
        )
        init_bou += xy_offset

        # xy_offset = self.xy_fc[0](decode_output)
        # xy_offset = xy_offset.sigmoid()
        # init_bou = inverse_sigmoid(init_bou) + xy_offset
        # init_bou = init_bou.sigmoid().clone()

        # init_bou = init_bou.clamp(0, 1)

        results = [init_bou]
        for i in range(self.up_scale_num):
            new_query = queries[:, current_query_num : current_query_num * 2]
            current_query_num *= 2

            current_query = torch.zeros((B, current_query_num, C)).to(
                src_flatten.device
            )
            current_query[:, ::2] = new_query
            current_query[:, 1::2] = decode_output
            # current_query = torch.cat([new_query, decode_output], 1)

            cur_bou = add_mid_points(results[-1])
            decode_output, _ = self.deform_decoders[i + 1](
                current_query,
                cur_bou,
                src_flatten,
                spatial_shapes,
                level_start_index,
                valid_ratios,
            )

            xy_offset = (
                (self.xy_fc[i + 1](decode_output).sigmoid() - 0.5)
                * self.offset_limit
                / 224
            )
            cur_bou += xy_offset

            # xy_offset = self.xy_fc[i + 1](decode_output)
            # cur_bou = inverse_sigmoid(cur_bou) + xy_offset
            # cur_bou = cur_bou.sigmoid().clone()

            # cur_bou = cur_bou.clamp(0, 1)

            results.append(cur_bou)
        results = [result * 224 for result in results]

        if self.training:
            return results
        else:
            return results[-1]


def get_bou_iou(
    index: int,
    boundary: torch.Tensor,
    mask: torch.Tensor,
    rasterizer,
) -> torch.Tensor:
    pred_sgm = rasterizer(boundary, 224, 224)
    pred_sgm[pred_sgm == -1] = 0
    pred_sgm = pred_sgm[index]
    boundary = boundary[index]
    mask = mask[index]
    intersection = pred_sgm * mask
    intersection = intersection.sum()
    union = pred_sgm.sum() + mask.sum() - intersection.sum()
    iou = intersection / union
    return iou


def get_batch_average_bou_iou(
    boundary: torch.Tensor,
    mask: torch.Tensor,
    rasterizer,
) -> torch.Tensor:
    with torch.no_grad():
        pred_sgm = rasterizer(boundary, 224, 224)
        pred_sgm[pred_sgm == -1] = 0
        pred_sgm = pred_sgm.flatten(1)
        mask = mask.flatten(1)
        intersection = pred_sgm * mask
        intersection = intersection.sum(-1)
        union = pred_sgm.sum(-1) + mask.sum(-1) - intersection
        iou = intersection / union
        return iou.mean()
