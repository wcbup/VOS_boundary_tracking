import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchsummary import summary
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, dropout=0.1, max_seq_len=512) -> None:
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


class Model(nn.Module):
    def __init__(
        self,
        d_token=2050,
        nhead=50,
        boundary_num=80,
    ):
        super(Model, self).__init__()
        res50 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.res50_bone = nn.Sequential(*list(res50.children())[:-3])
        self.positional_embedding = PositionalEncoding(d_token)
        # assert (d_token - 2) % 2 == 0
        # self.boundary_embedding = nn.Sequential(
        #     nn.LayerNorm(1024),
        #     nn.Linear(1024, (d_token - 2) // 2),
        #     nn.LayerNorm((d_token - 2) // 2),
        # )
        self.layernorm = nn.LayerNorm(d_token)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_token,
                nhead=nhead,
                batch_first=True,
            ),
            num_layers=6,
        )
        self.fc_list = nn.ModuleList()
        for i in range(boundary_num):
            self.fc_list.append(nn.Linear(d_token, 2))
        self.boundary_num = boundary_num

    def forward(
        self,
        previous_frame: torch.Tensor,
        current_frame: torch.Tensor,
        previous_boundary: torch.Tensor,
    ) -> torch.Tensor:
        pre_img_features = self.res50_bone(previous_frame)
        pre_img_features = F.interpolate(
            pre_img_features,
            size=(224, 224),
            mode="bilinear",
        )
        curr_img_features = self.res50_bone(current_frame)
        curr_img_features = F.interpolate(
            curr_img_features,
            size=(224, 224),
            mode="bilinear",
        )

        def get_bou_features(
            img_features: torch.Tensor, boundary: torch.Tensor
        ) -> torch.Tensor:
            bou_features = img_features[
                0, :, boundary[0, :, 0], boundary[0, :, 1]
            ].unsqueeze(0)
            for i in range(1, boundary.shape[0]):
                bou_features = torch.cat(
                    (
                        bou_features,
                        img_features[
                            i,
                            :,
                            boundary[i, :, 0],
                            boundary[i, :, 1],
                        ].unsqueeze(0),
                    ),
                    dim=0,
                )
            return bou_features

        pre_bou_features = get_bou_features(pre_img_features, previous_boundary)
        curr_bou_features = get_bou_features(curr_img_features, previous_boundary)
        pre_bou_features = pre_bou_features.permute(0, 2, 1)
        curr_bou_features = curr_bou_features.permute(0, 2, 1)
        # pre_bou_features = self.boundary_embedding(pre_bou_features)
        # curr_bou_features = self.boundary_embedding(curr_bou_features)

        tokens = torch.cat([curr_bou_features, pre_bou_features], dim=2)
        # tokens = torch.cat([tokens, previous_boundary.float() / 224], dim=2)
        tokens = torch.cat([tokens, previous_boundary.float()], dim=2)

        tokens = self.layernorm(tokens)
        tokens = self.positional_embedding(tokens)
        tokens = self.transformer_encoder(tokens)

        results = []
        for i in range(self.boundary_num):
            results.append(self.fc_list[i](tokens[:, i, :]))
        # return torch.stack(results, dim=1) * 224
        return torch.stack(results, dim=1)
