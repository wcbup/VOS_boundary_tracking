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
        nhead=1,
        boundary_num=80,
    ):
        super(Model, self).__init__()
        res50 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.res50_bone = nn.Sequential(*list(res50.children())[:-3])
        # freeze resnet50
        for param in self.res50_bone.parameters():
            param.requires_grad = False
        self.positional_embedding = PositionalEncoding(d_token)
        # assert (d_token - 2) % 2 == 0
        # self.boundary_embedding = nn.Sequential(
        #     nn.LayerNorm(1024),
        #     nn.Linear(1024, (d_token - 2) // 2),
        #     nn.LayerNorm((d_token - 2) // 2),
        # )
        # self.layernorm = nn.LayerNorm(d_token)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_token,
                nhead=nhead,
                batch_first=True,
            ),
            num_layers=1,
        )
        self.fc_list = nn.ModuleList()
        for i in range(boundary_num):
            self.fc_list.append(
                nn.Sequential(
                    nn.Linear(d_token, 2),
                )
            )
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

        # tokens = self.layernorm(tokens)
        tokens = self.positional_embedding(tokens)
        tokens = self.transformer_encoder(tokens)

        results = []
        for i in range(self.boundary_num):
            results.append(self.fc_list[i](tokens[:, i, :]))
        # return torch.stack(results, dim=1) * 224
        return torch.stack(results, dim=1)


class NeighborModel(nn.Module):
    def __init__(
        self,
    ):
        super(NeighborModel, self).__init__()
        res50 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.res50_bone = nn.Sequential(*list(res50.children())[:-3])
        # freeze resnet50
        for param in self.res50_bone.parameters():
            param.requires_grad = False

        d_token = 1024 + 7 * 7 * 4 + 2
        self.layernorm = nn.LayerNorm(d_token)
        self.positional_embedding = PositionalEncoding(d_token)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_token,
                nhead=1,
                batch_first=True,
            ),
            num_layers=1,
        )
        self.boundary_num = 80
        self.fc_list = nn.ModuleList()
        for i in range(self.boundary_num):
            self.fc_list.append(
                nn.Sequential(
                    nn.Linear(d_token, 2 + 1024),
                )
            )

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

        def get_neighbor_dot(
            query_features: torch.Tensor,
            boundary: torch.Tensor,
            cur_img_freature: torch.Tensor,
            scale_level: int,
        ) -> torch.Tensor:
            def get_dot_product(
                query_features: torch.Tensor, key_features: torch.Tensor
            ) -> torch.Tensor:
                return torch.matmul(
                    query_features.transpose(1, 2).unsqueeze(2),
                    key_features.transpose(1, 2).unsqueeze(2).transpose(2, 3),
                ).squeeze()

            if scale_level > 0:
                cur_img_freature = F.avg_pool2d(
                    cur_img_freature, 2**scale_level, 2**scale_level
                )
                boundary = boundary // (2**scale_level)
            device = query_features.device

            cur_img_freature = F.pad(cur_img_freature, (3, 3, 3, 3), "constant", 0)
            dot_results = []
            for x_offset in range(-3, 4):
                for y_offset in range(-3, 4):
                    dot_results.append(
                        get_dot_product(
                            query_features,
                            get_bou_features(
                                cur_img_freature,
                                boundary
                                + torch.tensor([x_offset, y_offset]).to(device),
                            ),
                        )
                    )
            return torch.stack(dot_results, dim=2)

        curr_boundary = previous_boundary
        results = []
        refine_num = 6
        for i in range(refine_num):
            query_features = get_bou_features(curr_img_features, curr_boundary)
            dot_patches = []
            for scale_level in range(4):
                dot_patches.append(
                    get_neighbor_dot(
                        query_features,
                        curr_boundary,
                        curr_img_features,
                        scale_level,
                    )
                )
            dot_patches = torch.cat(dot_patches, dim=2)
            tokens = torch.cat(
                [
                    query_features.permute(0, 2, 1),
                    dot_patches,
                    curr_boundary.float(),
                ],
                dim=2,
            )
            tokens = self.layernorm(tokens)
            tokens = self.positional_embedding(tokens)
            tokens = self.transformer_encoder(tokens)
            output = []
            for j in range(self.boundary_num):
                output.append(self.fc_list[j](tokens[:, j, :]))
            output = torch.stack(output, dim=1)
            boundary_offset = output[:, :, :2]
            query_offset = output[:, :, 2:]
            curr_boundary = curr_boundary + boundary_offset.long()
            curr_boundary.clamp(min=0, max=223)
            query_features = query_features + query_offset.transpose(1, 2)
            results.append(curr_boundary)
        return results
