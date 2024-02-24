import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchsummary import summary
import numpy as np


class Model(nn.Module):
    def __init__(self, boundary_num=80):
        super(Model, self).__init__()
        res50 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.res50_bone = nn.Sequential(*list(res50.children())[:-2])
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=4098,
                nhead=6,
                batch_first=True,
            ),
            num_layers=4,
        )
        self.fc_list = nn.ModuleList()
        for i in range(boundary_num):
            self.fc_list.append(nn.Linear(4098, 2))
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
                bou_features = img_features[0, :, boundary[0, :, 0], boundary[0, :, 1]].unsqueeze(0)
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
        pre_bou_features = pre_bou_features.permute(0, 2, 1)
        curr_bou_features = get_bou_features(curr_img_features, previous_boundary)
        curr_bou_features = curr_bou_features.permute(0, 2, 1)

        tokens = torch.cat([curr_bou_features, pre_bou_features], dim=2)
        # tokens = torch.cat([tokens, previous_boundary.float() / 224], dim=2)
        tokens = torch.cat([tokens, previous_boundary.float()], dim=2)

        tokens = self.transformer_encoder(tokens)

        results = []
        for i in range(self.boundary_num):
            results.append(self.fc_list[i](tokens[:, i, :]))
        # return torch.stack(results, dim=1) * 224
        return torch.stack(results, dim=1)