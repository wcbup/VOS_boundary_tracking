import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchsummary import summary
import numpy as np
from einops import rearrange


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, dropout=0.1, max_seq_len=1024) -> None:
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


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        f_dim = 512

        # encoder (downsampling)
        self.enc_conv0 = nn.Conv2d(3, f_dim, 3, padding=1)
        self.pool0 = nn.MaxPool2d(2, 2)  # 224 -> 112
        self.enc_conv1 = nn.Conv2d(f_dim, f_dim, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)  # 112 -> 56
        self.enc_conv2 = nn.Conv2d(f_dim, f_dim, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # 56 -> 28
        self.enc_conv3 = nn.Conv2d(f_dim, f_dim, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)  # 28 -> 14

        # bottleneck
        self.bottleneck_conv = nn.Conv2d(f_dim, f_dim, 3, padding=1)

        # decoder (upsampling)
        self.upsample0 = nn.Upsample(28)  # 14 -> 28
        self.dec_conv0 = nn.Conv2d(f_dim * 2, f_dim, 3, padding=1)
        self.upsample1 = nn.Upsample(56)  # 28 -> 56
        self.dec_conv1 = nn.Conv2d(f_dim * 2, f_dim, 3, padding=1)
        self.upsample2 = nn.Upsample(112)  # 56 -> 112
        self.dec_conv2 = nn.Conv2d(f_dim * 2, f_dim, 3, padding=1)
        self.upsample3 = nn.Upsample(224)  # 112 -> 224
        self.dec_conv3 = nn.Conv2d(f_dim * 2, f_dim * 2, 3, padding=1)

    def forward(self, x):
        # encoder
        e0 = F.relu(self.enc_conv0(x))
        e1 = F.relu(self.enc_conv1(self.pool0(e0)))
        e2 = F.relu(self.enc_conv2(self.pool1(e1)))
        e3 = F.relu(self.enc_conv3(self.pool2(e2)))
        # e2 = self.pool2(F.relu(self.enc_conv2(e1)))
        # e3 = self.pool3(F.relu(self.enc_conv3(e2)))

        # bottleneck
        b = F.relu(self.bottleneck_conv(self.pool3(e3)))

        # decoder
        # d0 = F.relu(self.dec_conv0(self.upsample0(b)))
        # d1 = F.relu(self.dec_conv1(self.upsample1(d0)))
        # d2 = F.relu(self.dec_conv2(self.upsample2(d1)))
        # d3 = self.dec_conv3(self.upsample3(d2))  # no activation
        # print(self.upsample0(b).shape)
        # print(e0.shape)
        # print(e1.shape)
        # print(e2.shape)
        # print(e3.shape)
        # print(torch.cat([e3, self.upsample0(b)], 1).shape)
        d0 = F.relu(self.dec_conv0(torch.cat([e3, self.upsample0(b)], 1)))
        d1 = F.relu(self.dec_conv1(torch.cat([e2, self.upsample1(d0)], 1)))
        d2 = F.relu(self.dec_conv2(torch.cat([e1, self.upsample2(d1)], 1)))
        d3 = self.dec_conv3(torch.cat([e0, self.upsample3(d2)], 1))
        return d3


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
        # self.fc_list = nn.ModuleList()
        # for i in range(self.boundary_num):
        #     self.fc_list.append(
        #         nn.Sequential(
        #             nn.Linear(d_token, 2 + 1024),
        #         )
        #     )
        self.output_encoder = nn.Sequential(
            nn.Linear(d_token, 2 + 1024),
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
                ).squeeze((2, 3))

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

        curr_boundary = previous_boundary.float()
        results = []
        refine_num = 6
        query_features = get_bou_features(curr_img_features, previous_boundary)
        for i in range(refine_num):
            dot_patches = []
            for scale_level in range(4):
                dot_patches.append(
                    get_neighbor_dot(
                        query_features,
                        curr_boundary.long(),
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
            # output = []
            # for j in range(self.boundary_num):
            #     output.append(self.fc_list[j](tokens[:, j, :]))
            # output = torch.stack(output, dim=1)
            output = self.output_encoder(tokens)
            boundary_offset = output[:, :, :2]
            query_offset = output[:, :, 2:]
            curr_boundary = curr_boundary + boundary_offset
            curr_boundary = curr_boundary.clamp(min=0, max=223)
            query_features = query_features + query_offset.transpose(1, 2)
            results.append(curr_boundary)
        return results


class IterativeModel(nn.Module):
    def __init__(self):
        super(IterativeModel, self).__init__()
        d_token = 1024 + 2
        self.boundary_num = 80
        res50 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.res50_bone = nn.Sequential(*list(res50.children())[:-3])
        # freeze resnet50
        for param in self.res50_bone.parameters():
            param.requires_grad = False
        self.query_encoder = nn.Sequential(
            nn.Linear(1024, d_token),
            nn.LayerNorm(d_token),
        )
        self.positional_embedding = PositionalEncoding(d_token)
        self.layer_norm = nn.LayerNorm(d_token)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_token,
                nhead=1,
                batch_first=True,
            ),
            num_layers=1,
        )
        self.q_offset_fc = nn.Sequential(
            nn.Linear(d_token, d_token),
        )
        self.boundary_fc = nn.Sequential(
            nn.Linear(d_token, 2),
        )
        self.refine_num = 3

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

        curr_boundary = previous_boundary.float()
        raw_query_features = get_bou_features(pre_img_features, previous_boundary)
        query_features = self.query_encoder(raw_query_features.permute(0, 2, 1))
        results = []
        for i in range(self.refine_num):
            boundary_features = get_bou_features(
                curr_img_features, curr_boundary.long()
            )
            boundary_features = boundary_features.permute(0, 2, 1)
            boundary_tokens = torch.cat([boundary_features, curr_boundary], dim=2)
            tokens = torch.cat([query_features, boundary_tokens], dim=1)
            tokens = self.layer_norm(tokens)
            tokens = self.positional_embedding(tokens)
            tokens = self.transformer_encoder(tokens)
            query_offset = tokens[:, : self.boundary_num, :]
            query_offset = self.q_offset_fc(query_offset)
            boundary_offset = tokens[:, self.boundary_num :, :]
            boundary_offset = self.boundary_fc(boundary_offset)
            curr_boundary = curr_boundary + boundary_offset
            curr_boundary = curr_boundary.clamp(min=0, max=223)
            query_features = query_features + query_offset
            results.append(curr_boundary)
        return results


class IterativeModel_Con(nn.Module):
    def __init__(self):
        super(IterativeModel_Con, self).__init__()
        d_token = 1024 * 2 + 2
        self.boundary_num = 80
        res50 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.res50_bone = nn.Sequential(*list(res50.children())[:-3])
        # freeze resnet50
        for param in self.res50_bone.parameters():
            param.requires_grad = False
        self.positional_embedding = PositionalEncoding(d_token)
        self.layer_norm = nn.LayerNorm(d_token)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_token,
                nhead=1,
                batch_first=True,
            ),
            num_layers=1,
        )
        self.output_encoder = nn.Sequential(
            nn.Linear(d_token, 1024 + 2),
        )
        self.refine_num = 3

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

        curr_boundary = previous_boundary.float()
        query_features = get_bou_features(pre_img_features, previous_boundary)
        query_features = query_features.permute(0, 2, 1)
        results = []
        for i in range(self.refine_num):
            boundary_features = get_bou_features(
                curr_img_features, curr_boundary.long()
            )
            boundary_features = boundary_features.permute(0, 2, 1)
            boundary_tokens = torch.cat([boundary_features, curr_boundary], dim=2)
            tokens = torch.cat([query_features, boundary_tokens], dim=2)
            tokens = self.layer_norm(tokens)
            tokens = self.positional_embedding(tokens)
            tokens = self.transformer_encoder(tokens)
            output = self.output_encoder(tokens)
            query_offset = output[:, :, :1024]
            boundary_offset = output[:, :, 1024:]
            curr_boundary = curr_boundary + boundary_offset
            curr_boundary = curr_boundary.clamp(min=0, max=223)
            query_features = query_features + query_offset
            results.append(curr_boundary)
        return results


def find_best_shift(boundary0: torch.Tensor, boundary1: torch.Tensor):
    def get_one_best_shift(boundary0: torch.Tensor, boundary1: torch.Tensor):
        best_shift = 0
        min_distance = (boundary0 - boundary1).abs().sum().item()
        for i in range(boundary0.shape[0]):
            distance = (boundary0 - boundary1.roll(i, 0)).abs().sum().item()
            if distance < min_distance:
                min_distance = distance
                best_shift = i
        return best_shift

    results = []
    for i in range(boundary0.shape[0]):
        results.append(get_one_best_shift(boundary0[i], boundary1[i]))

    return results


class IterativeModelWithFirst(nn.Module):
    def __init__(
        self,
        is_update=False,
    ):
        super(IterativeModelWithFirst, self).__init__()
        self.is_update = is_update
        d_token = 1024 + 2
        self.boundary_num = 80
        res50 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.res50_bone = nn.Sequential(*list(res50.children())[:-3])
        # freeze resnet50
        for param in self.res50_bone.parameters():
            param.requires_grad = False
        self.query_encoder = nn.Sequential(
            nn.Linear(1024, d_token),
            nn.LayerNorm(d_token),
        )
        self.positional_embedding = PositionalEncoding(d_token)
        self.layer_norm = nn.LayerNorm(d_token)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_token,
                nhead=1,
                batch_first=True,
            ),
            num_layers=1,
        )
        self.q_offset_fc = nn.Sequential(
            nn.Linear(d_token, d_token),
        )
        self.boundary_fc = nn.Sequential(
            nn.Linear(d_token, 2),
        )
        self.refine_num = 3

    def forward(
        self,
        first_frame: torch.Tensor,
        first_boundary: torch.Tensor,
        previous_frame: torch.Tensor,
        current_frame: torch.Tensor,
        previous_boundary: torch.Tensor,
    ) -> torch.Tensor:
        best_shift = find_best_shift(previous_boundary, first_boundary)
        for i in range(len(best_shift)):
            first_boundary[i] = first_boundary[i].roll(best_shift[i], 0)
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
        first_img_features = self.res50_bone(first_frame)
        first_img_features = F.interpolate(
            first_img_features,
            size=(224, 224),
            mode="bilinear",
        )

        curr_boundary = previous_boundary.float()
        raw_query_features = get_bou_features(pre_img_features, previous_boundary)
        query_features = self.query_encoder(raw_query_features.permute(0, 2, 1))
        first_queary_features = get_bou_features(first_img_features, first_boundary)
        first_queary_features = self.query_encoder(
            first_queary_features.permute(0, 2, 1)
        )
        results = []
        for i in range(self.refine_num):
            boundary_features = get_bou_features(
                curr_img_features, curr_boundary.long()
            )
            boundary_features = boundary_features.permute(0, 2, 1)
            boundary_tokens = torch.cat([boundary_features, curr_boundary], dim=2)
            tokens = torch.cat(
                [query_features, boundary_tokens, first_queary_features], dim=1
            )
            tokens = self.layer_norm(tokens)
            tokens = self.positional_embedding(tokens)
            tokens = self.transformer_encoder(tokens)
            query_offset = tokens[:, : self.boundary_num, :]
            query_offset = self.q_offset_fc(query_offset)
            boundary_offset = tokens[:, self.boundary_num : 2 * self.boundary_num, :]
            boundary_offset = self.boundary_fc(boundary_offset)
            curr_boundary = curr_boundary + boundary_offset
            curr_boundary = curr_boundary.clamp(min=0, max=223)
            query_features = query_features + query_offset
            if self.is_update:
                first_offset = tokens[:, 2 * self.boundary_num :, :]
                first_queary_features = first_queary_features + first_offset
            results.append(curr_boundary)
        return results


class IterativeModelWithFirst_Nei(nn.Module):
    def __init__(
        self,
        is_update=False,
    ):
        super(IterativeModelWithFirst_Nei, self).__init__()
        self.is_update = is_update
        d_token = 1024 + 2
        self.boundary_num = 80
        res50 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.res50_bone = nn.Sequential(*list(res50.children())[:-3])
        # freeze resnet50
        for param in self.res50_bone.parameters():
            param.requires_grad = False
        self.query_encoder = nn.Sequential(
            nn.Linear(1024, d_token),
            nn.LayerNorm(d_token),
        )
        self.positional_embedding = PositionalEncoding(d_token)
        self.layer_norm = nn.LayerNorm(d_token)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_token,
                nhead=1,
                batch_first=True,
            ),
            num_layers=1,
        )
        self.q_offset_fc = nn.Sequential(
            nn.Linear(d_token, d_token),
        )
        self.boundary_fc = nn.Sequential(
            nn.Linear(d_token, 2),
        )
        self.refine_num = 3
        self.neighbor_levels = [0, 2]
        self.neighbor_encoder = nn.Sequential(
            nn.Linear(1024 * 9 * len(self.neighbor_levels), d_token),
            nn.LayerNorm(d_token),
        )

    def forward(
        self,
        first_frame: torch.Tensor,
        first_boundary: torch.Tensor,
        previous_frame: torch.Tensor,
        current_frame: torch.Tensor,
        previous_boundary: torch.Tensor,
    ) -> torch.Tensor:
        best_shift = find_best_shift(previous_boundary, first_boundary)
        for i in range(len(best_shift)):
            first_boundary[i] = first_boundary[i].roll(best_shift[i], 0)
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
        first_img_features = self.res50_bone(first_frame)
        first_img_features = F.interpolate(
            first_img_features,
            size=(224, 224),
            mode="bilinear",
        )

        def get_neighbor_features_with_scales(
            img_features: torch.Tensor, boundary: torch.Tensor, scale_levels: list[int]
        ):
            def get_neighbor_features(
                img_features: torch.Tensor, boundary: torch.Tensor, scale_level: int
            ):
                device = img_features.device
                if scale_level > 0:
                    img_features = F.avg_pool2d(
                        img_features,
                        2**scale_level,
                        2**scale_level,
                    )
                    boundary = boundary // (2**scale_level)
                img_features = F.pad(img_features, (1, 1, 1, 1), "constant", 0)
                neighor_features = []
                for x_offset in range(-1, 2):
                    for y_offset in range(-1, 2):
                        neighor_features.append(
                            get_bou_features(
                                img_features,
                                boundary
                                + torch.tensor([x_offset, y_offset]).to(device),
                            )
                        )
                return torch.cat(neighor_features, dim=1)

            neighbor_features = get_neighbor_features(
                img_features, boundary, scale_levels[0]
            )
            for scale_level in scale_levels[1:]:
                neighbor_features = torch.cat(
                    (
                        neighbor_features,
                        get_neighbor_features(img_features, boundary, scale_level),
                    ),
                    dim=1,
                )
            return neighbor_features

        curr_boundary = previous_boundary.float()
        raw_query_features = get_bou_features(pre_img_features, previous_boundary)
        query_features = self.query_encoder(raw_query_features.permute(0, 2, 1))
        first_queary_features = get_bou_features(first_img_features, first_boundary)
        first_queary_features = self.query_encoder(
            first_queary_features.permute(0, 2, 1)
        )
        results = []
        for i in range(self.refine_num):
            neighbor_features = get_neighbor_features_with_scales(
                curr_img_features,
                curr_boundary.long(),
                self.neighbor_levels,
            )
            neighbor_features = neighbor_features.permute(0, 2, 1)
            neighbor_features = self.neighbor_encoder(neighbor_features)
            boundary_features = get_bou_features(
                curr_img_features, curr_boundary.long()
            )
            boundary_features = boundary_features.permute(0, 2, 1)
            boundary_tokens = torch.cat([boundary_features, curr_boundary], dim=2)
            tokens = torch.cat(
                [
                    query_features,
                    boundary_tokens,
                    first_queary_features,
                    neighbor_features,
                ],
                dim=1,
            )
            tokens = self.layer_norm(tokens)
            tokens = self.positional_embedding(tokens)
            tokens = self.transformer_encoder(tokens)
            query_offset = tokens[:, : self.boundary_num, :]
            query_offset = self.q_offset_fc(query_offset)
            boundary_offset = tokens[:, self.boundary_num : 2 * self.boundary_num, :]
            boundary_offset = self.boundary_fc(boundary_offset)
            curr_boundary = curr_boundary + boundary_offset
            curr_boundary = curr_boundary.clamp(min=0, max=223)
            query_features = query_features + query_offset
            if self.is_update:
                first_offset = tokens[:, 2 * self.boundary_num :, :]
                first_queary_features = first_queary_features + first_offset
            results.append(curr_boundary)
        return results


class IterativeModelWithFirst_Con(nn.Module):
    def __init__(
        self,
    ):
        super(IterativeModelWithFirst_Con, self).__init__()
        d_token = 1024 * 3 + 2
        self.boundary_num = 80
        res50 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.res50_bone = nn.Sequential(*list(res50.children())[:-3])
        # freeze resnet50
        for param in self.res50_bone.parameters():
            param.requires_grad = False
        self.positional_embedding = PositionalEncoding(d_token)
        self.layer_norm = nn.LayerNorm(d_token)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_token,
                nhead=1,
                batch_first=True,
            ),
            num_layers=1,
        )
        self.output_encoder = nn.Sequential(
            nn.Linear(d_token, 1024 + 2),
        )
        self.refine_num = 3

    def forward(
        self,
        first_frame: torch.Tensor,
        first_boundary: torch.Tensor,
        previous_frame: torch.Tensor,
        current_frame: torch.Tensor,
        previous_boundary: torch.Tensor,
    ) -> torch.Tensor:
        best_shift = find_best_shift(previous_boundary, first_boundary)
        for i in range(len(best_shift)):
            first_boundary[i] = first_boundary[i].roll(best_shift[i], 0)
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
        first_img_features = self.res50_bone(first_frame)
        first_img_features = F.interpolate(
            first_img_features,
            size=(224, 224),
            mode="bilinear",
        )

        # def get_best_match(feature0: torch.Tensor, feature1: torch.Tensor):
        #     best_shift = 0
        #     best_similarity = (feature0 * feature1).sum()
        #     for shift in range(feature0.shape[1]):
        #         similarity = (feature0 * feature1.roll(shift, dims=1)).sum()
        #         if similarity > best_similarity:
        #             best_similarity = similarity
        #             best_shift = shift
        #     return feature0, feature1.roll(best_shift, dims=1)

        curr_boundary = previous_boundary.float()
        raw_query_features = get_bou_features(pre_img_features, previous_boundary)
        pre_query_features = raw_query_features.permute(0, 2, 1)
        first_queary_features = get_bou_features(first_img_features, first_boundary)
        first_queary_features = first_queary_features.permute(0, 2, 1)
        # pre_query_features, first_queary_features = get_best_match(pre_query_features, first_queary_features)
        results = []
        for i in range(self.refine_num):
            boundary_features = get_bou_features(
                curr_img_features, curr_boundary.long()
            )
            boundary_features = boundary_features.permute(0, 2, 1)
            boundary_tokens = torch.cat([boundary_features, curr_boundary], dim=2)
            tokens = torch.cat(
                [pre_query_features, boundary_tokens, first_queary_features], dim=2
            )
            tokens = self.layer_norm(tokens)
            tokens = self.positional_embedding(tokens)
            tokens = self.transformer_encoder(tokens)
            output = self.output_encoder(tokens)
            query_offset = output[:, :, :1024]
            boundary_offset = output[:, :, 1024:]
            curr_boundary = curr_boundary + boundary_offset
            curr_boundary = curr_boundary.clamp(min=0, max=223)
            pre_query_features = pre_query_features + query_offset
            results.append(curr_boundary)
        return results


def get_img_tokens(img_features: torch.Tensor) -> torch.Tensor:
    img_tokens = rearrange(
        img_features,
        "b c h w -> b (h w) c",
    )
    return img_tokens


class IterWholeFirst(nn.Module):
    def __init__(self, add_xy_in_token=False, use_unet=False):
        super(IterWholeFirst, self).__init__()

        self.add_xy_in_token = add_xy_in_token
        self.boundary_num = 80
        self.refine_num = 3
        res50 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.res50_bone = nn.Sequential(*list(res50.children())[:-3])
        # freeze resnet50
        for param in self.res50_bone.parameters():
            param.requires_grad = False

        d_token = 1024 + 2
        self.positional_embedding = PositionalEncoding(d_token)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=1,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=1,
        )
        self.img_token_fc = nn.Sequential(
            nn.Linear(1024, d_token),
            nn.LayerNorm(d_token),
        )

        if not add_xy_in_token:
            self.q_token_fc = nn.Sequential(
                nn.Linear(1024, d_token),
                nn.LayerNorm(d_token),
            )
        self.use_unet = use_unet
        if use_unet:
            self.unet = UNet()
        self.layernorm = nn.LayerNorm(d_token)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_token,
            nhead=1,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=1,
        )
        if not add_xy_in_token:
            self.q_offset_fc = nn.Sequential(
                nn.Linear(d_token, d_token),
            )
        self.xy_offset_fc = nn.Sequential(
            nn.Linear(d_token, 2),
        )

    def forward(
        self,
        first_frame: torch.Tensor,
        first_boundary: torch.Tensor,
        previous_frame: torch.Tensor,
        current_frame: torch.Tensor,
        previous_boundary: torch.Tensor,
    ) -> torch.Tensor:
        best_shift = find_best_shift(previous_boundary, first_boundary)
        for i in range(len(best_shift)):
            first_boundary[i] = first_boundary[i].roll(best_shift[i], 0)
        pre_raw_img_features = self.res50_bone(previous_frame)
        first_raw_img_features = self.res50_bone(first_frame)
        curr_raw_img_features = self.res50_bone(current_frame)

        pre_img_tokens = get_img_tokens(pre_raw_img_features)
        cur_img_tokens = get_img_tokens(curr_raw_img_features)
        first_img_tokens = get_img_tokens(first_raw_img_features)

        img_tokens = torch.cat(
            [pre_img_tokens, cur_img_tokens, first_img_tokens], dim=1
        )
        img_tokens = self.img_token_fc(img_tokens)
        img_tokens = self.positional_embedding(img_tokens)
        memory = self.transformer_encoder(img_tokens)

        if self.use_unet:
            first_img_features = self.unet(first_frame)
            pre_img_features = self.unet(previous_frame)
            cur_img_feature = self.unet(current_frame)
        else:
            first_img_features = F.interpolate(
                first_raw_img_features,
                size=(224, 224),
                mode="bilinear",
            )
            pre_img_features = F.interpolate(
                pre_raw_img_features,
                size=(224, 224),
                mode="bilinear",
            )
            cur_img_feature = F.interpolate(
                curr_raw_img_features,
                size=(224, 224),
                mode="bilinear",
            )

        cur_boundary = previous_boundary.float()
        first_query = get_bou_features(first_img_features, first_boundary)
        pre_query = get_bou_features(pre_img_features, previous_boundary)

        if self.add_xy_in_token:
            first_query = first_query.permute(0, 2, 1)
            pre_query = pre_query.permute(0, 2, 1)
            first_query = torch.cat([first_query, first_boundary.float()], dim=2)
            pre_query = torch.cat([pre_query, previous_boundary.float()], dim=2)
        else:
            first_query = self.q_token_fc(first_query.permute(0, 2, 1))
            pre_query = self.q_token_fc(pre_query.permute(0, 2, 1))

        results = []
        for _ in range(self.refine_num):
            cur_boundary = cur_boundary.clamp(min=0, max=223)
            boundary_features = get_bou_features(cur_img_feature, cur_boundary.long())
            boundary_features = boundary_features.permute(0, 2, 1)
            boundary_tokens = torch.cat([boundary_features, cur_boundary], dim=2)
            input_tokens = torch.cat([pre_query, boundary_tokens, first_query], dim=1)
            input_tokens = self.layernorm(input_tokens)
            input_tokens = self.positional_embedding(input_tokens)
            output_tokens = self.transformer_decoder(
                input_tokens,
                memory,
            )

            if not self.add_xy_in_token:
                query_offset = output_tokens[:, : self.boundary_num, :]
                query_offset = self.q_offset_fc(query_offset)
                pre_query = pre_query + query_offset

            boundary_offset = output_tokens[
                :, self.boundary_num : 2 * self.boundary_num, :
            ]
            boundary_offset = self.xy_offset_fc(boundary_offset)
            cur_boundary = cur_boundary + boundary_offset
            results.append(cur_boundary)
        return results


class IterWholeFirst_Con(nn.Module):
    def __init__(self):
        super(IterWholeFirst_Con, self).__init__()

        self.refine_num = 3
        res50 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.res50_bone = nn.Sequential(*list(res50.children())[:-3])
        # freeze resnet50
        for param in self.res50_bone.parameters():
            param.requires_grad = False

        d_token = (1024 + 2) * 3
        head_num = 1
        self.positional_embedding = PositionalEncoding(d_token)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=head_num,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=1,
        )
        self.output_encoder = nn.Sequential(
            nn.Linear(d_token, 2),
        )
        self.query_encoder = nn.Sequential(
            # nn.Linear((1024 + 2) * 3, d_token),
            nn.LayerNorm(d_token),
        )
        self.layer_norm = nn.LayerNorm(d_token)
        self.img_token_fc = nn.Sequential(
            nn.Linear(1024, d_token),
            nn.LayerNorm(d_token),
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_token,
            nhead=head_num,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=1,
        )

    def forward(
        self,
        first_frame: torch.Tensor,
        first_boundary: torch.Tensor,
        previous_frame: torch.Tensor,
        current_frame: torch.Tensor,
        previous_boundary: torch.Tensor,
    ) -> torch.Tensor:
        best_shift = find_best_shift(previous_boundary, first_boundary)
        for i in range(len(best_shift)):
            first_boundary[i] = first_boundary[i].roll(best_shift[i], 0)
        pre_raw_img_features = self.res50_bone(previous_frame)
        first_raw_img_features = self.res50_bone(first_frame)
        curr_raw_img_features = self.res50_bone(current_frame)

        pre_img_tokens = get_img_tokens(pre_raw_img_features)
        cur_img_tokens = get_img_tokens(curr_raw_img_features)
        first_img_tokens = get_img_tokens(first_raw_img_features)

        img_tokens = torch.cat(
            [pre_img_tokens, cur_img_tokens, first_img_tokens], dim=1
        )
        img_tokens = self.img_token_fc(img_tokens)
        img_tokens = self.positional_embedding(img_tokens)
        memory = self.transformer_encoder(img_tokens)

        first_img_features = F.interpolate(
            first_raw_img_features,
            size=(224, 224),
            mode="bilinear",
        )
        pre_img_features = F.interpolate(
            pre_raw_img_features,
            size=(224, 224),
            mode="bilinear",
        )
        cur_img_feature = F.interpolate(
            curr_raw_img_features,
            size=(224, 224),
            mode="bilinear",
        )

        cur_boundary = previous_boundary.float()
        first_query = get_bou_features(first_img_features, first_boundary)
        pre_query = get_bou_features(pre_img_features, previous_boundary)
        first_query = first_query.permute(0, 2, 1)
        pre_query = pre_query.permute(0, 2, 1)

        first_query = torch.cat([first_query, first_boundary.float()], dim=2)
        pre_query = torch.cat([pre_query, previous_boundary.float()], dim=2)

        results = []
        for _ in range(self.refine_num):
            cur_boundary = cur_boundary.clamp(min=0, max=223)
            boundary_features = get_bou_features(cur_img_feature, cur_boundary.long())
            boundary_features = boundary_features.permute(0, 2, 1)
            boundary_tokens = torch.cat([boundary_features, cur_boundary], dim=2)

            input_tokens = torch.cat([pre_query, boundary_tokens, first_query], dim=2)
            input_tokens = self.query_encoder(input_tokens)
            input_tokens = self.positional_embedding(input_tokens)

            output_tokens = self.transformer_decoder(
                input_tokens,
                memory,
            )

            boundary_offset = self.output_encoder(output_tokens)
            cur_boundary = cur_boundary + boundary_offset
            results.append(cur_boundary)
        return results


class BaseFeatup(nn.Module):
    def __init__(self):
        super(BaseFeatup, self).__init__()
        self.refine_num = 3
        self.boundary_num = 80

        d_token = 384 + 2
        self.backbone = torch.hub.load(
            "mhamilton723/FeatUp",
            "dino16",
            use_norm=True,
        )
        # freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.img_token_fc = nn.Sequential(
            nn.Linear(384, d_token),
            nn.LayerNorm(d_token),
        )
        self.layernorm = nn.LayerNorm(d_token)
        self.positional_embedding = PositionalEncoding(d_token)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=1,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=1,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_token,
            nhead=1,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=1,
        )
        self.xy_offset_fc = nn.Sequential(
            nn.Linear(d_token, 2),
        )

    def forward(
        self,
        first_frame: torch.Tensor,
        first_boundary: torch.Tensor,
        previous_frame: torch.Tensor,
        current_frame: torch.Tensor,
        previous_boundary: torch.Tensor,
    ) -> torch.Tensor:
        best_shift = find_best_shift(previous_boundary, first_boundary)
        for i in range(len(best_shift)):
            first_boundary[i] = first_boundary[i].roll(best_shift[i], 0)

        pre_raw_img_feats = self.backbone.model(previous_frame)
        first_raw_img_feats = self.backbone.model(first_frame)
        cur_raw_img_feats = self.backbone.model(current_frame)

        pre_img_tokens = get_img_tokens(pre_raw_img_feats)
        cur_img_tokens = get_img_tokens(cur_raw_img_feats)
        first_img_tokens = get_img_tokens(first_raw_img_feats)

        img_tokens = torch.cat(
            [pre_img_tokens, cur_img_tokens, first_img_tokens], dim=1
        )
        img_tokens = self.img_token_fc(img_tokens)
        img_tokens = self.layernorm(img_tokens)
        img_tokens = self.positional_embedding(img_tokens)
        memory = self.transformer_encoder(img_tokens)

        pre_img_features = self.backbone(previous_frame)
        cur_img_features = self.backbone(current_frame)
        first_img_features = self.backbone(first_frame)

        pre_bou_feats = get_bou_features(pre_img_features, previous_boundary)
        first_bou_feats = get_bou_features(first_img_features, first_boundary)
        pre_bou_feats = pre_bou_feats.permute(0, 2, 1)
        first_bou_feats = first_bou_feats.permute(0, 2, 1)

        first_tokens = torch.cat([first_bou_feats, first_boundary.float()], dim=2)
        pre_tokens = torch.cat([pre_bou_feats, previous_boundary.float()], dim=2)

        results = []
        cur_boundary = previous_boundary.float()
        for _ in range(self.refine_num):
            cur_boundary = cur_boundary.clamp(min=0, max=223)
            cur_bou_feats = get_bou_features(cur_img_features, cur_boundary.long())
            cur_bou_feats = cur_bou_feats.permute(0, 2, 1)
            cur_tokens = torch.cat([cur_bou_feats, cur_boundary], dim=2)
            input_tokens = torch.cat([pre_tokens, cur_tokens, first_tokens], dim=1)
            input_tokens = self.layernorm(input_tokens)
            input_tokens = self.positional_embedding(input_tokens)
            output_tokens = self.transformer_decoder(input_tokens, memory)
            bou_offset = output_tokens[:, self.boundary_num : 2 * self.boundary_num, :]
            bou_offset = self.xy_offset_fc(bou_offset)
            cur_boundary = cur_boundary + bou_offset
            results.append(cur_boundary)
        return results
