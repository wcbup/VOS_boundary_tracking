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
import gc
from deform_model import (
    _get_clones,
    _get_activation_fn,
    get_valid_ratio,
    PositionalEncoding,
    DeformableTransformerEncoderLayer,
    DeformableTransformerDecoderLayer,
    DeformableTransformerEncoder,
    DeformableTransformerDecoder,
    MLP,
    get_bounding_box,
    add_mid_points,
    get_batch_average_bou_iou,
)
import math


class DAVIS_withPoint(Dataset):
    def __init__(
        self,
        raw_set: DAVIS_Rawset,
        point_num: int,
        is_train: bool,
    ) -> None:
        super().__init__()
        self.point_num = point_num
        # remove all the video with empty frame
        empty_video_idx = []
        for video_idx, video_data in enumerate(raw_set.data_set):
            for frame_data in video_data:
                img, mask = frame_data
                if mask.sum() == 0:
                    empty_video_idx.append(video_idx)
                    break
        self.raw_data_set = []
        if is_train:
            train_point_path = "sample_results/train_256_uniform.json"
        else:
            train_point_path = "sample_results/val_256_uniform.json"
        with open(train_point_path, "r") as f:
            points = json.load(f)
        for video_idx, video_data in enumerate(raw_set.data_set):
            if video_idx in empty_video_idx:
                continue
            self.raw_data_set.append([])
            for frame_idx, frame_data in enumerate(video_data):
                img, mask = frame_data
                point_data = points[video_idx][frame_idx][str(point_num)]["boundary"]
                point = torch.tensor(point_data)
                self.raw_data_set[-1].append((img, mask, point))
        self.data = []
        for video_idx, video_data in enumerate(self.raw_data_set):
            for frame_idx in range(len(video_data) - 1):
                self.data.append(
                    (
                        video_idx,
                        frame_idx,
                        video_data[0],
                        video_data[frame_idx],
                        video_data[frame_idx + 1],
                    )
                )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        video_idx, frame_idx, first_frame, previous_frame, current_frame = self.data[
            idx
        ]
        return video_idx, frame_idx, first_frame, previous_frame, current_frame


class DAVIS_withPointRandom(Dataset):
    def __init__(
        self,
        raw_set: DAVIS_Rawset,
        point_num: int,
        is_train: bool,
        sample_offset=10,
    ) -> None:
        super().__init__()
        self.sample_offset = sample_offset
        self.point_num = point_num
        # remove all the video with empty frame
        empty_video_idx = []
        for video_idx, video_data in enumerate(raw_set.data_set):
            for frame_data in video_data:
                img, mask = frame_data
                if mask.sum() == 0:
                    empty_video_idx.append(video_idx)
                    break
        self.raw_data_set = []
        if is_train:
            train_point_path = "sample_results/train_256_uniform.json"
        else:
            train_point_path = "sample_results/val_256_uniform.json"
        with open(train_point_path, "r") as f:
            points = json.load(f)
        for video_idx, video_data in enumerate(raw_set.data_set):
            if video_idx in empty_video_idx:
                continue
            self.raw_data_set.append([])
            for frame_idx, frame_data in enumerate(video_data):
                img, mask = frame_data
                point_data = points[video_idx][frame_idx][str(point_num)]["boundary"]
                point = torch.tensor(point_data)
                self.raw_data_set[-1].append((img, mask, point))

    def __len__(self):
        return len(self.raw_data_set)

    def __getitem__(self, idx: int):
        video_idx = idx
        video_data = self.raw_data_set[idx]
        first_frame = video_data[0]
        video_len = len(video_data)
        previous_idx = random.randint(0, video_len - 2)
        sample_low = previous_idx + 1
        sample_high = min(previous_idx + self.sample_offset, video_len - 1)
        current_idx = random.randint(sample_low, sample_high)
        previous_frame = video_data[previous_idx]
        current_frame = video_data[current_idx]
        return video_idx, previous_idx, first_frame, previous_frame, current_frame

        # video_idx, frame_idx, first_frame, previous_frame, current_frame = self.data[
        #     idx
        # ]
        # return video_idx, frame_idx, first_frame, previous_frame, current_frame


class DeformableTransformerExtraDecoderLayer(nn.Module):
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

        # extra cross attention
        self.extra_cross_attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.extra_norm = nn.LayerNorm(d_model)
        self.extra_dropout = nn.Dropout(dropout)

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
        extra_memory,
        src_padding_mask=None,
    ):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1)
        )[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # extra cross attention
        tgt_extra = self.extra_cross_attn(
            self.with_pos_embed(tgt, query_pos),
            extra_memory,
            extra_memory,
        )[0]
        tgt = tgt + self.extra_dropout(tgt_extra)
        tgt = self.extra_norm(tgt)

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


class DeformableTransformerExtraDecoder(nn.Module):
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
        extra_memory,
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
                extra_memory,
                src_padding_mask,
            )

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


class DeformVideo(nn.Module):
    def __init__(
        self,
        layer_num=1,
        up_scale_num=4,
        head_num=6,
        medium_level_size=[14, 28, 56, 112],
        offset_limit=56,
        n_points=4,
        mem_point_num=64,
        freeze_backbone=True,
    ) -> None:
        super(DeformVideo, self).__init__()
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

        enc_layer = DeformableTransformerEncoderLayer(
            d_model=d_model,
            d_ffn=d_ffn,
            n_levels=n_levels,
            n_heads=head_num,
            n_points=n_points,
        )
        dec_layer = DeformableTransformerDecoderLayer(
            d_model=d_model,
            d_ffn=d_ffn,
            n_levels=n_levels,
            n_heads=head_num,
            n_points=n_points,
        )

        self.first_query_embed = nn.Embedding(mem_point_num, d_model)
        xavier_uniform_(self.first_query_embed.weight)
        self.first_layer_norm = nn.LayerNorm(d_model)
        self.fir_enc = DeformableTransformerEncoder(
            enc_layer,
            num_layers=layer_num,
        )
        self.fir_dec = DeformableTransformerDecoder(
            dec_layer,
            num_layers=layer_num,
        )
        self.previous_query_embed = nn.Embedding(mem_point_num, d_model)
        xavier_uniform_(self.previous_query_embed.weight)
        self.previous_layer_norm = nn.LayerNorm(d_model)
        self.pre_enc = DeformableTransformerEncoder(
            enc_layer,
            num_layers=layer_num,
        )
        self.pre_dec = DeformableTransformerDecoder(
            dec_layer,
            num_layers=layer_num,
        )
        self.extra_layer_norm = nn.LayerNorm(d_model)

        query_num = 4
        for _ in range(up_scale_num):
            query_num *= 2
        self.current_query_embed = nn.Embedding(query_num, d_model)
        xavier_uniform_(self.current_query_embed.weight)
        self.current_layer_norm = nn.LayerNorm(d_model)
        self.cur_enc = DeformableTransformerEncoder(
            enc_layer,
            num_layers=layer_num,
        )
        extra_dec_layer = DeformableTransformerExtraDecoderLayer(
            d_model=d_model,
            d_ffn=d_ffn,
            n_levels=n_levels,
            n_heads=head_num,
            n_points=n_points,
        )
        extra_dec = DeformableTransformerExtraDecoder(
            extra_dec_layer,
            num_layers=layer_num,
        )
        self.extra_decs = _get_clones(extra_dec, up_scale_num + 1)
        xy_fc = MLP(d_model, d_model, 2, 3)
        self.xy_fcs = _get_clones(xy_fc, up_scale_num + 1)

    def forward(
        self,
        fir_img: torch.Tensor,
        fir_bou: torch.Tensor,
        pre_img: torch.Tensor,
        pre_bou: torch.Tensor,
        pre_sgm: torch.Tensor,
        cur_img: torch.Tensor,
    ):
        fir_bou = fir_bou / 224
        pre_bou = pre_bou / 224
        (
            fir_img_srcs_flatten,
            fir_spatial_shapes,
            fir_level_start_index,
            fir_valid_ratios,
        ) = self._get_enced_img_scrs(
            fir_img,
            self.fir_enc,
            self.first_layer_norm,
        )
        (
            pre_img_srcs_flatten,
            pre_spatial_shapes,
            pre_level_start_index,
            pre_valid_ratios,
        ) = self._get_enced_img_scrs(
            pre_img,
            self.pre_enc,
            self.previous_layer_norm,
        )
        (
            cur_img_srcs_flatten,
            cur_spatial_shapes,
            cur_level_start_index,
            cur_valid_ratios,
        ) = self._get_enced_img_scrs(
            cur_img,
            self.cur_enc,
            self.current_layer_norm,
        )

        B, S, C = fir_img_srcs_flatten.shape
        first_queries = (
            self.first_query_embed.weight.unsqueeze(0).repeat(B, 1, 1).cuda()
        )
        first_memory, _ = self.fir_dec(
            first_queries,
            fir_bou,
            fir_img_srcs_flatten,
            fir_spatial_shapes,
            fir_level_start_index,
            fir_valid_ratios,
        )
        previous_queries = (
            self.previous_query_embed.weight.unsqueeze(0).repeat(B, 1, 1).cuda()
        )
        previous_memory, _ = self.pre_dec(
            previous_queries,
            pre_bou,
            pre_img_srcs_flatten,
            pre_spatial_shapes,
            pre_level_start_index,
            pre_valid_ratios,
        )
        extra_memory = torch.cat([first_memory, previous_memory], 1)
        extra_memory = self.extra_layer_norm(extra_memory)
        extra_memory = self.pos_enoc(extra_memory)

        cur_queries = (
            self.current_query_embed.weight.unsqueeze(0).repeat(B, 1, 1).cuda()
        )
        init_bou = get_bounding_box(pre_sgm).cuda() / 224
        current_query_num = init_bou.shape[1]
        current_query = cur_queries[:, :current_query_num]
        decode_output, _ = self.extra_decs[0](
            current_query,
            init_bou,
            cur_img_srcs_flatten,
            cur_spatial_shapes,
            cur_level_start_index,
            cur_valid_ratios,
            extra_memory,
        )

        xy_offset = (
            (self.xy_fcs[0](decode_output).sigmoid() - 0.5) * self.offset_limit / 224
        )
        init_bou += xy_offset

        results = [init_bou]
        for i in range(self.up_scale_num):
            new_query = cur_queries[:, current_query_num : current_query_num * 2]
            current_query_num *= 2

            current_query = torch.zeros((B, current_query_num, C)).to(
                cur_img_srcs_flatten.device
            )
            current_query[:, ::2] = new_query
            current_query[:, 1::2] = decode_output

            cur_bou = add_mid_points(results[-1])
            decode_output, _ = self.extra_decs[i + 1](
                current_query,
                cur_bou,
                cur_img_srcs_flatten,
                cur_spatial_shapes,
                cur_level_start_index,
                cur_valid_ratios,
                extra_memory,
            )

            xy_offset = (
                (self.xy_fcs[i + 1](decode_output).sigmoid() - 0.5)
                * self.offset_limit
                / 224
            )
            cur_bou += xy_offset

            results.append(cur_bou)
        results = [result * 224 for result in results]

        if self.training:
            return results
        else:
            result = results[-1]
            result = result.clamp(0, 223)
            return result

    def _get_img_scrs(self, img: torch.Tensor, layernorm: nn.LayerNorm):
        feats = self.featup(img)
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
            [get_valid_ratio(mask) for mask in padding_masks],
            1,
        ).cuda()
        spatial_shapes = torch.as_tensor(
            spatial_shapes,
            dtype=torch.long,
            device=src_flatten.device,
        )
        src_flatten = layernorm(src_flatten)
        src_flatten = self.pos_enoc(src_flatten)
        return src_flatten, spatial_shapes, level_start_index, valid_ratios

    def _get_enced_img_scrs(
        self,
        img: torch.Tensor,
        encoder: DeformableTransformerEncoder,
        layernorm: nn.LayerNorm,
    ):
        src_flatten, spatial_shapes, level_start_index, valid_ratios = (
            self._get_img_scrs(img, layernorm)
        )
        src_flatten = encoder(
            src=src_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
        )
        return src_flatten, spatial_shapes, level_start_index, valid_ratios


class VideoInferer:
    def __init__(
        self,
        dataset: DAVIS_withPoint,
        gt_rasterizer: SoftPolygon,
    ) -> None:
        self.data_set = dataset.raw_data_set
        self.gt_rasterizer = gt_rasterizer

    def infer_one_video(self, video_idx: int, model: nn.Module):
        infer_results = []
        video_data = self.data_set[video_idx]
        model.eval()
        fir_img, fir_mask, fir_point = video_data[0]
        pre_img, pre_mask, pre_point = video_data[0]
        fir_img = fir_img.unsqueeze(0)
        pre_img = pre_img.unsqueeze(0)
        fir_mask = fir_mask.unsqueeze(0)
        pre_mask = pre_mask.unsqueeze(0)
        fir_point = fir_point.unsqueeze(0)
        pre_point = pre_point.unsqueeze(0)
        infer_results.append(None)
        with torch.no_grad():
            for i in range(1, len(video_data)):
                cur_img, cur_mask, cur_point = video_data[i]
                cur_img = cur_img.unsqueeze(0)
                cur_mask = cur_mask.unsqueeze(0)
                cur_point = cur_point.unsqueeze(0)
                pred_bou = model(
                    fir_img.cuda(),
                    fir_point.cuda(),
                    pre_img.cuda(),
                    pre_point.cuda(),
                    pre_mask.cuda(),
                    cur_img.cuda(),
                )
                pred_mask = self.gt_rasterizer(pred_bou, 224, 224)
                pred_mask[pred_mask == -1] = 0
                iou = get_batch_average_bou_iou(
                    pred_bou, cur_mask.cuda(), self.gt_rasterizer
                )
                infer_results.append((pred_bou, pred_mask, iou.item()))
                pre_img, pre_mask, pre_point = cur_img, pred_mask, pred_bou
        return infer_results

    def infer_all_videos(self, model: nn.Module, use_tqdm=False):
        self.infer_results = []
        # for video_idx in tqdm(range(len(self.data_set))):
        if use_tqdm:
            for video_idx in tqdm(range(len(self.data_set))):
                infer_results = self.infer_one_video(video_idx, model)
                self.infer_results.append(infer_results)
        else:
            for video_idx in range(len(self.data_set)):
                infer_results = self.infer_one_video(video_idx, model)
                self.infer_results.append(infer_results)

    def compute_video_iou(self, video_idx: int):
        infer_results = self.infer_results[video_idx]
        ious = [result[-1] for result in infer_results[1:]]
        return np.mean(ious)

    def compute_all_videos_iou(self):
        self.video_ious = []
        for video_idx in range(len(self.data_set)):
            iou = self.compute_video_iou(video_idx)
            self.video_ious.append(iou)
        # return the average iou
        return np.mean(self.video_ious)

    def show_video_results(
        self,
        video_idx: int,
        mask_alpha=0.2,
        img_per_line=5,
    ):
        video_data = self.data_set[video_idx]
        pred_results = self.infer_results[video_idx]
        frame_num = len(video_data)
        line_num = frame_num // img_per_line + 1
        plt.figure(figsize=(20, 4 * line_num))
        for i, pred_data in enumerate(pred_results):
            plt.subplot(line_num, img_per_line, i + 1)
            cur_img, cur_mask, cur_point = video_data[i]
            plt.imshow(normalize(cur_img).permute(1, 2, 0))
            plt.imshow(cur_mask, alpha=mask_alpha)
            if pred_data is None:
                plt.title("ground truth")
                plt.axis("off")
                plt.plot(cur_point[:, 0], cur_point[:, 1], "r")
                plt.scatter(cur_point[:, 0], cur_point[:, 1], c="r", s=5)
            else:
                pred_bou, pred_mask, iou = pred_data
                plt.title(f"iou: {iou:.4f}")
                plt.axis("off")
                pred_bou = pred_bou[0].detach().cpu().numpy()
                plt.plot(pred_bou[:, 0], pred_bou[:, 1], "r")
                plt.scatter(pred_bou[:, 0], pred_bou[:, 1], c="r", s=5)


class DeformLightVideo(nn.Module):
    def __init__(
        self,
        layer_num=1,
        head_num=6,
        medium_level_size=[14, 28, 56, 112],
        offset_limit=10,
        n_points=4,
        mem_point_num=64,
        freeze_backbone=True,
    ) -> None:
        super(DeformLightVideo, self).__init__()
        self.offset_limit = offset_limit
        self.n_points = n_points
        self.point_num = mem_point_num
        self.medium_level_size = medium_level_size
        self.featup = torch.hub.load(
            "mhamilton723/FeatUp",
            "dino16",
            use_norm=True,
        )
        if freeze_backbone:
            for param in self.featup.parameters():
                param.requires_grad = False
        d_model = 384
        d_ffn = 1024
        n_levels = len(medium_level_size) + 1
        self.pos_enoc = PositionalEncoding(d_model)

        enc_layer = DeformableTransformerEncoderLayer(
            d_model=d_model,
            d_ffn=d_ffn,
            n_levels=n_levels,
            n_heads=head_num,
            n_points=n_points,
        )
        dec_layer = DeformableTransformerDecoderLayer(
            d_model=d_model,
            d_ffn=d_ffn,
            n_levels=n_levels,
            n_heads=head_num,
            n_points=n_points,
        )

        self.first_query_embed = nn.Embedding(mem_point_num, d_model)
        xavier_uniform_(self.first_query_embed.weight)
        self.first_layer_norm = nn.LayerNorm(d_model)
        self.fir_enc = DeformableTransformerEncoder(
            enc_layer,
            num_layers=layer_num,
        )
        self.fir_dec = DeformableTransformerDecoder(
            dec_layer,
            num_layers=layer_num,
        )
        self.previous_query_embed = nn.Embedding(mem_point_num, d_model)
        xavier_uniform_(self.previous_query_embed.weight)
        self.previous_layer_norm = nn.LayerNorm(d_model)
        self.pre_enc = DeformableTransformerEncoder(
            enc_layer,
            num_layers=layer_num,
        )
        self.pre_dec = DeformableTransformerDecoder(
            dec_layer,
            num_layers=layer_num,
        )
        self.extra_layer_norm = nn.LayerNorm(d_model)

        self.current_query_embed = nn.Embedding(mem_point_num, d_model)
        xavier_uniform_(self.current_query_embed.weight)
        self.current_layer_norm = nn.LayerNorm(d_model)
        self.cur_enc = DeformableTransformerEncoder(
            enc_layer,
            num_layers=layer_num,
        )
        extra_dec_layer = DeformableTransformerExtraDecoderLayer(
            d_model=d_model,
            d_ffn=d_ffn,
            n_levels=n_levels,
            n_heads=head_num,
            n_points=n_points,
        )
        self.cur_dec = DeformableTransformerExtraDecoder(
            extra_dec_layer,
            num_layers=layer_num,
        )
        self.xy_fc = MLP(d_model, d_model, 2, 3)

    def forward(
        self,
        fir_img: torch.Tensor,
        fir_bou: torch.Tensor,
        pre_img: torch.Tensor,
        pre_bou: torch.Tensor,
        pre_sgm: torch.Tensor,
        cur_img: torch.Tensor,
    ):
        fir_bou = fir_bou / 224
        pre_bou = pre_bou / 224
        (
            fir_img_srcs_flatten,
            fir_spatial_shapes,
            fir_level_start_index,
            fir_valid_ratios,
        ) = self._get_enced_img_scrs(
            fir_img,
            self.fir_enc,
            self.first_layer_norm,
        )
        (
            pre_img_srcs_flatten,
            pre_spatial_shapes,
            pre_level_start_index,
            pre_valid_ratios,
        ) = self._get_enced_img_scrs(
            pre_img,
            self.pre_enc,
            self.previous_layer_norm,
        )
        (
            cur_img_srcs_flatten,
            cur_spatial_shapes,
            cur_level_start_index,
            cur_valid_ratios,
        ) = self._get_enced_img_scrs(
            cur_img,
            self.cur_enc,
            self.current_layer_norm,
        )

        B, S, C = fir_img_srcs_flatten.shape
        first_queries = (
            self.first_query_embed.weight.unsqueeze(0).repeat(B, 1, 1).cuda()
        )
        first_memory, _ = self.fir_dec(
            first_queries,
            fir_bou,
            fir_img_srcs_flatten,
            fir_spatial_shapes,
            fir_level_start_index,
            fir_valid_ratios,
        )
        previous_queries = (
            self.previous_query_embed.weight.unsqueeze(0).repeat(B, 1, 1).cuda()
        )
        previous_memory, _ = self.pre_dec(
            previous_queries,
            pre_bou,
            pre_img_srcs_flatten,
            pre_spatial_shapes,
            pre_level_start_index,
            pre_valid_ratios,
        )
        extra_memory = torch.cat([first_memory, previous_memory], 1)
        extra_memory = self.extra_layer_norm(extra_memory)
        extra_memory = self.pos_enoc(extra_memory)

        current_queries = (
            self.current_query_embed.weight.unsqueeze(0).repeat(B, 1, 1).cuda()
        )
        cur_bou = pre_bou
        decode_output, _ = self.cur_dec(
            current_queries,
            cur_bou,
            cur_img_srcs_flatten,
            cur_spatial_shapes,
            cur_level_start_index,
            cur_valid_ratios,
            extra_memory,
        )

        xy_offset = self.xy_fc(decode_output)
        xy_offset = (xy_offset.sigmoid() - 0.5) * self.offset_limit / 224

        result = pre_bou + xy_offset
        result = result * 224
        if not self.training:
            result = result.clamp(0, 223)
        return result

    def _get_img_scrs(self, img: torch.Tensor, layernorm: nn.LayerNorm):
        feats = self.featup(img)
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
            [get_valid_ratio(mask) for mask in padding_masks],
            1,
        ).cuda()
        spatial_shapes = torch.as_tensor(
            spatial_shapes,
            dtype=torch.long,
            device=src_flatten.device,
        )
        src_flatten = layernorm(src_flatten)
        src_flatten = self.pos_enoc(src_flatten)
        return src_flatten, spatial_shapes, level_start_index, valid_ratios

    def _get_enced_img_scrs(
        self,
        img: torch.Tensor,
        encoder: DeformableTransformerEncoder,
        layernorm: nn.LayerNorm,
    ):
        src_flatten, spatial_shapes, level_start_index, valid_ratios = (
            self._get_img_scrs(img, layernorm)
        )
        src_flatten = encoder(
            src=src_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
        )
        return src_flatten, spatial_shapes, level_start_index, valid_ratios


class IMGPositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, d_model=384, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = d_model // 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        # x = tensor_list.tensors
        # mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class DeformLightVideoPos(nn.Module):
    def __init__(
        self,
        layer_num=1,
        head_num=6,
        medium_level_size=[14, 28, 56, 112],
        offset_limit=10,
        n_points=4,
        mem_point_num=64,
        freeze_backbone=True,
    ) -> None:
        super(DeformLightVideoPos, self).__init__()
        self.offset_limit = offset_limit
        self.n_points = n_points
        self.point_num = mem_point_num
        self.medium_level_size = medium_level_size
        self.featup = torch.hub.load(
            "mhamilton723/FeatUp",
            "dino16",
            use_norm=True,
        )
        if freeze_backbone:
            for param in self.featup.parameters():
                param.requires_grad = False
        d_model = 384
        d_ffn = 1024
        n_levels = len(medium_level_size) + 1
        self.pos_enoc = PositionalEncoding(d_model)
        self.level_pos = nn.Embedding(n_levels, d_model)
        self.img_pos = IMGPositionEmbeddingSine(d_model=d_model)
        xavier_uniform_(self.level_pos.weight)

        enc_layer = DeformableTransformerEncoderLayer(
            d_model=d_model,
            d_ffn=d_ffn,
            n_levels=n_levels,
            n_heads=head_num,
            n_points=n_points,
        )
        dec_layer = DeformableTransformerDecoderLayer(
            d_model=d_model,
            d_ffn=d_ffn,
            n_levels=n_levels,
            n_heads=head_num,
            n_points=n_points,
        )

        self.first_query_embed = nn.Embedding(mem_point_num, d_model)
        xavier_uniform_(self.first_query_embed.weight)
        self.first_layer_norm = nn.LayerNorm(d_model)
        self.fir_enc = DeformableTransformerEncoder(
            enc_layer,
            num_layers=layer_num,
        )
        self.fir_dec = DeformableTransformerDecoder(
            dec_layer,
            num_layers=layer_num,
        )
        self.previous_query_embed = nn.Embedding(mem_point_num, d_model)
        xavier_uniform_(self.previous_query_embed.weight)
        self.previous_layer_norm = nn.LayerNorm(d_model)
        self.pre_enc = DeformableTransformerEncoder(
            enc_layer,
            num_layers=layer_num,
        )
        self.pre_dec = DeformableTransformerDecoder(
            dec_layer,
            num_layers=layer_num,
        )
        self.extra_layer_norm = nn.LayerNorm(d_model)

        self.current_query_embed = nn.Embedding(mem_point_num, d_model)
        xavier_uniform_(self.current_query_embed.weight)
        self.current_layer_norm = nn.LayerNorm(d_model)
        self.cur_enc = DeformableTransformerEncoder(
            enc_layer,
            num_layers=layer_num,
        )
        extra_dec_layer = DeformableTransformerExtraDecoderLayer(
            d_model=d_model,
            d_ffn=d_ffn,
            n_levels=n_levels,
            n_heads=head_num,
            n_points=n_points,
        )
        self.cur_dec = DeformableTransformerExtraDecoder(
            extra_dec_layer,
            num_layers=layer_num,
        )
        self.xy_fc = MLP(d_model, d_model, 2, 3)

    def forward(
        self,
        fir_img: torch.Tensor,
        fir_bou: torch.Tensor,
        pre_img: torch.Tensor,
        pre_bou: torch.Tensor,
        pre_sgm: torch.Tensor,
        cur_img: torch.Tensor,
    ):
        fir_bou = fir_bou / 224
        pre_bou = pre_bou / 224
        (
            fir_img_srcs_flatten,
            fir_spatial_shapes,
            fir_level_start_index,
            fir_valid_ratios,
        ) = self._get_enced_img_scrs(
            fir_img,
            self.fir_enc,
            self.first_layer_norm,
        )
        (
            pre_img_srcs_flatten,
            pre_spatial_shapes,
            pre_level_start_index,
            pre_valid_ratios,
        ) = self._get_enced_img_scrs(
            pre_img,
            self.pre_enc,
            self.previous_layer_norm,
        )
        (
            cur_img_srcs_flatten,
            cur_spatial_shapes,
            cur_level_start_index,
            cur_valid_ratios,
        ) = self._get_enced_img_scrs(
            cur_img,
            self.cur_enc,
            self.current_layer_norm,
        )

        B, S, C = fir_img_srcs_flatten.shape
        first_queries = (
            self.first_query_embed.weight.unsqueeze(0).repeat(B, 1, 1).cuda()
        )
        first_memory, _ = self.fir_dec(
            first_queries,
            fir_bou,
            fir_img_srcs_flatten,
            fir_spatial_shapes,
            fir_level_start_index,
            fir_valid_ratios,
        )
        previous_queries = (
            self.previous_query_embed.weight.unsqueeze(0).repeat(B, 1, 1).cuda()
        )
        previous_memory, _ = self.pre_dec(
            previous_queries,
            pre_bou,
            pre_img_srcs_flatten,
            pre_spatial_shapes,
            pre_level_start_index,
            pre_valid_ratios,
        )
        extra_memory = torch.cat([first_memory, previous_memory], 1)
        extra_memory = self.extra_layer_norm(extra_memory)
        extra_memory = self.pos_enoc(extra_memory)

        current_queries = (
            self.current_query_embed.weight.unsqueeze(0).repeat(B, 1, 1).cuda()
        )
        cur_bou = pre_bou
        decode_output, _ = self.cur_dec(
            current_queries,
            cur_bou,
            cur_img_srcs_flatten,
            cur_spatial_shapes,
            cur_level_start_index,
            cur_valid_ratios,
            extra_memory,
        )

        xy_offset = self.xy_fc(decode_output)
        xy_offset = (xy_offset.sigmoid() - 0.5) * self.offset_limit / 224

        result = pre_bou + xy_offset
        result = result * 224
        if not self.training:
            return result.clamp(0, 223)
        return result

    def _get_img_scrs(self, img: torch.Tensor, layernorm: nn.LayerNorm):
        feats = self.featup(img)
        srcs = []
        padding_masks = []
        pos_embeds = []
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
            pos_embeds.append(self.img_pos(src, padding_masks[-1]))
        src_flatten = []
        spatial_shapes = []
        pos_embed_flatten = []
        # for src in srcs:
        # for src, pos in zip(srcs, pos_embeds):
        for lvl, (src, pos) in enumerate(zip(srcs, pos_embeds)):
            src_flatten.append(
                rearrange(src, "b c h w -> b (h w) c"),
            )
            spatial_shapes.append(src.shape[-2:])
            pos = rearrange(pos, "b c h w -> b (h w) c")
            pos = self.level_pos.weight[lvl].view(1, 1, -1) + pos
            pos_embed_flatten.append(pos)

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
        pos_embed_flatten = torch.cat(pos_embed_flatten, 1).cuda()
        valid_ratios = torch.stack(
            [get_valid_ratio(mask) for mask in padding_masks],
            1,
        ).cuda()
        spatial_shapes = torch.as_tensor(
            spatial_shapes,
            dtype=torch.long,
            device=src_flatten.device,
        )
        src_flatten = layernorm(src_flatten)
        # src_flatten = self.pos_enoc(src_flatten)

        return (
            src_flatten,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            pos_embed_flatten,
        )

    def _get_enced_img_scrs(
        self,
        img: torch.Tensor,
        encoder: DeformableTransformerEncoder,
        layernorm: nn.LayerNorm,
    ):
        (
            src_flatten,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            pos_embed_flatten,
        ) = self._get_img_scrs(img, layernorm)
        src_flatten += pos_embed_flatten
        src_flatten = encoder(
            src=src_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
        )
        return src_flatten, spatial_shapes, level_start_index, valid_ratios


class DAVIS_withPointSet(Dataset):
    def __init__(
        self,
        raw_set: DAVIS_Rawset,
        is_train: bool,
        point_nums=[4, 8, 16, 32, 64],
    ) -> None:
        super().__init__()
        # remove all the video with empty frame
        empty_video_idx = []
        for video_idx, video_data in enumerate(raw_set.data_set):
            for frame_data in video_data:
                img, mask = frame_data
                if mask.sum() == 0:
                    empty_video_idx.append(video_idx)
                    break
        self.raw_data_set = []
        if is_train:
            point_path = "sample_results/train_256_uniform.json"
        else:
            point_path = "sample_results/val_256_uniform.json"
        with open(point_path, "r") as f:
            points = json.load(f)

        def get_points(video_idx, frame_idx, point_num):
            return torch.tensor(
                points[video_idx][frame_idx][str(point_num)]["boundary"]
            ).float()

        def get_points_set(video_idx, frame_idx):
            point_set = [
                get_points(video_idx, frame_idx, point_num) for point_num in point_nums
            ]
            # return torch.nested.nested_tensor(point_set)
            return point_set

        for video_idx, video_data in enumerate(raw_set.data_set):
            if video_idx in empty_video_idx:
                continue
            self.raw_data_set.append([])
            for frame_idx, frame_data in enumerate(video_data):
                img, mask = frame_data
                points_set = get_points_set(video_idx, frame_idx)
                self.raw_data_set[-1].append((img, mask, points_set))
        self.data = []
        for video_idx, video_data in enumerate(self.raw_data_set):
            for frame_idx in range(len(video_data) - 1):
                self.data.append(
                    (
                        video_idx,
                        frame_idx,
                        video_data[0],
                        video_data[frame_idx],
                        video_data[frame_idx + 1],
                    )
                )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        video_idx, frame_idx, first_frame, previous_frame, current_frame = self.data[
            idx
        ]
        return video_idx, frame_idx, first_frame, previous_frame, current_frame


class DAVIS_withPointSetRandom(Dataset):
    def __init__(
        self,
        raw_set: DAVIS_Rawset,
        is_train: bool,
        point_nums=[4, 8, 16, 32, 64],
        sample_offset=10,
    ) -> None:
        super().__init__()
        # remove all the video with empty frame
        self.sample_offset = sample_offset
        empty_video_idx = []
        for video_idx, video_data in enumerate(raw_set.data_set):
            for frame_data in video_data:
                img, mask = frame_data
                if mask.sum() == 0:
                    empty_video_idx.append(video_idx)
                    break
        self.raw_data_set = []
        if is_train:
            point_path = "sample_results/train_256_uniform.json"
        else:
            point_path = "sample_results/val_256_uniform.json"
        with open(point_path, "r") as f:
            points = json.load(f)

        def get_points(video_idx, frame_idx, point_num):
            return torch.tensor(
                points[video_idx][frame_idx][str(point_num)]["boundary"]
            ).float()

        def get_points_set(video_idx, frame_idx):
            point_set = [
                get_points(video_idx, frame_idx, point_num) for point_num in point_nums
            ]
            # return torch.nested.nested_tensor(point_set)
            return point_set

        for video_idx, video_data in enumerate(raw_set.data_set):
            if video_idx in empty_video_idx:
                continue
            self.raw_data_set.append([])
            for frame_idx, frame_data in enumerate(video_data):
                img, mask = frame_data
                points_set = get_points_set(video_idx, frame_idx)
                self.raw_data_set[-1].append((img, mask, points_set))

    def __len__(self):
        return len(self.raw_data_set)

    def __getitem__(self, idx: int):
        video_idx = idx
        video_data = self.raw_data_set[video_idx]
        first_frame = video_data[0]
        video_len = len(video_data)
        previous_idx = random.randint(0, video_len - 2)
        sample_low = previous_idx + 1
        sample_high = min(previous_idx + self.sample_offset, video_len - 1)
        current_idx = random.randint(sample_low, sample_high)
        previous_frame = video_data[previous_idx]
        current_frame = video_data[current_idx]
        return video_idx, previous_idx, first_frame, previous_frame, current_frame


class DefInitVideo(nn.Module):
    def __init__(
        self,
        layer_num=1,
        up_scale_num=4,
        head_num=6,
        medium_level_size=[14, 28, 56, 112],
        offset_limit=10,
        n_points=4,
        mem_point_num=64,
        freeze_backbone=True,
    ):
        super(DefInitVideo, self).__init__()
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

        enc_layer = DeformableTransformerEncoderLayer(
            d_model=d_model,
            d_ffn=d_ffn,
            n_levels=n_levels,
            n_heads=head_num,
            n_points=n_points,
        )
        dec_layer = DeformableTransformerDecoderLayer(
            d_model=d_model,
            d_ffn=d_ffn,
            n_levels=n_levels,
            n_heads=head_num,
            n_points=n_points,
        )

        self.first_query_embed = nn.Embedding(mem_point_num, d_model)
        xavier_uniform_(self.first_query_embed.weight)
        self.first_layer_norm = nn.LayerNorm(d_model)
        self.fir_enc = DeformableTransformerEncoder(
            enc_layer,
            num_layers=layer_num,
        )
        self.fir_dec = DeformableTransformerDecoder(
            dec_layer,
            num_layers=layer_num,
        )
        self.previous_query_embed = nn.Embedding(mem_point_num, d_model)
        xavier_uniform_(self.previous_query_embed.weight)
        self.previous_layer_norm = nn.LayerNorm(d_model)
        self.pre_enc = DeformableTransformerEncoder(
            enc_layer,
            num_layers=layer_num,
        )
        self.pre_dec = DeformableTransformerDecoder(
            dec_layer,
            num_layers=layer_num,
        )
        self.extra_layer_norm = nn.LayerNorm(d_model)

        query_num = 4
        for _ in range(up_scale_num):
            query_num *= 2
        self.current_query_embed = nn.Embedding(query_num, d_model)
        xavier_uniform_(self.current_query_embed.weight)
        self.current_layer_norm = nn.LayerNorm(d_model)
        self.cur_enc = DeformableTransformerEncoder(
            enc_layer,
            num_layers=layer_num,
        )
        extra_dec_layer = DeformableTransformerExtraDecoderLayer(
            d_model=d_model,
            d_ffn=d_ffn,
            n_levels=n_levels,
            n_heads=head_num,
            n_points=n_points,
        )
        extra_dec = DeformableTransformerExtraDecoder(
            extra_dec_layer,
            num_layers=layer_num,
        )
        self.extra_decs = _get_clones(extra_dec, up_scale_num + 1)
        xy_fc = MLP(d_model, d_model, 2, 3)
        self.xy_fcs = _get_clones(xy_fc, up_scale_num + 1)

    def forward(
        self,
        fir_img: torch.Tensor,
        fir_bou: torch.Tensor,
        pre_img: torch.Tensor,
        pre_bou_set: list[torch.Tensor],
        cur_img: torch.Tensor,
    ):
        tmp_bou_set = []
        for bou in pre_bou_set:
            tmp_bou_set.append(bou.cuda() / 224)
        pre_bou_set = tmp_bou_set
        fir_bou = fir_bou / 224

        (
            fir_img_srcs_flatten,
            fir_spatial_shapes,
            fir_level_start_index,
            fir_valid_ratios,
        ) = self._get_enced_img_scrs(
            fir_img,
            self.fir_enc,
            self.first_layer_norm,
        )
        (
            pre_img_srcs_flatten,
            pre_spatial_shapes,
            pre_level_start_index,
            pre_valid_ratios,
        ) = self._get_enced_img_scrs(
            pre_img,
            self.pre_enc,
            self.previous_layer_norm,
        )
        (
            cur_img_srcs_flatten,
            cur_spatial_shapes,
            cur_level_start_index,
            cur_valid_ratios,
        ) = self._get_enced_img_scrs(
            cur_img,
            self.cur_enc,
            self.current_layer_norm,
        )

        B, S, C = fir_img_srcs_flatten.shape
        first_queries = (
            self.first_query_embed.weight.unsqueeze(0).repeat(B, 1, 1).cuda()
        )
        first_memory, _ = self.fir_dec(
            first_queries,
            fir_bou,
            fir_img_srcs_flatten,
            fir_spatial_shapes,
            fir_level_start_index,
            fir_valid_ratios,
        )
        previous_queries = (
            self.previous_query_embed.weight.unsqueeze(0).repeat(B, 1, 1).cuda()
        )
        previous_memory, _ = self.pre_dec(
            previous_queries,
            pre_bou_set[-1],
            pre_img_srcs_flatten,
            pre_spatial_shapes,
            pre_level_start_index,
            pre_valid_ratios,
        )
        extra_memory = torch.cat([first_memory, previous_memory], 1)
        extra_memory = self.extra_layer_norm(extra_memory)
        extra_memory = self.pos_enoc(extra_memory)

        cur_queries = (
            self.current_query_embed.weight.unsqueeze(0).repeat(B, 1, 1).cuda()
        )
        init_bou = pre_bou_set[0]
        current_query_num = init_bou.shape[1]
        current_query = cur_queries[:, :current_query_num]
        decode_output, _ = self.extra_decs[0](
            current_query,
            init_bou,
            cur_img_srcs_flatten,
            cur_spatial_shapes,
            cur_level_start_index,
            cur_valid_ratios,
            extra_memory,
        )

        xy_offset = (
            (self.xy_fcs[0](decode_output).sigmoid() - 0.5) * self.offset_limit / 224
        )
        init_bou += xy_offset

        results = [init_bou]
        for i in range(self.up_scale_num):
            new_query = cur_queries[:, current_query_num : current_query_num * 2]
            current_query_num *= 2

            current_query = torch.zeros((B, current_query_num, C)).to(
                cur_img_srcs_flatten.device
            )
            current_query[:, ::2] = new_query
            current_query[:, 1::2] = decode_output

            new_bou = torch.zeros((B, current_query_num, 2)).to(
                cur_img_srcs_flatten.device
            )
            new_bou[:, 1::2] = results[-1]
            new_points = pre_bou_set[i + 1][:, ::2]
            # print(new_points.shape)
            # print(new_bou.shape)
            # print(new_bou[:, ::2].shape)
            new_bou[:, ::2] = new_points

            cur_bou = new_bou
            decode_output, _ = self.extra_decs[i + 1](
                current_query,
                cur_bou,
                cur_img_srcs_flatten,
                cur_spatial_shapes,
                cur_level_start_index,
                cur_valid_ratios,
                extra_memory,
            )

            xy_offset = (
                (self.xy_fcs[i + 1](decode_output).sigmoid() - 0.5)
                * self.offset_limit
                / 224
            )
            cur_bou += xy_offset

            results.append(cur_bou)
        results = [result * 224 for result in results]

        return results

    def _get_img_scrs(self, img: torch.Tensor, layernorm: nn.LayerNorm):
        feats = self.featup(img)
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
            [get_valid_ratio(mask) for mask in padding_masks],
            1,
        ).cuda()
        spatial_shapes = torch.as_tensor(
            spatial_shapes,
            dtype=torch.long,
            device=src_flatten.device,
        )
        src_flatten = layernorm(src_flatten)
        src_flatten = self.pos_enoc(src_flatten)
        return src_flatten, spatial_shapes, level_start_index, valid_ratios

    def _get_enced_img_scrs(
        self,
        img: torch.Tensor,
        encoder: DeformableTransformerEncoder,
        layernorm: nn.LayerNorm,
    ):
        src_flatten, spatial_shapes, level_start_index, valid_ratios = (
            self._get_img_scrs(img, layernorm)
        )
        src_flatten = encoder(
            src=src_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
        )
        return src_flatten, spatial_shapes, level_start_index, valid_ratios


class InitVideoInferer:
    def __init__(
        self,
        dataset: DAVIS_withPointSet,
        gt_rasterizer: SoftPolygon,
    ) -> None:
        self.dataset = dataset.raw_data_set
        self.gt_rasterizer = gt_rasterizer

    def infer_one_video(self, video_idx: int, model: nn.Module):
        infer_results = []
        video_data = self.dataset[video_idx]
        model.eval()

        def get_batch_data(frame_data: tuple):
            img, mask, points_set = frame_data
            img = img.unsqueeze(0)
            mask = mask.unsqueeze(0)
            points_set = [points.unsqueeze(0) for points in points_set]
            return img, mask, points_set

        fir_img, fir_mask, fir_points_set = get_batch_data(video_data[0])
        pre_img, pre_mask, pre_points_set = get_batch_data(video_data[0])
        infer_results.append(None)
        with torch.no_grad():
            for i in range(1, len(video_data)):
                cur_img, cur_mask, cur_points_set = get_batch_data(video_data[i])
                pred_bous = model(
                    fir_img.cuda(),
                    fir_points_set[-1].cuda(),
                    pre_img.cuda(),
                    pre_points_set,
                    cur_img.cuda(),
                )
                pred_mask = self.gt_rasterizer(pred_bous[-1], 224, 224)
                pred_mask[pred_mask == -1] = 0
                iou = get_batch_average_bou_iou(
                    pred_bous[-1],
                    cur_mask.cuda(),
                    self.gt_rasterizer,
                )
                infer_results.append((pred_bous, pred_mask, iou.item()))
                pre_img, pre_mask, pre_points_set = cur_img, pred_mask, pred_bous
        return infer_results

    def infer_all_videos(self, model: nn.Module, use_tqdm=False):
        self.infer_results = []
        if use_tqdm:
            video_range = tqdm(range(len(self.dataset)))
        else:
            video_range = range(len(self.dataset))
        for video_idx in video_range:
            self.infer_results.append(self.infer_one_video(video_idx, model))

    def compute_video_iou(self, video_idx: int):
        infer_results = self.infer_results[video_idx]
        ious = [result[-1] for result in infer_results[1:]]
        return np.mean(ious)

    def compute_all_iou(self):
        ious = []
        for video_idx in range(len(self.dataset)):
            ious.append(self.compute_video_iou(video_idx))
        return np.mean(ious)

    def show_video_results(
        self,
        video_idx: int,
        mask_alpha=0.2,
        img_per_row=5,
    ):
        video_data = self.dataset[video_idx]
        pred_results = self.infer_results[video_idx]
        frame_num = len(video_data)
        line_num = frame_num // img_per_row + 1
        plt.figure(figsize=(20, 4 * line_num))
        for i, pred_data in enumerate(pred_results):
            plt.subplot(line_num, img_per_row, i + 1)
            cur_img, cur_mask, cur_points = video_data[i]
            cur_point = cur_points[-1]
            plt.imshow(normalize(cur_img).permute(1, 2, 0))
            plt.imshow(cur_mask, alpha=mask_alpha)
            if pred_data is None:
                plt.title("ground truth")
                plt.axis("off")
                plt.plot(cur_point[:, 0], cur_point[:, 1], "r")
                plt.scatter(cur_point[:, 0], cur_point[:, 1], c="r", s=5)
            else:
                pred_bous, pred_mask, iou = pred_data
                pred_bou = pred_bous[-1]
                plt.title(f"iou: {iou:.4f}")
                plt.axis("off")
                pred_bou = pred_bou[0].detach().cpu().numpy()
                plt.plot(pred_bou[:, 0], pred_bou[:, 1], "r")
                plt.scatter(pred_bou[:, 0], pred_bou[:, 1], c="r", s=5)
