from Loader_17 import DAVIS_Rawset, normalize
from deform_video import (
    DAVIS_withPoint,
    IMGPositionEmbeddingSine,
    PositionalEncoding,
    get_valid_ratio,
    MLP,
)
from deform_model import (
    DeformableTransformerDecoderLayer,
    DeformableTransformerDecoder,
    get_batch_average_bou_iou,
    add_mid_points,
)
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import Adam
from einops import rearrange, repeat, reduce
import numpy as np
import torch.nn.functional as F
import gc
from polygon import SoftPolygon, RasLoss
import random
from tqdm import tqdm
from preprocess_utensils import get_boundary_points, uniform_sample_points


class DAVIS_Video(Dataset):
    def __init__(
        self,
        raw_set: list[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
        frame_num: int,
        video_idx_list: list[int] = None,
        is_eval: bool = False,
        use_fir_point: bool = True,
    ):
        self.raw_set = raw_set
        if video_idx_list is not None:
            self.raw_set = [self.raw_set[i] for i in video_idx_list]
        self.frame_num = frame_num
        self.is_eval = is_eval
        self.use_fir_point = use_fir_point

    def __len__(self):
        return len(self.raw_set)

    def __getitem__(self, idx):
        video_data = self.raw_set[idx]
        total_frame_num = len(video_data)
        selected_frame_idxs = []
        if total_frame_num < self.frame_num:
            selected_frame_idxs = list(range(total_frame_num))
        else:
            # uniformly sample frame_num frames, and the first frame is always included
            selected_frame_idxs = [0]
            interval = (total_frame_num - 1) // (self.frame_num - 1)
            if self.is_eval:
                offset = 0
            else:
                offset_min = -(interval - 1)
                offset_max = (total_frame_num - 1) - (self.frame_num - 1) * interval
                offset = random.randint(offset_min, offset_max)
            for i in range(1, self.frame_num):
                selected_frame_idxs.append(i * interval + offset)
        selected_frame_idxs = sorted(selected_frame_idxs)
        img_list = []
        mask_list = []
        point_list = []
        fir_point = video_data[0][2]
        for frame_idx in selected_frame_idxs:
            img, mask, point = video_data[frame_idx]
            img_list.append(img)
            mask_list.append(mask)
            if self.use_fir_point:
                point_list.append(fir_point)
            else:
                point_list.append(point)
        return torch.stack(img_list), torch.stack(mask_list), torch.stack(point_list)


class Cotracker(nn.Module):
    def __init__(
        self,
        point_num: int,
        refine_iter: int = 6,
        freeze_featup: bool = True,
        d_model: int = 256,
        n_heads: int = 8,
        n_points: int = 8,
        d_ffn: int = 1024,
        n_layers: int = 1,
    ):
        super(Cotracker, self).__init__()
        self.refine_iter = refine_iter
        self.featup = torch.hub.load(
            "mhamilton723/FeatUp",
            "dino16",
            use_norm=True,
        )
        if freeze_featup:
            for param in self.featup.parameters():
                param.requires_grad = False
        d_featup = 384
        if d_featup != d_model:
            self.featup_fc = nn.Linear(d_featup, d_model)
        else:
            self.featup_fc = nn.Identity()
        self.medium_level_size = [14, 28, 56, 112]
        n_levels = len(self.medium_level_size) + 1
        self.offset_ratio = 0.2
        self.pos_2d = IMGPositionEmbeddingSine(d_model)
        self.pos_1d = PositionalEncoding(d_model)
        self.level_pos = nn.Embedding(n_levels, d_model)
        nn.init.xavier_uniform_(self.level_pos.weight)

        self.query_embed = nn.Embedding(point_num, d_model)
        nn.init.xavier_uniform_(self.query_embed.weight)
        dec_layer = DeformableTransformerDecoderLayer(
            d_model=d_model,
            d_ffn=d_ffn,
            n_levels=n_levels,
            n_heads=n_heads,
            n_points=n_points,
        )
        self.def_dec_layernorm = nn.LayerNorm(d_model)
        self.def_decoder = DeformableTransformerDecoder(
            decoder_layer=dec_layer,
            num_layers=n_layers,
        )
        self.fir_layernorm = nn.LayerNorm(d_model)
        self.fir_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model,
                n_heads,
                d_ffn,
                batch_first=True,
            ),
            num_layers=n_layers,
        )
        self.other_layernorm = nn.LayerNorm(d_model)
        self.other_decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model,
                n_heads,
                d_ffn,
                batch_first=True,
            ),
            num_layers=n_layers,
        )
        self.delta_xy_fc = MLP(d_model, d_model, 2, 3)
        self.delta_query_fc = MLP(d_model, d_model, d_model, 3)

    def forward(self, imgs: torch.Tensor, points: torch.Tensor):
        batch_size, frame_num, _, img_h, img_w = imgs.shape
        _, _, point_num, _ = points.shape

        points = points / 224.0

        fir_img = imgs[:, 0]
        fir_points = points[:, 0]
        (
            fir_src_flatten,
            fir_spatial_shapes,
            fir_level_start_index,
            fir_valid_ratios,
            fir_pos_embed_flatten,
        ) = self._get_img_scrs(fir_img, self.def_dec_layernorm)
        fir_src_flatten += fir_pos_embed_flatten
        fir_query_embed = repeat(self.query_embed.weight, "p d -> b p d", b=batch_size)
        fir_point_feats, _ = self.def_decoder(
            fir_query_embed,
            fir_points,
            fir_src_flatten,
            fir_spatial_shapes,
            fir_level_start_index,
            fir_valid_ratios,
        )
        fir_point_feats = self.fir_layernorm(fir_point_feats)
        fir_query = self.pos_1d(fir_point_feats)
        fir_memory = self.fir_encoder(fir_query)

        other_imgs = imgs[:, 1:]
        other_points = points[:, 1:]
        other_imgs = rearrange(other_imgs, "b f c h w -> (b f) c h w")
        other_points = rearrange(other_points, "b f p d -> (b f) p d")
        (
            other_src_flatten,
            other_spatial_shapes,
            other_level_start_index,
            other_valid_ratios,
            other_pos_embed_flatten,
        ) = self._get_img_scrs(other_imgs, self.def_dec_layernorm)
        other_src_flatten += other_pos_embed_flatten
        other_queries = repeat(
            self.query_embed.weight, "p d -> b p d", b=batch_size * (frame_num - 1)
        )

        output_coords = []
        for i in range(self.refine_iter):
            other_point_feats, _ = self.def_decoder(
                other_queries,
                other_points,
                other_src_flatten,
                other_spatial_shapes,
                other_level_start_index,
                other_valid_ratios,
            )
            other_point_queries = rearrange(
                other_point_feats, "(b f)p d -> b (f p) d", f=frame_num - 1
            )
            other_point_queries = self.other_layernorm(other_point_queries)
            other_point_queries = self.pos_1d(other_point_queries)
            other_point_queries = self.other_decoder(
                tgt=other_point_queries,
                memory=fir_memory,
            )
            other_point_queries = rearrange(
                other_point_queries, "b (f p) d -> (b f) p d", f=frame_num - 1
            )
            delta_xy = self.delta_xy_fc(other_point_queries)
            delta_xy = (delta_xy.sigmoid() - 0.5) * self.offset_ratio
            other_points = other_points + delta_xy
            result = rearrange(
                other_points, "(b f) p d -> b f p d", b=batch_size, f=frame_num - 1
            )
            result = result * 224.0
            output_coords.append(result)
            delta_query = self.delta_query_fc(other_point_feats)
            other_queries = other_queries + delta_query
        output_coords = torch.stack(output_coords, 1)
        return output_coords

    def _get_img_scrs(self, img: torch.Tensor, layernorm: nn.LayerNorm):
        b, c, h, w = img.shape
        feats = self.featup(img)
        feats = rearrange(feats, "b c h w -> b (h w) c")
        feats = self.featup_fc(feats)
        feats = rearrange(feats, "b (h w) c -> b c h w", h=h, w=w)
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
            pos_embeds.append(self.pos_2d(src, padding_masks[-1]))
        src_flatten = []
        spatial_shapes = []
        pos_embed_flatten = []

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

        return (
            src_flatten,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            pos_embed_flatten,
        )


class VideoLoss(nn.Module):
    def __init__(self, ras_loss: RasLoss, gt_rasterizer: SoftPolygon):
        super(VideoLoss, self).__init__()
        self.ras_loss = ras_loss
        self.gt_rasterizer = gt_rasterizer

    def forward(self, pred_points: torch.Tensor, gt_masks: torch.Tensor):
        pred_points = rearrange(pred_points, "b i f p xy -> i (b f) p xy")
        gt_masks = rearrange(gt_masks, "b f h w -> (b f) h w")
        total_loss = 0
        total_iou = 0
        iter_num = pred_points.shape[0]
        for i in range(iter_num):
            pred_point = pred_points[i]
            loss = self.ras_loss(pred_point, gt_masks)
            total_loss += loss
        total_loss /= iter_num
        iou = get_batch_average_bou_iou(
            pred_points[-1],
            gt_masks,
            self.gt_rasterizer,
        )
        return total_loss, iou


class VideoLoss(nn.Module):
    def __init__(self, ras_loss: RasLoss, gt_rasterizer: SoftPolygon):
        super(VideoLoss, self).__init__()
        self.ras_loss = ras_loss
        self.gt_rasterizer = gt_rasterizer

    def forward(self, pred_points: torch.Tensor, gt_masks: torch.Tensor):
        pred_points = rearrange(pred_points, "b i f p xy -> i (b f) p xy")
        gt_masks = rearrange(gt_masks, "b f h w -> (b f) h w")
        total_loss = 0
        total_iou = 0
        iter_num = pred_points.shape[0]
        for i in range(iter_num):
            pred_point = pred_points[i]
            loss = self.ras_loss(pred_point, gt_masks)
            total_loss += loss
        total_loss /= iter_num
        iou = get_batch_average_bou_iou(
            pred_points[-1],
            gt_masks,
            self.gt_rasterizer,
        )
        return total_loss, iou


def _get_boundary_points(mask: torch.Tensor, point_num: int) -> torch.Tensor:
    boundary = get_boundary_points(mask.numpy().astype(np.uint8))
    boundary = uniform_sample_points(boundary, point_num)
    boundary = torch.tensor(boundary, dtype=torch.float32)
    return boundary


class CoEvaler:
    def __init__(
        self,
        raw_set: list[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
        gt_rasterizer: SoftPolygon,
        frame_num: int = 10,
        use_uniform_points: bool = False,
    ):
        self.raw_set = raw_set
        self.frame_num = frame_num
        self.gt_rasterizer = gt_rasterizer
        self.total_pred_results = []
        self.use_uniform_points = use_uniform_points

    def eval_one_video(self, video_idx: int, model: Cotracker):
        video_data = self.raw_set[video_idx]
        total_frame_num = len(video_data)
        img_list = []
        mask_list = []
        point_list = []
        for frame_idx in range(total_frame_num):
            img, mask, point = video_data[frame_idx]
            img_list.append(img)
            mask_list.append(mask)
            point_list.append(point)
        total_masks = torch.stack(mask_list)
        total_points = torch.stack(point_list)
        total_pred_points = torch.zeros_like(total_points)
        total_pred_points[0] = total_points[0]
        interval = (total_frame_num - 1) // (self.frame_num - 1)
        offset_min = -(interval - 1)
        offset_max = (total_frame_num - 1) - (self.frame_num - 1) * interval
        done_idx_list = [0]
        for offset in range(offset_min, offset_max + 1):
            selected_frame_idxs = [0]
            for i in range(1, self.frame_num):
                selected_frame_idxs.append(i * interval + offset)
            selected_frame_idxs = sorted(selected_frame_idxs)
            img_list = []
            mask_list = []
            point_list = []
            fir_point = video_data[0][2]
            if self.use_uniform_points:
                fir_point = _get_boundary_points(video_data[0][1], 32)
                if fir_point.shape[0] != 32 or len(fir_point.shape) != 2:
                    fir_point = video_data[0][2]
            for frame_idx in selected_frame_idxs:
                img, mask, point = video_data[frame_idx]
                img_list.append(img)
                mask_list.append(mask)
                point_list.append(fir_point)
            imgs = torch.stack(img_list).unsqueeze(0)
            masks = torch.stack(mask_list).unsqueeze(0)
            points = torch.stack(point_list).unsqueeze(0)
            with torch.no_grad():
                model.eval()
                pred_points = model(imgs.cuda(), points.cuda())
                pred_points = pred_points.clamp(0, 223)
            pred_points = pred_points[0][-1]
            for result_idx, frame_idx in enumerate(selected_frame_idxs[1:]):
                if frame_idx in done_idx_list:
                    continue
                done_idx_list.append(frame_idx)
                total_pred_points[frame_idx] = pred_points[result_idx]
        total_iou = get_batch_average_bou_iou(
            total_pred_points.cuda(),
            total_masks.cuda(),
            self.gt_rasterizer,
        )
        return total_iou, total_pred_points

    def compute_avg_iou(self):
        iou_list = [iou for iou, _ in self.total_pred_results]
        return sum(iou_list) / len(iou_list)

    def eval_all_video(self, model: Cotracker, use_tqdm=False):
        self.total_pred_results = []
        video_num = len(self.raw_set)
        if use_tqdm:
            video_idx_list = tqdm(range(video_num))
        else:
            video_idx_list = range(video_num)
        for video_idx in video_idx_list:
            iou, pred_points = self.eval_one_video(video_idx, model)
            self.total_pred_results.append((iou, pred_points))
        return self.compute_avg_iou()

    def show_iou_distribution(self):
        iou_list = [iou.item() for iou, _ in self.total_pred_results]
        plt.hist(iou_list, bins=20)
        plt.show()

    def show_video_result(
        self,
        video_idx: int,
        gt_rasterizer: SoftPolygon,
        img_per_line=5,
        mask_alpha=0.2,
    ):
        video_data = self.raw_set[video_idx]
        frame_num = len(video_data)
        line_num = (frame_num - 1) // img_per_line + 1
        total_pred_points = self.total_pred_results[video_idx][1]
        plt.figure(figsize=(img_per_line * 4, line_num * 4))
        for i, pred_point in enumerate(total_pred_points):
            plt.subplot(line_num, img_per_line, i + 1)
            plt.axis("off")
            img, mask, point = video_data[i]
            plt.imshow(normalize(img).permute(1, 2, 0))
            iou = get_batch_average_bou_iou(
                pred_point.unsqueeze(0).cuda(), mask.unsqueeze(0).cuda(), gt_rasterizer
            ).item()
            plt.title(f"IoU: {iou:.4f}")
            plt.scatter(pred_point[:, 0], pred_point[:, 1], c="r", s=5)
            plt.plot(pred_point[:, 0], pred_point[:, 1], c="r")
            plt.imshow(mask, alpha=mask_alpha)


def get_batch_4_extrame_points(point_batch: torch.Tensor):
    result = torch.zeros((point_batch.shape[0], 4, 2)).to(point_batch.device)
    # get the highest point
    _, idx = torch.min(point_batch[:, :, 1], dim=1)
    result[:, 0] = point_batch[torch.arange(point_batch.shape[0]), idx]
    # get the rightest point
    _, idx = torch.max(point_batch[:, :, 0], dim=1)
    result[:, 1] = point_batch[torch.arange(point_batch.shape[0]), idx]
    # get the lowest point
    _, idx = torch.max(point_batch[:, :, 1], dim=1)
    result[:, 2] = point_batch[torch.arange(point_batch.shape[0]), idx]
    # get the leftest point
    _, idx = torch.min(point_batch[:, :, 0], dim=1)
    result[:, 3] = point_batch[torch.arange(point_batch.shape[0]), idx]
    return result


class CotrackerIter(nn.Module):
    def __init__(
        self,
        point_num: int,
        up_iter_num: int = 3,
        refine_iter: int = 3,
        freeze_featup: bool = True,
        d_model: int = 256,
        n_heads: int = 8,
        n_points: int = 9,
        d_ffn: int = 1024,
    ):
        super(CotrackerIter, self).__init__()
        self.refine_iter = refine_iter
        self.up_iter_num = up_iter_num
        self.featup = torch.hub.load(
            "mhamilton723/FeatUp",
            "dino16",
            use_norm=True,
        )
        if freeze_featup:
            for param in self.featup.parameters():
                param.requires_grad = False
        d_featup = 384
        if d_featup != d_model:
            self.featup_fc = nn.Linear(d_featup, d_model)
        else:
            self.featup_fc = nn.Identity()
        self.medium_level_size = [28, 56, 112]
        n_levels = len(self.medium_level_size) + 1
        n_layers = 1
        self.offset_ratio = 0.3
        self.pos_2d = IMGPositionEmbeddingSine(d_model)
        self.pos_1d = PositionalEncoding(d_model)
        self.level_pos = nn.Embedding(n_levels, d_model)
        nn.init.xavier_uniform_(self.level_pos.weight)

        self.query_embed = nn.Embedding(point_num, d_model)
        nn.init.xavier_uniform_(self.query_embed.weight)
        dec_layer = DeformableTransformerDecoderLayer(
            d_model=d_model,
            d_ffn=d_ffn,
            n_levels=n_levels,
            n_heads=n_heads,
            n_points=n_points,
        )
        self.def_dec_layernorm = nn.LayerNorm(d_model)
        self.def_decoder = DeformableTransformerDecoder(
            decoder_layer=dec_layer,
            num_layers=n_layers,
        )
        self.fir_layernorm = nn.LayerNorm(d_model)
        self.fir_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model,
                n_heads,
                d_ffn,
                batch_first=True,
            ),
            num_layers=1,
        )
        self.other_layernorm = nn.LayerNorm(d_model)
        self.other_decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model,
                n_heads,
                d_ffn,
                batch_first=True,
            ),
            num_layers=1,
        )
        self.delta_xy_fc = MLP(d_model, d_model, 2, 3)
        self.delta_query_fc = MLP(d_model, d_model, d_model, 3)

    def forward(self, imgs: torch.Tensor, points: torch.Tensor):
        batch_size, frame_num, _, img_h, img_w = imgs.shape
        _, _, point_num, _ = points.shape

        points = points / 224.0

        fir_img = imgs[:, 0]
        fir_points = points[:, 0]
        (
            fir_src_flatten,
            fir_spatial_shapes,
            fir_level_start_index,
            fir_valid_ratios,
            fir_pos_embed_flatten,
        ) = self._get_img_scrs(fir_img, self.def_dec_layernorm)
        fir_src_flatten += fir_pos_embed_flatten
        fir_query_embed = repeat(self.query_embed.weight, "p d -> b p d", b=batch_size)
        fir_point_feats, _ = self.def_decoder(
            fir_query_embed,
            fir_points,
            fir_src_flatten,
            fir_spatial_shapes,
            fir_level_start_index,
            fir_valid_ratios,
        )
        fir_point_feats = self.fir_layernorm(fir_point_feats)
        fir_query = self.pos_1d(fir_point_feats)
        fir_memory = self.fir_encoder(fir_query)

        other_imgs = imgs[:, 1:]
        fir_extrame_points = get_batch_4_extrame_points(fir_points)
        # expand the extrame points to the other frames
        other_points = repeat(fir_extrame_points, "b p d -> b f p d", f=frame_num - 1)
        other_imgs = rearrange(other_imgs, "b f c h w -> (b f) c h w")
        other_points = rearrange(other_points, "b f p d -> (b f) p d")
        (
            other_src_flatten,
            other_spatial_shapes,
            other_level_start_index,
            other_valid_ratios,
            other_pos_embed_flatten,
        ) = self._get_img_scrs(other_imgs, self.def_dec_layernorm)
        other_src_flatten += other_pos_embed_flatten
        cur_point_num = other_points.shape[1]
        total_queries = repeat(
            self.query_embed.weight, "p d -> b p d", b=batch_size * (frame_num - 1)
        )
        cur_queries = total_queries[:, :cur_point_num]

        output_coords = []
        for i in range(self.up_iter_num + 1):
            tmp_coords = []
            for j in range(self.refine_iter):
                other_point_feats, _ = self.def_decoder(
                    cur_queries,
                    other_points,
                    other_src_flatten,
                    other_spatial_shapes,
                    other_level_start_index,
                    other_valid_ratios,
                )
                other_point_queries = rearrange(
                    other_point_feats, "(b f) p d -> b (f p) d", f=frame_num - 1
                )
                other_point_queries = self.other_layernorm(other_point_queries)
                other_point_queries = self.pos_1d(other_point_queries)
                other_point_queries = self.other_decoder(
                    tgt=other_point_queries,
                    memory=fir_memory,
                )
                other_point_queries = rearrange(
                    other_point_queries, "b (f p) d -> (b f) p d", f=frame_num - 1
                )
                delta_xy = self.delta_xy_fc(other_point_queries)
                delta_xy = (delta_xy.sigmoid() - 0.5) * self.offset_ratio
                other_points = other_points + delta_xy
                tmp_result = rearrange(
                    other_points, "(b f) p d -> b f p d", b=batch_size, f=frame_num - 1
                )
                tmp_coords.append(tmp_result)
                delta_query = self.delta_query_fc(other_point_feats)
                cur_queries = cur_queries + delta_query
            tmp_coords = torch.stack(tmp_coords, 1) * 224.0
            output_coords.append(tmp_coords)

            if i == self.up_iter_num:
                break
            # add new mid points
            other_points = add_mid_points(other_points)
            new_qeuries = total_queries[:, cur_point_num : cur_point_num * 2]
            cur_point_num = cur_point_num * 2
            old_queries = cur_queries
            cur_queries = torch.zeros(
                (
                    cur_queries.shape[0],
                    cur_point_num,
                    cur_queries.shape[2],
                )
            ).cuda()

            cur_queries[:, ::2] = new_qeuries
            cur_queries[:, 1::2] = old_queries

        return output_coords

    def _get_img_scrs(self, img: torch.Tensor, layernorm: nn.LayerNorm):
        b, c, h, w = img.shape
        feats = self.featup(img)
        feats = rearrange(feats, "b c h w -> b (h w) c")
        feats = self.featup_fc(feats)
        feats = rearrange(feats, "b (h w) c -> b c h w", h=h, w=w)
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
            pos_embeds.append(self.pos_2d(src, padding_masks[-1]))
        src_flatten = []
        spatial_shapes = []
        pos_embed_flatten = []

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

        return (
            src_flatten,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            pos_embed_flatten,
        )


class IterVideoLoss(nn.Module):
    def __init__(self, ras_loss: RasLoss, gt_rasterizer: SoftPolygon):
        super(IterVideoLoss, self).__init__()
        self.video_loss = VideoLoss(ras_loss, gt_rasterizer)

    def forward(self, pred_points: torch.Tensor, gt_masks: torch.Tensor):
        avg_loss = 0
        last_iou = 0
        for i in range(len(pred_points)):
            loss, iou = self.video_loss(pred_points[i], gt_masks)
            avg_loss += loss
            last_iou = iou
        return avg_loss, last_iou


class IterVideoLoss(nn.Module):
    def __init__(self, ras_loss: RasLoss, gt_rasterizer: SoftPolygon):
        super(IterVideoLoss, self).__init__()
        self.video_loss = VideoLoss(ras_loss, gt_rasterizer)

    def forward(self, pred_points: torch.Tensor, gt_masks: torch.Tensor):
        avg_loss = 0
        last_iou = 0
        for i in range(len(pred_points)):
            loss, iou = self.video_loss(pred_points[i], gt_masks)
            avg_loss += loss
            last_iou = iou
        avg_loss /= len(pred_points)
        return avg_loss, last_iou


class CoEvalerIter:
    def __init__(
        self,
        raw_set: list[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
        gt_rasterizer: SoftPolygon,
        frame_num: int = 10,
    ):
        self.raw_set = raw_set
        self.frame_num = frame_num
        self.gt_rasterizer = gt_rasterizer
        self.total_pred_results = []

    def eval_one_video(self, video_idx: int, model: CotrackerIter):
        video_data = self.raw_set[video_idx]
        total_frame_num = len(video_data)
        img_list = []
        mask_list = []
        point_list = []
        for frame_idx in range(total_frame_num):
            img, mask, point = video_data[frame_idx]
            img_list.append(img)
            mask_list.append(mask)
            point_list.append(point)
        total_masks = torch.stack(mask_list)
        total_points = torch.stack(point_list)
        total_pred_points = torch.zeros_like(total_points)
        total_pred_points[0] = total_points[0]
        interval = (total_frame_num - 1) // (self.frame_num - 1)
        offset_min = -(interval - 1)
        offset_max = (total_frame_num - 1) - (self.frame_num - 1) * interval
        done_idx_list = [0]
        for offset in range(offset_min, offset_max + 1):
            selected_frame_idxs = [0]
            for i in range(1, self.frame_num):
                selected_frame_idxs.append(i * interval + offset)
            selected_frame_idxs = sorted(selected_frame_idxs)
            img_list = []
            mask_list = []
            point_list = []
            fir_point = video_data[0][2]
            for frame_idx in selected_frame_idxs:
                img, mask, point = video_data[frame_idx]
                img_list.append(img)
                mask_list.append(mask)
                point_list.append(fir_point)
            imgs = torch.stack(img_list).unsqueeze(0)
            masks = torch.stack(mask_list).unsqueeze(0)
            points = torch.stack(point_list).unsqueeze(0)
            with torch.no_grad():
                model.eval()
                pred_points = model(imgs.cuda(), points.cuda())[-1]
                pred_points = pred_points.clamp(0, 223)
            pred_points = pred_points[0][-1]
            for result_idx, frame_idx in enumerate(selected_frame_idxs[1:]):
                if frame_idx in done_idx_list:
                    continue
                done_idx_list.append(frame_idx)
                total_pred_points[frame_idx] = pred_points[result_idx]
        total_iou = get_batch_average_bou_iou(
            total_pred_points.cuda(),
            total_masks.cuda(),
            self.gt_rasterizer,
        )
        return total_iou, total_pred_points

    def compute_avg_iou(self):
        iou_list = [iou for iou, _ in self.total_pred_results]
        return sum(iou_list) / len(iou_list)

    def eval_all_video(self, model: CotrackerIter, use_tqdm=False):
        self.total_pred_results = []
        video_num = len(self.raw_set)
        if use_tqdm:
            video_idx_list = tqdm(range(video_num))
        else:
            video_idx_list = range(video_num)
        for video_idx in video_idx_list:
            iou, pred_points = self.eval_one_video(video_idx, model)
            self.total_pred_results.append((iou, pred_points))
        return self.compute_avg_iou()

    def show_iou_distribution(self):
        iou_list = [iou.item() for iou, _ in self.total_pred_results]
        plt.hist(iou_list, bins=20)
        plt.show()

    def show_video_result(
        self,
        video_idx: int,
        gt_rasterizer: SoftPolygon,
        img_per_line=5,
        mask_alpha=0.2,
    ):
        video_data = self.raw_set[video_idx]
        frame_num = len(video_data)
        line_num = (frame_num - 1) // img_per_line + 1
        total_pred_points = self.total_pred_results[video_idx][1]
        plt.figure(figsize=(img_per_line * 4, line_num * 4))
        for i, pred_point in enumerate(total_pred_points):
            plt.subplot(line_num, img_per_line, i + 1)
            plt.axis("off")
            img, mask, point = video_data[i]
            plt.imshow(normalize(img).permute(1, 2, 0))
            iou = get_batch_average_bou_iou(
                pred_point.unsqueeze(0).cuda(), mask.unsqueeze(0).cuda(), gt_rasterizer
            ).item()
            plt.title(f"IoU: {iou:.4f}")
            plt.scatter(pred_point[:, 0], pred_point[:, 1], c="r", s=5)
            plt.plot(pred_point[:, 0], pred_point[:, 1], c="r")
            plt.imshow(mask, alpha=mask_alpha)
        plt.show()


class DAVIS_Windows(Dataset):
    def __init__(
        self,
        raw_set: list[list[tuple[torch.Tensor, torch.Tensor]]],
        window_size: int,
    ):
        self.raw_set = raw_set
        self.window_size = window_size

    def __len__(self):
        return len(self.raw_set)

    def __getitem__(self, idx: int):
        video_data = self.raw_set[idx]
        video_len = len(video_data)
        start_min = 0
        start_max = video_len - self.window_size - 1
        if start_max < start_min:
            start_idx = 0
            window_size = video_len
        else:
            start_idx = random.randint(start_min, start_max)
            window_size = self.window_size
        img_list = []
        mask_list = []
        point_list = []
        for i in range(window_size):
            img, mask, point = video_data[start_idx + i]
            img_list.append(img)
            mask_list.append(mask)
            point_list.append(point)
        imgs = torch.stack(img_list, 0)
        masks = torch.stack(mask_list, 0)
        points = torch.stack(point_list, 0)
        return imgs, masks, points


class DAVIS_Windows(Dataset):
    def __init__(
        self,
        raw_set: list[list[tuple[torch.Tensor, torch.Tensor]]],
        window_size: int,
    ):
        self.raw_set = raw_set
        self.window_size = window_size

    def __len__(self):
        return len(self.raw_set)

    def __getitem__(self, idx: int):
        video_data = self.raw_set[idx]
        video_len = len(video_data)
        start_min = 0
        start_max = video_len - self.window_size - 1
        if start_max < start_min:
            start_idx = 0
            window_size = video_len
        else:
            start_idx = random.randint(start_min, start_max)
            window_size = self.window_size
        img_list = []
        mask_list = []
        point_list = []
        for i in range(window_size):
            img, mask, point = video_data[start_idx + i]
            img_list.append(img)
            mask_list.append(mask)
            point_list.append(point)
        imgs = torch.stack(img_list, 0)
        masks = torch.stack(mask_list, 0)
        points = torch.stack(point_list, 0)
        return imgs, masks, points


class CoWinEvaler:
    def __init__(
        self,
        raw_set: list[list[tuple[torch.Tensor, torch.Tensor]]],
        gt_rasterizer: SoftPolygon,
        frame_num: int = 10,
    ) -> None:
        self.raw_set = raw_set
        self.gt_rasterizer = gt_rasterizer
        self.frame_num = frame_num
        self.total_pred_results = []

    def eval_one_video(self, video_idx: int, model: Cotracker):
        video_data = self.raw_set[video_idx]
        video_len = len(video_data)
        img_list = []
        mask_list = []
        point_list = []
        for frame_idx in range(video_len):
            img, mask, point = video_data[frame_idx]
            img_list.append(img)
            mask_list.append(mask)
            point_list.append(point)
        total_imgs = torch.stack(img_list, 0).unsqueeze(0)
        total_masks = torch.stack(mask_list, 0).unsqueeze(0)
        total_points = torch.stack(point_list, 0).unsqueeze(0)
        fir_point = total_points[:, 0]
        cur_points = repeat(fir_point, "b p xy -> b f p xy", f=self.frame_num)
        total_pred_results = fir_point.unsqueeze(0)
        half_win_size = self.frame_num // 2
        sliding_win_num = (video_len - self.frame_num) // half_win_size + 1
        with torch.no_grad():
            for i in range(0, sliding_win_num):
                cur_imgs = total_imgs[
                    :, i * half_win_size : i * half_win_size + self.frame_num
                ]
                cur_imgs = cur_imgs.cuda()
                cur_points = cur_points.cuda()
                cur_pred_points = model(cur_imgs, cur_points)[:, -1]
                new_points = repeat(
                    cur_pred_points[:, -1], "b p xy -> b f p xy", f=half_win_size
                )
                old_points = cur_pred_points[:, -half_win_size:]
                cur_points = torch.cat([old_points, new_points], 1)
                if i == 0:
                    total_pred_results = torch.cat(
                        [total_pred_results, cur_pred_points.cpu()], 1
                    )
                else:
                    total_pred_results = torch.cat(
                        [total_pred_results, old_points.cpu()], 1
                    )
            cur_pred_len = total_pred_results.shape[1]
            if cur_pred_len < video_len:
                cur_imgs = total_imgs[:, -self.frame_num :]
                cur_imgs = cur_imgs.cuda()
                left_len = video_len - cur_pred_len
                new_points = repeat(
                    total_pred_results[:, -1], "b p xy -> b f p xy", f=left_len
                )
                old_point_num = self.frame_num - left_len
                old_points = total_pred_results[:, -old_point_num:]
                cur_points = torch.cat([old_points, new_points], 1)
                cur_points = cur_points.cuda()
                cur_pred_points = model(cur_imgs, cur_points)[:, -1]
                left_pred_points = cur_pred_points[:, -left_len:]
                total_pred_results = torch.cat(
                    [total_pred_results, left_pred_points.cpu()], 1
                )
        total_pred_results = total_pred_results.squeeze(0)
        total_masks = total_masks.squeeze(0)
        total_iou = get_batch_average_bou_iou(
            total_pred_results.cuda(),
            total_masks.cuda(),
            self.gt_rasterizer,
        ).item()
        return total_iou, total_pred_results

    def compute_avg_iou(self):
        iou_list = [iou for iou, _ in self.total_pred_results]
        return sum(iou_list) / len(iou_list)

    def eval_all_videos(self, model: Cotracker, use_tqdm: bool = False):
        self.total_pred_results = []
        video_num = len(self.raw_set)
        if use_tqdm:
            video_iter = tqdm(range(video_num))
        else:
            video_iter = range(video_num)
        for video_idx in video_iter:
            iou, pred_results = self.eval_one_video(video_idx, model)
            self.total_pred_results.append((iou, pred_results))
        return self.compute_avg_iou()

    def show_iou_distribution(self):
        iou_list = [iou for iou, _ in self.total_pred_results]
        plt.hist(iou_list, bins=20)
        plt.show()

    def show_video_result(
        self,
        video_idx: int,
        gt_rasterizer: SoftPolygon,
        img_per_line=5,
        mask_alpha=0.2,
    ):
        video_data = self.raw_set[video_idx]
        frame_num = len(video_data)
        line_num = (frame_num - 1) // img_per_line + 1
        total_pred_points = self.total_pred_results[video_idx][1]
        plt.figure(figsize=(img_per_line * 4, line_num * 4))
        for i, pred_point in enumerate(total_pred_points):
            plt.subplot(line_num, img_per_line, i + 1)
            plt.axis("off")
            img, mask, point = video_data[i]
            plt.imshow(normalize(img).permute(1, 2, 0))
            iou = get_batch_average_bou_iou(
                pred_point.unsqueeze(0).cuda(), mask.unsqueeze(0).cuda(), gt_rasterizer
            ).item()
            plt.title(f"IoU: {iou:.4f}")
            plt.scatter(pred_point[:, 0], pred_point[:, 1], c="r", s=5)
            plt.plot(pred_point[:, 0], pred_point[:, 1], c="r")
            plt.imshow(mask, alpha=mask_alpha)


class CotrackerLight(nn.Module):
    def __init__(
        self,
        point_num: int,
        refine_iter: int = 1,
        freeze_featup: bool = True,
        d_model: int = 128,
        n_heads: int = 8,
        n_points: int = 4,
        d_ffn: int = 512,
        n_layers: int = 1,
        offset_ratio: float = 1.0,
    ):
        super(CotrackerLight, self).__init__()
        self.refine_iter = refine_iter
        self.featup = torch.hub.load(
            "mhamilton723/FeatUp",
            "dino16",
            use_norm=True,
        )
        if freeze_featup:
            for param in self.featup.parameters():
                param.requires_grad = False
        d_featup = 384
        if d_featup != d_model:
            self.featup_fc = nn.Linear(d_featup, d_model)
        else:
            self.featup_fc = nn.Identity()
        self.medium_level_size = [14, 28, 56, 112]
        n_levels = len(self.medium_level_size) + 1
        self.offset_ratio = offset_ratio
        self.pos_2d = IMGPositionEmbeddingSine(d_model)
        self.pos_1d = PositionalEncoding(d_model)
        self.level_pos = nn.Embedding(n_levels, d_model)
        nn.init.xavier_uniform_(self.level_pos.weight)

        self.query_embed = nn.Embedding(point_num, d_model)
        nn.init.xavier_uniform_(self.query_embed.weight)
        dec_layer = DeformableTransformerDecoderLayer(
            d_model=d_model,
            d_ffn=d_ffn,
            n_levels=n_levels,
            n_heads=n_heads,
            n_points=n_points,
        )
        self.def_dec_layernorm = nn.LayerNorm(d_model)
        self.def_decoder = DeformableTransformerDecoder(
            decoder_layer=dec_layer,
            num_layers=n_layers,
        )
        self.fir_layernorm = nn.LayerNorm(d_model)
        self.fir_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model,
                n_heads,
                d_ffn,
                batch_first=True,
            ),
            num_layers=n_layers,
        )
        self.other_layernorm = nn.LayerNorm(d_model)
        self.other_decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model,
                n_heads,
                d_ffn,
                batch_first=True,
            ),
            num_layers=n_layers,
        )
        self.delta_xy_fc = MLP(d_model, d_model, 2, 2)

    def forward(self, imgs: torch.Tensor, points: torch.Tensor):
        batch_size, frame_num, _, img_h, img_w = imgs.shape
        _, _, point_num, _ = points.shape

        points = points / 224.0

        fir_img = imgs[:, 0]
        fir_points = points[:, 0]
        (
            fir_src_flatten,
            fir_spatial_shapes,
            fir_level_start_index,
            fir_valid_ratios,
            fir_pos_embed_flatten,
        ) = self._get_img_scrs(fir_img, self.def_dec_layernorm)
        fir_src_flatten += fir_pos_embed_flatten
        fir_query_embed = repeat(self.query_embed.weight, "p d -> b p d", b=batch_size)
        fir_point_feats, _ = self.def_decoder(
            fir_query_embed,
            fir_points,
            fir_src_flatten,
            fir_spatial_shapes,
            fir_level_start_index,
            fir_valid_ratios,
        )
        fir_point_feats = self.fir_layernorm(fir_point_feats)
        fir_query = self.pos_1d(fir_point_feats)
        fir_memory = self.fir_encoder(fir_query)

        other_imgs = imgs[:, 1:]
        other_points = points[:, 1:]
        other_imgs = rearrange(other_imgs, "b f c h w -> (b f) c h w")
        other_points = rearrange(other_points, "b f p d -> (b f) p d")
        (
            other_src_flatten,
            other_spatial_shapes,
            other_level_start_index,
            other_valid_ratios,
            other_pos_embed_flatten,
        ) = self._get_img_scrs(other_imgs, self.def_dec_layernorm)
        other_src_flatten += other_pos_embed_flatten
        other_queries = repeat(
            self.query_embed.weight, "p d -> b p d", b=batch_size * (frame_num - 1)
        )

        output_coords = []
        other_point_feats, _ = self.def_decoder(
            other_queries,
            other_points,
            other_src_flatten,
            other_spatial_shapes,
            other_level_start_index,
            other_valid_ratios,
        )
        other_point_queries = rearrange(
            other_point_feats, "(b f)p d -> b (f p) d", f=frame_num - 1
        )
        other_point_queries = self.other_layernorm(other_point_queries)
        other_point_queries = self.pos_1d(other_point_queries)
        other_point_queries = self.other_decoder(
            tgt=other_point_queries,
            memory=fir_memory,
        )
        other_point_queries = rearrange(
            other_point_queries, "b (f p) d -> (b f) p d", f=frame_num - 1
        )
        delta_xy = self.delta_xy_fc(other_point_queries)
        delta_xy = (delta_xy.sigmoid() - 0.5) * self.offset_ratio
        other_points = other_points + delta_xy
        result = rearrange(
            other_points, "(b f) p d -> b f p d", b=batch_size, f=frame_num - 1
        )
        result = result * 224.0
        output_coords.append(result)
        output_coords = torch.stack(output_coords, 1)
        return output_coords

    def _get_img_scrs(self, img: torch.Tensor, layernorm: nn.LayerNorm):
        b, c, h, w = img.shape
        feats = self.featup(img)
        feats = rearrange(feats, "b c h w -> b (h w) c")
        feats = self.featup_fc(feats)
        feats = rearrange(feats, "b (h w) c -> b c h w", h=h, w=w)
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
            pos_embeds.append(self.pos_2d(src, padding_masks[-1]))
        src_flatten = []
        spatial_shapes = []
        pos_embed_flatten = []

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

        return (
            src_flatten,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            pos_embed_flatten,
        )
