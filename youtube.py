import json
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Dataset
import random
import torchvision
from Loader_17 import normalize
import torchvision.tv_tensors
from preprocess_utensils import get_boundary_points, uniform_sample_points
from einops import repeat
from tqdm import tqdm


class FramesMeta:
    def __init__(
        self,
        frame_names: list[str],
        video_name: str,
        object_idx: int,
        is_train: bool,
    ) -> None:
        self.frame_names = frame_names
        self.video_name = video_name
        self.is_train = is_train
        self.object_idx = object_idx

    def __len__(self) -> int:
        return len(self.frame_names)

    def __getitem__(self, frame_idx: int) -> tuple[np.array, np.array]:
        frame_prefix = f"youtube/{'train' if self.is_train else 'valid'}"
        img_path = (
            Path(frame_prefix)
            / "JPEGImages"
            / self.video_name
            / f"{self.frame_names[frame_idx]}.jpg"
        )
        annotations_path = (
            Path(frame_prefix)
            / "Annotations"
            / self.video_name
            / f"{self.frame_names[frame_idx]}.png"
        )
        img = Image.open(img_path)
        mask = Image.open(annotations_path)
        img = np.array(img)
        mask = np.array(mask)
        mask = mask == self.object_idx + 1
        mask = mask.astype(np.uint8)
        return img, mask


class VideoMeta:
    def __init__(
        self,
        video_meta: dict,
        video_name: str,
        is_train: bool,
    ) -> None:
        self.video_dict = video_meta
        self.video_name = video_name
        self.object_names = list(video_meta.keys())
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.object_names)

    def __getitem__(self, object_idx: int) -> FramesMeta:
        return FramesMeta(
            self.video_dict[self.object_names[object_idx]]["frames"],
            self.video_name,
            object_idx,
            self.is_train,
        )


class YoutubeMeta:
    def __init__(self, is_train: bool) -> None:
        self.is_train = is_train
        if is_train:
            meta_json_path = "youtube/train/meta.json"
        else:
            meta_json_path = "youtube/valid/meta.json"
        with open(meta_json_path, "r") as f:
            self.meta = json.load(f)
        self.video_names = list(self.meta["videos"].keys())

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx) -> VideoMeta:
        video_name = self.video_names[idx]
        video_meta = self.meta["videos"][video_name]["objects"]
        return VideoMeta(video_meta, video_name, self.is_train)

class YoutubeDataset(Dataset):
    def __init__(
        self,
        is_train: bool,
        point_num: int,
        frame_num: int,
        is_eval: bool,
    ) -> None:
        self.youtube_meta = YoutubeMeta(is_train)
        self.point_num = point_num
        self.frame_num = frame_num
        self.is_eval = is_eval
        self.img_transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize((224, 224)),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.mask_transform = v2.Compose(
            [
                v2.Resize((224, 224), interpolation="nearest"),
            ]
        )

    def __len__(self) -> int:
        return len(self.youtube_meta)

    def __getitem__(self, idx) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        video_meta = self.youtube_meta[idx]
        object_idx = random.randint(0, len(video_meta) - 1)
        fir_mask = video_meta[object_idx][0][1]
        count = 0
        while fir_mask.sum() == 0:
            if count > 5:
                # reselect the video_idx
                idx = random.randint(0, len(self.youtube_meta) - 1)
                video_meta = self.youtube_meta[idx]
            object_idx = random.randint(0, len(video_meta) - 1)
            fir_mask = video_meta[object_idx][0][1]
            # resample if the first frame has no mask
            count += 1
        total_frame_num = len(video_meta[object_idx])
        selected_frame_idxs = []
        if total_frame_num <= self.frame_num:
            selected_frame_idxs = list(range(total_frame_num))
        else:
            # uniformly sample frame_num frames, and the first frame is always included
            selected_frame_idxs.append(0)
            interval = (total_frame_num - 1) // (self.frame_num - 1)
            if self.is_eval:
                offset = 0
            else:
                offset_min = -(interval - 1)
                offset_max = (total_frame_num - 1) - (self.frame_num - 1) * interval
                offset = random.randint(offset_min, offset_max)
            for i in range(1, self.frame_num):
                selected_frame_idxs.append(interval * i + offset)
        img_list = []
        mask_list = []
        for idx in selected_frame_idxs:
            img, mask = video_meta[object_idx][idx]
            img = self.img_transform(img)
            mask = self.mask_transform(torchvision.tv_tensors.Mask(mask))
            img_list.append(img)
            mask_list.append(mask)

        def _get_boundary_points(mask: torch.Tensor, point_num: int) -> torch.Tensor:
            boundary = get_boundary_points(mask.numpy())
            boundary = uniform_sample_points(boundary, point_num)
            boundary = torch.tensor(boundary, dtype=torch.float32)
            return boundary

        fir_mask = mask_list[0]
        fir_point = _get_boundary_points(fir_mask, self.point_num)
        points = repeat(fir_point, "n p -> f n p", f=len(mask_list))
        imgs = torch.stack(img_list)
        masks = torch.stack(mask_list)
        return imgs, masks, points
