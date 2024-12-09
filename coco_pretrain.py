import pathlib
import torch
import torch.utils.data
import torchvision
import torch.nn as nn

from torchvision import datasets, tv_tensors
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from Loader_17 import normalize
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from deform_video import DeformLightVideoPos
from deform_model import get_batch_average_bou_iou
from polygon import RasLoss, SoftPolygon
import torch.optim as optim
import cv2 as cv
from preprocess_utensils import get_boundary_points, uniform_sample_points
import gc


class PreTransformer:
    def __init__(self) -> None:
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

    def __call__(
        self, img: Image.Image, mask: tv_tensors.Mask
    ) -> tuple[tv_tensors.Image, tv_tensors.Mask]:
        img = self.img_transform(img)
        mask = self.mask_transform(mask)
        return img, mask


class AffineFrameGenerator:
    def __init__(self) -> None:
        self.big_shift_params = dict(
            angle=(-70, 70),
            translate=(0.1, 0.3),
            scale=(0.5, 1.5),
            shear=(-30, 30),
        )
        self.small_shift_params = dict(
            angle=(-10, 10),
            translate=(-0.05, 0.05),
            scale=(-0.1, 0.1),
            shear=(-10, 10),
        )

    def __call__(self, img: tv_tensors.Image, mask: tv_tensors.Mask) -> tuple[
        tuple[tv_tensors.Image, tv_tensors.Mask],
        tuple[tv_tensors.Image, tv_tensors.Mask],
    ]:
        shift_angle = random.uniform(*self.big_shift_params["angle"])
        shift_translate = (
            random.uniform(*self.big_shift_params["translate"]),
            random.uniform(*self.big_shift_params["translate"]),
        )
        shift_scale = random.uniform(*self.big_shift_params["scale"])
        shift_shear = random.uniform(*self.big_shift_params["shear"])
        pre_frame_img = v2.functional.affine(
            img,
            angle=shift_angle,
            translate=shift_translate,
            scale=shift_scale,
            shear=shift_shear,
        )
        pre_frame_mask = v2.functional.affine(
            mask,
            angle=shift_angle,
            translate=shift_translate,
            scale=shift_scale,
            shear=shift_shear,
        )

        next_shift_angle = shift_angle + random.uniform(
            *self.small_shift_params["angle"]
        )
        next_shift_translate = (
            shift_translate[0] + random.uniform(*self.small_shift_params["translate"]),
            shift_translate[1] + random.uniform(*self.small_shift_params["translate"]),
        )
        next_shift_scale = shift_scale + random.uniform(
            *self.small_shift_params["scale"]
        )
        next_shift_shear = shift_shear + random.uniform(
            *self.small_shift_params["shear"]
        )
        cur_frame_img = v2.functional.affine(
            img,
            angle=next_shift_angle,
            translate=next_shift_translate,
            scale=next_shift_scale,
            shear=next_shift_shear,
        )
        cur_frame_mask = v2.functional.affine(
            mask,
            angle=next_shift_angle,
            translate=next_shift_translate,
            scale=next_shift_scale,
            shear=next_shift_shear,
        )
        return (pre_frame_img, pre_frame_mask), (cur_frame_img, cur_frame_mask)


class CocoPretrainDataset(Dataset):
    def __init__(
        self,
        dataset: datasets.CocoDetection,
        use_tqdm=False,
        small_subset=False,
        point_num=64,
        dataset_size=None,
    ) -> None:
        self.point_num = point_num
        self.dataset_size = dataset_size
        self.pre_transformer = PreTransformer()
        self.frame_generator = AffineFrameGenerator()
        # only keep the images with masks
        dataset = datasets.wrap_dataset_for_transforms_v2(
            dataset, target_keys=["masks"]
        )
        if small_subset:
            dataset = torch.utils.data.Subset(dataset, range(100))
        # if use_tqdm:
        #     dataset = tqdm(dataset)
        # self.dataset = [item for item in dataset if "masks" in item[1]]
        self.dataset = dataset
        self.gt_rasterizer = SoftPolygon(1, "hard_mask").cuda()

    def __len__(self) -> int:
        if self.dataset_size is not None:
            return self.dataset_size
        else:
            return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # randomly choose one image
        def _get_new_idx():
            while True:
                random.seed()
                idx = random.randint(0, len(self.dataset) - 1)
                sample = self.dataset[idx]
                if "masks" in sample[1]:
                    return idx

        if self.dataset_size is not None:
            idx = _get_new_idx()
        sample = self.dataset[idx]
        while "masks" not in sample[1]:
            idx = _get_new_idx()
            sample = self.dataset[idx]
        img, target = sample
        mask = target["masks"]
        resized_img, raw_resized_mask = self.pre_transformer(img, mask)

        def _get_boundary_points(mask: torch.Tensor) -> torch.Tensor:
            boundary = get_boundary_points(mask.numpy())
            boundary = uniform_sample_points(boundary, self.point_num)
            boundary = torch.tensor(boundary, dtype=torch.float32)
            return boundary

        counter = 0
        while True:
            if counter > 10:
                idx = _get_new_idx()
                sample = self.dataset[idx]
                img, target = sample
                mask = target["masks"]
                resized_img, raw_resized_mask = self.pre_transformer(img, mask)
            counter += 1
            # randomly choose one mask until we find a non-empty mask
            rand_mask_idx = random.randint(0, len(mask) - 1)
            resized_mask = tv_tensors.Mask(raw_resized_mask[rand_mask_idx])
            (pre_frame_img, pre_frame_mask), (cur_frame_img, cur_frame_mask) = (
                self.frame_generator(resized_img, resized_mask)
            )
            if cur_frame_mask.sum() > 0 and pre_frame_mask.sum() > 0:
                first_boundary = _get_boundary_points(resized_mask)
                prev_boundary = _get_boundary_points(pre_frame_mask)
                curr_boundary = _get_boundary_points(cur_frame_mask)
                if not (
                    first_boundary.shape
                    == prev_boundary.shape
                    == curr_boundary.shape
                    == (self.point_num, 2)
                ):
                    continue
                break

        cur_frame_mask = self.gt_rasterizer(
            curr_boundary.unsqueeze(0).cuda(),
            cur_frame_mask.shape[0],
            cur_frame_mask.shape[1],
        )
        cur_frame_mask[cur_frame_mask == -1] = 0
        cur_frame_mask = cur_frame_mask.squeeze(0)
        # move curr_frame_mask to the same device as curr_frame_img
        cur_frame_mask = cur_frame_mask.to(cur_frame_img.device)

        return {
            "first_img": resized_img,
            "first_mask": resized_mask,
            "first_boundary": first_boundary,
            "prev_img": pre_frame_img,
            "prev_mask": pre_frame_mask,
            "prev_boundary": prev_boundary,
            "curr_img": cur_frame_img,
            "curr_mask": cur_frame_mask,
            "curr_boundary": curr_boundary,
        }


class AffineVideoGenerator:
    def __init__(self, frame_num: int) -> None:
        self.shift_params = dict(
            angle=(-10, 10),
            translate=(-0.05, 0.05),
            scale=(-0.1, 0.1),
            shear=(-10, 10),
        )
        self.frame_num = frame_num

    def __call__(self, img: tv_tensors.Image, mask: tv_tensors.Mask) -> tuple[
        tuple[tv_tensors.Image, tv_tensors.Mask],
        tuple[tv_tensors.Image, tv_tensors.Mask],
    ]:
        shift_angle = 0
        shift_translate = (0, 0)
        shift_scale = 1
        shift_shear = 0
        imgs = [img]
        masks = [mask]
        for i in range(1, self.frame_num):
            shift_angle += random.uniform(*self.shift_params["angle"])
            shift_translate = (
                shift_translate[0] + random.uniform(*self.shift_params["translate"]),
                shift_translate[1] + random.uniform(*self.shift_params["translate"]),
            )
            shift_scale += random.uniform(*self.shift_params["scale"])
            shift_shear += random.uniform(*self.shift_params["shear"])
            if shift_scale < 0.1:
                shift_scale = 0.1
            imgs.append(
                v2.functional.affine(
                    img,
                    angle=shift_angle,
                    translate=shift_translate,
                    scale=shift_scale,
                    shear=shift_shear,
                )
            )
            masks.append(
                v2.functional.affine(
                    mask,
                    angle=shift_angle,
                    translate=shift_translate,
                    scale=shift_scale,
                    shear=shift_shear,
                )
            )
        return (imgs, masks)


class CocoVideoDataset(Dataset):
    def __init__(
        self,
        dataset: datasets.CocoDetection,
        small_subset=False,
        point_num=32,
        frame_num=10,
        dataset_size=None,
    ) -> None:
        super().__init__()
        self.pre_transformer = PreTransformer()
        self.video_generator = AffineVideoGenerator(frame_num)
        self.point_num = point_num
        self.dataset = datasets.wrap_dataset_for_transforms_v2(
            dataset, target_keys=["masks"]
        )
        if small_subset:
            self.dataset = torch.utils.data.Subset(self.dataset, range(10))
        self.gt_rasterizer = SoftPolygon(1, "hard_mask").cuda()
        self.dataset_size = dataset_size

    def __len__(self) -> int:
        if self.dataset_size is not None:
            return self.dataset_size
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # randomly choose one image
        def _get_new_idx():
            while True:
                random.seed()
                idx = random.randint(0, len(self.dataset) - 1)
                sample = self.dataset[idx]
                if "masks" in sample[1]:
                    return idx

        def _get_boundary_points(mask: torch.Tensor) -> torch.Tensor:
            boundary = get_boundary_points(mask.numpy())
            boundary = uniform_sample_points(boundary, self.point_num)
            boundary = torch.tensor(boundary, dtype=torch.float32)
            return boundary

        while True:
            idx = _get_new_idx()
            sample = self.dataset[idx]
            img, target = sample
            mask = target["masks"]
            resized_img, resized_mask = self.pre_transformer(img, mask)
            rand_mask_idx = random.randint(0, len(resized_mask) - 1)
            resized_mask = tv_tensors.Mask(resized_mask[rand_mask_idx])
            imgs, masks = self.video_generator(resized_img, resized_mask)
            # check if all masks are valid
            if all(mask.sum() > 0 for mask in masks):
                points = []
                for mask in masks:
                    boundary_points = _get_boundary_points(mask)
                    points.append(boundary_points)
                # check if all points' shape is (point_num, 2)
                if all(point.shape == (self.point_num, 2) for point in points):
                    break
        imgs = torch.stack(imgs)
        masks = torch.stack(masks)
        points = torch.stack(points)
        return imgs, masks, points