import json
from PIL import Image
import torch
import numpy as np
import torchvision.transforms as transforms
from tenLoader import normalize
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from polygon import SoftPolygon
from preprocess_utensils import get_boundary_iou
from torchvision.transforms import InterpolationMode
import random
from einops import repeat
from preprocess_utensils import get_boundary_points, uniform_sample_points


def reserve_color(anno_array: np.ndarray, color: list[int]) -> np.ndarray:
    color = np.array(color)
    mask = np.all(anno_array == color, axis=-1)
    return mask


class DAVIS_Rawset:
    def __init__(self, is_train=True) -> None:
        if is_train:
            data_type = "train"
        else:
            data_type = "val"
        data_path = f"./2017/{data_type}_video_datas.json"
        video_datas = json.load(open(data_path, "r"))
        self.data_set = []
        img_transforms = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        anno_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )
        for video_name, video_data in video_datas.items():
            img_paths = video_data["img_paths"]
            anno_paths = video_data["anno_paths"]
            anno_colors = video_data["anno_colors"]
            for anno_color in anno_colors:
                self.data_set.append([])
                for img_path, anno_path in zip(img_paths, anno_paths):
                    img_tensor = img_transforms(Image.open(img_path))
                    anno_img = Image.open(anno_path).convert("RGB")
                    anno_array = np.array(anno_img)
                    anno_mask = reserve_color(anno_array, anno_color)
                    anno_tensor = anno_transform(Image.fromarray(anno_mask))
                    anno_tensor = anno_tensor.squeeze(0)
                    self.data_set[-1].append((img_tensor, anno_tensor))

    def get_item(self, video_idx, frame_idx):
        return self.data_set[video_idx][frame_idx]


class DAVIS_Dataset(Dataset):
    def __init__(self, raw_set: DAVIS_Rawset) -> None:
        super().__init__()
        self.raw_set = raw_set
        self.data = []
        for video_idx, video in enumerate(self.raw_set.data_set):
            for frame_idx in range(len(video) - 1):
                self.data.append(
                    (
                        video_idx,
                        frame_idx,
                        video[0],
                        video[frame_idx],
                        video[frame_idx + 1],
                    )
                )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_idx, frame_idx, fir_frame, pre_frame, cur_frame = self.data[idx]
        fir_img, fir_sgm = fir_frame
        pre_img, pre_sgm = pre_frame
        cur_img, cur_sgm = cur_frame
        return (
            video_idx,
            frame_idx,
            fir_img,
            fir_sgm,
            pre_img,
            pre_sgm,
            cur_img,
            cur_sgm,
        )


class DAVIS_Infer:
    def __init__(self, raw_set: DAVIS_Rawset) -> None:
        self.raw_set = raw_set
        self.hard_polygon = SoftPolygon(0.01, mode="hard_mask").cuda()
        self.total_results = []
        self.best_video_index = -1
        self.worst_video_index = -1

    def infer_video(self, model: torch.nn.Module, video_idx: int):
        infer_results = []
        model.eval()
        fir_img, fir_sgm = self.raw_set.data_set[video_idx][0]
        infer_results.append(None)
        pre_img, pre_sgm = self.raw_set.data_set[video_idx][0]
        with torch.no_grad():
            for frame_idx in range(1, len(self.raw_set.data_set[video_idx])):
                cur_img, cur_sgm = self.raw_set.data_set[video_idx][frame_idx]
                pred_bou = model(
                    fir_img.unsqueeze(0).cuda(),
                    fir_sgm.unsqueeze(0).cuda(),
                    pre_img.unsqueeze(0).cuda(),
                    pre_sgm.unsqueeze(0).cuda(),
                    cur_img.unsqueeze(0).cuda(),
                )
                pred_mask = self.hard_polygon(
                    pred_bou.cuda().float(),
                    224,
                    224,
                )
                pred_mask[pred_mask == -1] = 0
                infer_results.append(pred_bou.squeeze(0).cpu().detach().numpy())
                pre_img, pre_sgm = cur_img, pred_mask.squeeze(0)
        return infer_results

    def infer_model(self, model: torch.nn.Module):
        self.total_results = []
        for video_idx in range(len(self.raw_set.data_set)):
            infer_results = self.infer_video(model, video_idx)
            self.total_results.append(infer_results)
        return self.total_results

    def get_video_iou(self, video_idx: int) -> float:
        total_iou = 1
        for frame_idx in range(1, len(self.raw_set.data_set[video_idx])):
            cur_img, cur_sgm = self.raw_set.data_set[video_idx][frame_idx]
            iou = get_boundary_iou(
                cur_sgm.cpu().detach().numpy(),
                self.total_results[video_idx][frame_idx],
            )
            # print(f"Frame {frame_idx} IOU: {iou}")
            total_iou += iou
        return total_iou / len(self.raw_set.data_set[video_idx])

    def get_total_iou(self) -> float:
        total_iou = 0
        best_iou = 0
        worst_iou = 1
        for video_idx in range(len(self.raw_set.data_set)):
            iou = self.get_video_iou(video_idx)
            total_iou += iou
            if iou > best_iou:
                best_iou = iou
                self.best_video_index = video_idx
            if iou < worst_iou:
                worst_iou = iou
                self.worst_video_index = video_idx
        return total_iou / len(self.raw_set.data_set)

    def show_infer_results(
        self,
        video_idx: int,
        figsize=(10, 35),
        interval=5,
    ):
        infer_results = self.total_results[video_idx]
        raw_set = self.raw_set
        fir_img, fir_sgm = raw_set.data_set[video_idx][0]
        plot_height = len(infer_results) // (5 * interval) + 1

        plt.figure(figsize=figsize)
        plt.subplot(plot_height, 5, 1)
        plt.imshow(normalize(fir_img).permute(1, 2, 0))
        plt.imshow(fir_sgm, alpha=0.4, cmap="jet")
        plt.title(f"Frame {0}")
        plt.axis("off")
        figure_idx = 2
        for frame_idx in range(1, len(infer_results)):
            if frame_idx % interval != 0:
                continue
            cur_img, cur_sgm = raw_set.data_set[video_idx][frame_idx]
            plt.subplot(plot_height, 5, figure_idx)
            figure_idx += 1
            plt.imshow(normalize(cur_img).permute(1, 2, 0))
            infer_bou = infer_results[frame_idx]
            plt.plot(infer_bou[:, 0], infer_bou[:, 1], "r")
            plt.axis("off")
            plt.title(f"Frame {frame_idx}")
        plt.show()


class DAVIS_Rawset_LowMem:
    def __init__(self, is_train=True) -> None:
        if is_train:
            data_type = "train"
        else:
            data_type = "val"
        data_path = f"./2017/{data_type}_video_datas.json"
        video_datas = json.load(open(data_path, "r"))
        self.data_set = []
        self.img_transforms = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.anno_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )
        for video_name, video_data in video_datas.items():
            img_paths = video_data["img_paths"]
            anno_paths = video_data["anno_paths"]
            anno_colors = video_data["anno_colors"]
            for anno_color in anno_colors:
                self.data_set.append([])
                for img_path, anno_path in zip(img_paths, anno_paths):
                    self.data_set[-1].append((img_path, anno_path, anno_color))

    def get_item(self, video_idx, frame_idx):
        img_path, anno_path, anno_color = self.data_set[video_idx][frame_idx]
        img_tensor = self.img_transforms(Image.open(img_path))
        anno_img = Image.open(anno_path).convert("RGB")
        anno_array = np.array(anno_img)
        anno_mask = reserve_color(anno_array, anno_color)
        anno_tensor = self.anno_transform(Image.fromarray(anno_mask))
        anno_tensor = anno_tensor.squeeze(0)
        return img_tensor, anno_tensor


class DAVIS_Dataset_LowMem(Dataset):
    def __init__(self, raw_set: DAVIS_Rawset_LowMem) -> None:
        super().__init__()
        self.raw_set = raw_set
        self.data = []
        for video_idx, video in enumerate(self.raw_set.data_set):
            for frame_idx in range(len(video) - 1):
                self.data.append((video_idx, frame_idx))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_idx, frame_idx = self.data[idx]
        # get first frame
        fir_img, fir_sgm = self.raw_set.get_item(video_idx, 0)
        # get previous frame
        pre_img, pre_sgm = self.raw_set.get_item(video_idx, frame_idx)
        # get current frame
        cur_img, cur_sgm = self.raw_set.get_item(video_idx, frame_idx + 1)
        return (
            video_idx,
            frame_idx,
            fir_img,
            fir_sgm,
            pre_img,
            pre_sgm,
            cur_img,
            cur_sgm,
        )


class DAVIS_Infer_LowMem:
    def __init__(self, raw_set: DAVIS_Rawset_LowMem) -> None:
        self.raw_set = raw_set
        self.hard_polygon = SoftPolygon(0.01, mode="hard_mask").cuda()
        self.total_results = []
        self.best_video_index = -1
        self.worst_video_index = -1

    def infer_video(self, model: torch.nn.Module, video_idx: int):
        infer_results = []
        model.eval()
        fir_img, fir_sgm = self.raw_set.get_item(video_idx, 0)
        infer_results.append(None)
        pre_img, pre_sgm = self.raw_set.get_item(video_idx, 0)
        with torch.no_grad():
            for frame_idx in range(1, len(self.raw_set.data_set[video_idx])):
                cur_img, cur_sgm = self.raw_set.get_item(video_idx, frame_idx)
                pred_bou = model(
                    fir_img.unsqueeze(0).cuda(),
                    fir_sgm.unsqueeze(0).cuda(),
                    pre_img.unsqueeze(0).cuda(),
                    pre_sgm.unsqueeze(0).cuda(),
                    cur_img.unsqueeze(0).cuda(),
                )
                pred_mask = self.hard_polygon(
                    pred_bou.cuda().float(),
                    224,
                    224,
                )
                pred_mask[pred_mask == -1] = 0
                infer_results.append(pred_bou.squeeze(0).cpu().detach().numpy())
                pre_img, pre_sgm = cur_img, pred_mask.squeeze(0)
        return infer_results

    def infer_model(self, model: torch.nn.Module):
        self.total_results = []
        for video_idx in range(len(self.raw_set.data_set)):
            infer_results = self.infer_video(model, video_idx)
            self.total_results.append(infer_results)
        return self.total_results

    def get_video_iou(self, video_idx: int) -> float:
        total_iou = 1
        for frame_idx in range(1, len(self.raw_set.data_set[video_idx])):
            cur_img, cur_sgm = self.raw_set.get_item(video_idx, frame_idx)
            iou = get_boundary_iou(
                cur_sgm.cpu().detach().numpy(),
                self.total_results[video_idx][frame_idx],
            )
            # print(f"Frame {frame_idx} IOU: {iou}")
            total_iou += iou
        return total_iou / len(self.raw_set.data_set[video_idx])

    def get_total_iou(self) -> float:
        total_iou = 0
        best_iou = 0
        worst_iou = 1
        for video_idx in range(len(self.raw_set.data_set)):
            iou = self.get_video_iou(video_idx)
            total_iou += iou
            if iou > best_iou:
                best_iou = iou
                self.best_video_index = video_idx
            if iou < worst_iou:
                worst_iou = iou
                self.worst_video_index = video_idx
        return total_iou / len(self.raw_set.data_set)

    def show_infer_results(
        self,
        video_idx: int,
        figsize=(10, 35),
        interval=5,
    ):
        infer_results = self.total_results[video_idx]
        raw_set = self.raw_set
        fir_img, fir_sgm = raw_set.data_set[video_idx][0]
        plot_height = len(infer_results) // (5 * interval) + 1

        plt.figure(figsize=figsize)
        plt.subplot(plot_height, 5, 1)
        plt.imshow(normalize(fir_img).permute(1, 2, 0))
        plt.imshow(fir_sgm, alpha=0.4, cmap="jet")
        plt.title(f"Frame {0}")
        plt.axis("off")
        figure_idx = 2
        for frame_idx in range(1, len(infer_results)):
            if frame_idx % interval != 0:
                continue
            cur_img, cur_sgm = raw_set.data_set[video_idx][frame_idx]
            plt.subplot(plot_height, 5, figure_idx)
            figure_idx += 1
            plt.imshow(normalize(cur_img).permute(1, 2, 0))
            infer_bou = infer_results[frame_idx]
            plt.plot(infer_bou[:, 0], infer_bou[:, 1], "r")
            plt.axis("off")
            plt.title(f"Frame {frame_idx}")
        plt.show()


class DAVIS_Aug_Rawset:
    def __init__(self, is_train=True) -> None:
        if is_train:
            data_type = "train"
        else:
            data_type = "val"
        data_path = f"./2017/{data_type}_video_datas.json"
        video_datas = json.load(open(data_path, "r"))
        self.data_set = []
        img_transforms = transforms.Compose([])
        anno_transform = transforms.Compose([])
        for video_name, video_data in video_datas.items():
            img_paths = video_data["img_paths"]
            anno_paths = video_data["anno_paths"]
            anno_colors = video_data["anno_colors"]
            for anno_color in anno_colors:
                self.data_set.append([])
                for img_path, anno_path in zip(img_paths, anno_paths):
                    img_tensor = img_transforms(Image.open(img_path).convert("RGB"))
                    anno_img = Image.open(anno_path).convert("RGB")
                    anno_array = np.array(anno_img)
                    anno_mask = reserve_color(anno_array, anno_color)
                    anno_tensor = anno_transform(Image.fromarray(anno_mask))
                    # anno_tensor = anno_tensor.squeeze(0)
                    # print(np.array(anno_tensor).shape)
                    self.data_set[-1].append((img_tensor, anno_tensor))

    def get_item(self, video_idx, frame_idx):
        return self.data_set[video_idx][frame_idx]


def _get_boundary_points(mask: torch.Tensor, point_num: int) -> torch.Tensor:
    boundary = get_boundary_points(mask.numpy().astype(np.uint8))
    boundary = uniform_sample_points(boundary, point_num)
    boundary = torch.tensor(boundary, dtype=torch.float32)
    return boundary


class DAVIS_Aug_Dataset(Dataset):
    def __init__(
        self,
        raw_set: list,
        point_num: int,
        frame_num: int,
        is_eval: bool,
    ) -> None:
        super().__init__()
        # remove all the video with empty frame
        empty_video_idxs = []
        for video_idx, video_data in enumerate(raw_set):
            for frame_data in video_data:
                img, mask = frame_data
                mask = np.array(mask)
                if mask.sum() == 0:
                    empty_video_idxs.append(video_idx)
                    break
        self.data = []
        for video_idx, video_data in enumerate(raw_set):
            if video_idx in empty_video_idxs:
                continue
            self.data.append(video_data)
        self.point_num = point_num
        self.frame_num = frame_num
        self.is_eval = is_eval
        im_mean = (124, 116, 104)
        self.flip = transforms.RandomHorizontalFlip()
        self.img_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (224, 224),
                    scale=(0.36, 1.00),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.RandomAffine(
                    degrees=15,
                    shear=10,
                    interpolation=InterpolationMode.BICUBIC,
                    fill=im_mean,
                ),
                transforms.ColorJitter(0.1, 0.03, 0.03, 0),
                transforms.RandomGrayscale(0.05),
                transforms.ColorJitter(0.01, 0.01, 0.01, 0),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.anno_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (224, 224),
                    scale=(0.36, 1.00),
                    interpolation=InterpolationMode.NEAREST,
                ),
                transforms.RandomAffine(
                    degrees=15,
                    shear=10,
                    interpolation=InterpolationMode.NEAREST,
                    fill=0,
                ),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_data = self.data[idx]
        total_frame_num = len(video_data)
        selected_frame_idxs = []
        if total_frame_num <= self.frame_num:
            selected_frame_idxs = list(range(total_frame_num))
        else:
            selected_frame_idxs.append(0)
            interval = (total_frame_num - 1) // (self.frame_num - 1)
            if self.is_eval:
                offset = 0
            else:
                offset_min = -(interval - 1)
                offset_max = (total_frame_num - 1) - (self.frame_num - 1) * interval
                offset = random.randint(offset_min, offset_max)
            for i in range(1, self.frame_num):
                selected_frame_idxs.append(i * interval + offset)
        img_list = []
        mask_list = []

        def reseed(seed):
            random.seed(seed)
            torch.manual_seed(seed)

        flip_seed = np.random.randint(2147483647)
        for frame_idx in selected_frame_idxs:
            raw_img, raw_mask = video_data[frame_idx]
            sequence_seed = np.random.randint(2147483647)
            reseed(flip_seed)
            img = self.flip(raw_img)
            reseed(sequence_seed)
            img = self.img_transforms(img)
            reseed(flip_seed)
            mask = self.flip(raw_mask)
            reseed(sequence_seed)
            mask = self.anno_transform(mask)
            if mask.sum() == 0:
                point = None
            else:
                point = _get_boundary_points(mask.squeeze(0), self.point_num)
            while point == None or point.shape[0] != self.point_num or len(point.shape) != 2:
                sequence_seed = np.random.randint(2147483647)
                reseed(flip_seed)
                img = self.flip(raw_img)
                reseed(sequence_seed)
                img = self.img_transforms(img)
                reseed(flip_seed)
                mask = self.flip(raw_mask)
                reseed(sequence_seed)
                mask = self.anno_transform(mask)
                if mask.sum() == 0:
                    point = None
                else:
                    point = _get_boundary_points(mask.squeeze(0), self.point_num)
            img_list.append(img)
            mask_list.append(mask.squeeze(0))
        imgs = torch.stack(img_list)
        masks = torch.stack(mask_list)

        fir_point = _get_boundary_points(masks[0], self.point_num)
        points = repeat(fir_point, "n p -> f n p", f=len(mask_list))
        return imgs, masks, points
