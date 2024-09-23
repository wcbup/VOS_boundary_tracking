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