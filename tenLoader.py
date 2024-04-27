import glob
import json
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from preprocess_utensils import get_gray_image, get_boundary_iou
import random
from torch.utils.data import DataLoader


def normalize(x: torch.Tensor) -> torch.Tensor:
    return (x - x.min()) / (x.max() - x.min())


class TenVideoTest:
    def __init__(self, json_path="./10video/train/total_data.json") -> None:
        with open(json_path, "r") as f:
            self.data = json.load(f)
        self.video_names = list(self.data.keys())
        self.video_names.sort()

        self.data_set = []
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        for video_name in self.video_names:
            self.data_set.append([])
            for data in self.data[video_name]:
                frame_path = data[0]
                frame = Image.open(frame_path).convert("RGB")
                frame = self.transform(frame)
                boundary = data[1]
                boundary = np.array(boundary).astype(np.int32)
                boundary = torch.tensor(boundary)
                sgm = get_gray_image(frame_path)
                self.data_set[-1].append((frame, boundary, sgm))

    def get_item(self, video_idx, frame_idx):
        return self.data_set[video_idx][frame_idx]

    def show_video(self, video_idx, show_sgm=False):
        plt.figure(figsize=(8, 20))
        for idx, data in enumerate(self.data_set[video_idx]):
            frame = data[0]
            bou = data[1]
            plt.subplot(10, 4, idx + 1)
            plt.imshow(normalize(frame.permute(1, 2, 0)))
            plt.plot(bou[:, 0].cpu(), bou[:, 1].cpu(), "r")
            plt.axis("off")
            plt.title(f"frame {idx}")
        if show_sgm:
            for idx, data in enumerate(self.data_set[video_idx]):
                bou = data[1]
                sgm = data[2]
                plt.subplot(10, 4, idx + 1 + 20)
                plt.imshow(normalize(sgm))
                plt.plot(bou[:, 0].cpu(), bou[:, 1].cpu(), "r")
                plt.axis("off")
                plt.title(f"frame {idx}")


class TenVideoDataset(Dataset):
    def __init__(self, json_path="./10video/train/total_data.json") -> None:
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        with open(json_path, "r") as f:
            self.raw_data = json.load(f)
        self.video_names = list(self.raw_data.keys())
        self.video_names.sort()
        self.data = []
        for name, video in self.raw_data.items():
            name_idx = self.video_names.index(name)
            for i in range(len(video) - 1):
                self.data.append((name_idx, i, video[0], video[i], video[i + 1]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        name_idx, pre_frame_idx, first_frame, pre_frame, cur_frame = self.data[idx]

        first_img = Image.open(first_frame[0]).convert("RGB")
        pre_img = Image.open(pre_frame[0]).convert("RGB")
        cur_img = Image.open(cur_frame[0]).convert("RGB")
        first_img = self.transform(first_img)
        pre_img = self.transform(pre_img)
        cur_img = self.transform(cur_img)

        first_bou = np.array(first_frame[1]).astype(np.int32)
        pre_bou = np.array(pre_frame[1]).astype(np.int32)
        cur_bou = np.array(cur_frame[1]).astype(np.int32)
        first_bou = torch.tensor(first_bou)
        pre_bou = torch.tensor(pre_bou)
        cur_bou = torch.tensor(cur_bou)

        return (
            name_idx,
            pre_frame_idx,
            first_img,
            pre_img,
            cur_img,
            first_bou,
            pre_bou,
            cur_bou,
        )


class TenVideoInfer:
    def __init__(self, test_set: TenVideoTest, device="cuda") -> None:
        self.test_set = test_set
        self.device = device
        self.infer_results = {}

    def infer_at_video(
        self,
        model: torch.nn.Module,
        video_idx: int,
        infer_num: int = 20,
    ):
        def infer_whole_at_index(
            model: torch.nn.Module,
            name_idx: int,
            frame_idx: int,
            device="cuda",
        ):
            def infer_at_index(
                model: torch.nn.Module,
                name_idx: int,
                start_idx: int,
                end_idx: int,
                device="cuda",
            ):
                infer_results = {}
                model.to(device)
                model.eval()
                if end_idx < start_idx:
                    step = -1
                else:
                    step = 1
                fir_img, fir_bou, fir_sgm = self.test_set.get_item(name_idx, start_idx)
                infer_results[start_idx] = fir_bou
                # plt.subplot(10, 4, start_idx + 1)
                # plt.imshow(normalize(fir_img.permute(1, 2, 0)))
                # plt.plot(fir_bou[:, 0], fir_bou[:, 1], "r")
                # plt.axis("off")
                # plt.title(f"frame {start_idx}")
                start_idx += step
                pre_bou = fir_bou
                pre_img = fir_img
                for i in range(start_idx, end_idx + step, step):
                    cur_img, cur_bou, cur_sgm = self.test_set.get_item(name_idx, i)
                    with torch.no_grad():
                        results = model(
                            fir_img.unsqueeze(0).to(device),
                            fir_bou.unsqueeze(0).to(device),
                            pre_img.unsqueeze(0).to(device),
                            cur_img.unsqueeze(0).to(device),
                            pre_bou.unsqueeze(0).to(device),
                        )
                    pre_bou = results[-1].int().squeeze(0).clamp(0, 223)
                    infer_results[i] = pre_bou
                    pre_img = cur_img
                    # plt.subplot(10, 4, i + 1)
                    # plt.imshow(normalize(cur_img.permute(1, 2, 0)))
                    # plt.plot(
                    #     pre_bou[:, 0].cpu().numpy(),
                    #     pre_bou[:, 1].cpu().numpy(),
                    # )
                    # plt.axis("off")
                    # plt.title(f"frame {i}")
                return infer_results

            result1 = infer_at_index(model, name_idx, frame_idx, 0, device)
            result2 = infer_at_index(model, name_idx, frame_idx, 19, device)
            return {**result1, **result2}

        infer_results = []
        video_len = len(self.test_set.data_set[video_idx])
        if infer_num >= len(self.test_set.data_set[video_idx]):
            infer_range = range(video_len)
        elif infer_num <= 0:
            infer_range = [0]
        else:
            interval = video_len // infer_num
            random_start = random.randint(0, interval - 1)
            infer_range = range(random_start, video_len, interval)
        for i in infer_range:
            infer_results.append(infer_whole_at_index(model, video_idx, i))
        return infer_results

    def infer_model(self, model: torch.nn.Module, infer_num: int = 20):

        for name_idx in range(len(self.test_set.video_names)):
            self.infer_results[name_idx] = self.infer_at_video(
                model,
                name_idx,
                infer_num,
            )

    def get_iou(self, video_idx, frame_idx):
        infer_bou = self.infer_results[video_idx][0][frame_idx]
        _, _, sgm = self.test_set.get_item(video_idx, frame_idx)
        iou = get_boundary_iou(sgm, infer_bou.cpu().numpy())
        return iou

    def get_video_iou(self, video_idx):
        total_iou = 0
        for frame_idx in range(20):
            total_iou += self.get_iou(video_idx, frame_idx)
        return total_iou / 20

    def get_total_iou(self):
        total_iou = 0
        for video_idx in range(len(self.test_set.video_names)):
            total_iou += self.get_video_iou(video_idx)
        return total_iou / len(self.test_set.video_names)

    def show_infer_result(self, video_idx, start_frame_idx):
        plt.figure(figsize=(15, 20))
        for frame_idx, boundary in self.infer_results[video_idx][
            start_frame_idx
        ].items():
            plt.subplot(5, 4, frame_idx + 1)
            img, _, _ = self.test_set.get_item(video_idx, frame_idx)
            plt.imshow(normalize(img.permute(1, 2, 0)))
            plt.plot(boundary[:, 0].cpu(), boundary[:, 1].cpu(), "r")
            plt.axis("off")
            plt.title(f"frame {frame_idx}")

    def get_boundary(self, video_idx, frame_idx):
        if len(self.infer_results) == 0:
            return self.test_set.get_item(video_idx, frame_idx)[1]
        else:
            random_index = random.randint(0, len(self.infer_results[video_idx]) - 1)
            return self.infer_results[video_idx][random_index][frame_idx]
