import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import json
import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from preprocess_utensils import get_gray_image


def normalize_image(x: torch.Tensor) -> torch.Tensor:
    return (x - x.min()) / (x.max() - x.min())


class DAVIS_Seq2(torch.utils.data.Dataset):
    def __init__(self, is_uniform=True, is_one_video=False):
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        if is_uniform:
            with open("./uniform_samples_80.json", "r") as f:
                tmp_data: dict[str, list[tuple[str, str, list]]] = json.loads(f.read())

        else:
            with open("./simplify_samples_80.json", "r") as f:
                tmp_data: dict[str, list[tuple[str, str, list]]] = json.loads(f.read())
        self.data = []
        if is_one_video:
            video = tmp_data["bear"]
            for i in range(len(video) - 1):
                self.data.append((video[i], video[i + 1]))
        else:
            for name, video in tmp_data.items():
                for i in range(len(video) - 1):
                    self.data.append((video[i], video[i + 1]))

    def __len__(self):
        return len(self.data)

    def __getitem__(
        self, idx
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        frame1, frame2 = self.data[idx]
        frame1_image = Image.open(frame1[0])
        frame2_image = Image.open(frame2[0])
        frame1_image = self.transform(frame1_image)
        frame2_image = self.transform(frame2_image)
        frame1_boundary = np.array(frame1[2]).astype(np.int32)
        frame1_boundary = torch.Tensor(frame1_boundary).int()
        frame2_boundary = np.array(frame2[2]).astype(np.int32)
        frame2_boundary = torch.Tensor(frame2_boundary).int()
        return frame1_image, frame2_image, frame1_boundary, frame2_boundary


class BallDataset(Dataset):
    def __init__(
        self,
        json_path="./ball/uniform_samples_80.json",
        is_previous=True,
        output_first=False,
        transform=None,
    ):
        self.json_path = json_path
        self.is_previous = is_previous
        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.transform = transform
        tmp_data = json.load(open(json_path, "r"))
        self.output_first = output_first
        if output_first:
            self.first_data = tmp_data[0]
        self.data = []
        if is_previous:
            self.data.append((tmp_data[1], tmp_data[0]))
            for i in range(len(tmp_data) - 1):
                self.data.append((tmp_data[i], tmp_data[i + 1]))
        else:
            for i in range(len(tmp_data) - 1):
                self.data.append((tmp_data[0], tmp_data[i + 1]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        frame1, frame2 = self.data[idx]
        frame1_img = Image.open(frame1[0]).convert("RGB")
        frame2_img = Image.open(frame2[0]).convert("RGB")
        frame1_img = self.transform(frame1_img)
        frame2_img = self.transform(frame2_img)
        frame1_boundary = np.array(frame1[1]).astype(np.int32)
        frame1_boundary = torch.tensor(frame1_boundary).int()
        frame2_boundary = np.array(frame2[1]).astype(np.int32)
        frame2_boundary = torch.tensor(frame2_boundary).int()
        if not self.is_previous:
            pre_idx = 0
            curr_idx = idx + 1
        elif idx > 0:
            pre_idx = idx - 1
            curr_idx = idx
        else:
            pre_idx = 1
            curr_idx = 0
        output = (
            frame1_img,
            frame2_img,
            frame1_boundary,
            frame2_boundary,
            pre_idx,
            curr_idx,
        )
        if self.output_first:
            first_img = Image.open(self.first_data[0]).convert("RGB")
            first_img = self.transform(first_img)
            first_boundary = np.array(self.first_data[1]).astype(np.int32)
            first_boundary = torch.tensor(first_boundary).int()
            return first_img, first_boundary, *output
        else:
            return output


class Balltest(torch.utils.data.Dataset):
    def __init__(self, json_path="./ball/uniform_samples_80.json"):
        self.json_path = json_path
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.data = json.load(open(json_path, "r"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, boundary = self.data[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        sgm = get_gray_image(img_path)
        boundary = np.array(boundary).astype(np.int32)
        boundary = torch.tensor(boundary).int()
        return img, sgm, boundary


class DAVIS_test(torch.utils.data.Dataset):
    def __init__(self, video_name: str = "bear", is_uniform=True):
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        if is_uniform:
            with open("./uniform_samples_80.json", "r") as f:
                tmp_data: dict[str, list[tuple[str, str, list]]] = json.loads(f.read())
        else:
            with open("./simplify_samples_80.json", "r") as f:
                tmp_data: dict[str, list[tuple[str, str, list]]] = json.loads(f.read())

        self.data = tmp_data[video_name]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, sgm_path, boundary = self.data[idx]
        img = Image.open(img_path)
        sgm = get_gray_image(sgm_path)
        img = self.transform(img)
        boundary = np.array(boundary).astype(np.int32)
        boundary = torch.tensor(boundary).int()
        return img, sgm, boundary


class OneVideoDataset(Dataset):
    def __init__(self, video_name="bear"):
        with open("./uniform_samples_80.json", "r") as f:
            total_data: dict[str, list[tuple[str, str, list]]] = json.loads(f.read())
        self.raw_data = total_data[video_name]
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.data = []
        for i in range(len(self.raw_data) - 1):
            self.data.append(
                (
                    self.raw_data[0],
                    self.raw_data[i],
                    self.raw_data[i + 1],
                    i,
                    i + 1,
                )
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fir_frame, pre_frame, cur_frame, pre_idx, cur_idx = self.data[idx]
        fir_img = Image.open(fir_frame[0]).convert("RGB")
        pre_img = Image.open(pre_frame[0]).convert("RGB")
        cur_img = Image.open(cur_frame[0]).convert("RGB")
        fir_img = self.transform(fir_img)
        pre_img = self.transform(pre_img)
        cur_img = self.transform(cur_img)

        fir_bou = np.array(fir_frame[2]).astype(np.int32)
        pre_bou = np.array(pre_frame[2]).astype(np.int32)
        cur_bou = np.array(cur_frame[2]).astype(np.int32)

        return (
            fir_img,
            fir_bou,
            pre_img,
            cur_img,
            pre_bou,
            cur_bou,
            pre_idx,
            cur_idx,
        )
