import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import json
import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt

def normalize_image(x: torch.Tensor) -> torch.Tensor:
    return (x - x.min()) / (x.max() - x.min())

class DAVIS_Seq2(torch.utils.data.Dataset):
    def __init__(self, is_uniform=True, is_one_video=False):
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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