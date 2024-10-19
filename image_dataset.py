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


class DAVIS_IMG_Dataset(Dataset):
    def __init__(self, rawset: DAVIS_Rawset, is_train: bool, val_sample_num: int = 4):
        self.is_train = is_train
        if not is_train:
            self.val_sample_num = val_sample_num
        # remove the empty frame
        empty_frame_idx = []
        for video_idx, video_data in enumerate(rawset.data_set):
            for frame_idx, frame_data in enumerate(video_data):
                img, mask = frame_data
                if mask.sum() == 0:
                    empty_frame_idx.append((video_idx, frame_idx))
        self.data_set = []
        # add the data without empty frame
        for video_idx, video_data in enumerate(rawset.data_set):
            self.data_set.append([])
            for frame_idx, frame_data in enumerate(video_data):
                if (video_idx, frame_idx) in empty_frame_idx:
                    continue
                img, mask = frame_data
                self.data_set[-1].append((img, mask))

    def __len__(self):
        if self.is_train:
            return len(self.data_set)
        else:
            return len(self.data_set) * self.val_sample_num

    def __getitem__(self, idx: int):
        if self.is_train:
            video_idx = idx
            video_data = self.data_set[video_idx]
            # random select one frame
            frame_idx = random.randint(0, len(video_data) - 1)
            img, mask = video_data[frame_idx]
            return img, mask
        else:
            # get the video index and frame index
            video_idx = idx // self.val_sample_num
            video_data = self.data_set[video_idx]
            video_data_len = len(video_data)
            frame_step = video_data_len // self.val_sample_num
            frame_idx = (idx % self.val_sample_num) * frame_step
            img, mask = video_data[frame_idx]
            return img, mask
