from dataloader import DAVIS_test, OneVideoDataset
from ModelInfer import ModelInfer
from model import GeneralExtra
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import PIL.Image as Image
import numpy as np
from preprocess_utensils import get_gray_image, get_boundary_iou
import json
from loss import order_loss, chamer_distance_loss
import random
import time
import logging


coutinue_train = False
coutinue_epoch = None
continue_interval = None
continue_inter_num = None
model_name = "bear_featupConv"
log_path = f"./log/{model_name}.log"
logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.info(f"Start training {model_name}.")

# Load the dataset
train_dataset = OneVideoDataset()
data_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
testset = DAVIS_test()
model_infer = ModelInfer(testset)

# Load the model
feat_dim = 384
model = GeneralExtra(
    nn.Sequential(
        nn.Conv2d(
            in_channels=feat_dim,
            out_channels=feat_dim,
            kernel_size=3,
            padding=1,
            stride=1,
        ),
        nn.ReLU(),
        nn.Conv2d(
            in_channels=feat_dim,
            out_channels=feat_dim,
            kernel_size=3,
            padding=1,
            stride=1,
        ),
        nn.ReLU(),
    ),
    encoder="featup",
).cuda()

# Load the optimizer
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)

dict_loss = {}
dict_iou = {}
interval_epochs = 80
inter_num = 13
epoch_index = 0

if coutinue_train:
    model.load_state_dict(torch.load(f"./model/{model_name}.pth"))
    with open(f"./log/{model_name}_loss.json", "r") as f:
        dict_loss = json.load(f)
    with open(f"./log/{model_name}_iou.json", "r") as f:
        dict_iou = json.load(f)
    epoch_index = coutinue_epoch
    interval_epochs = continue_interval
    inter_num = continue_inter_num

for interval in range(inter_num):
    for e in range(interval_epochs):
        model.train()
        mean_loss = 0
        for (
            fir_img,
            fir_bou,
            pre_img,
            cur_img,
            pre_bou,
            cur_bou,
            pre_idx,
            cur_idx,
        ) in data_loader:
            pre_idx = pre_idx.item()
            pre_bou = model_infer.get_boundary(pre_idx)
            pre_bou = pre_bou.unsqueeze(0).cuda()
            optimizer.zero_grad()
            results = model(
                fir_img.cuda(),
                fir_bou.cuda(),
                pre_img.cuda(),
                cur_img.cuda(),
                pre_bou.cuda(),
            )
            refine_num = len(results)
            loss = 0
            for i in range(refine_num):
                loss += 0.8 ** (refine_num - i - 1) * order_loss(
                    results[i], cur_bou.cuda()
                )
            loss.backward()
            optimizer.step()
            mean_loss += loss.item()
        mean_loss /= len(data_loader)
        logging.info(f"Epoch {epoch_index}: Loss {mean_loss:.4f}")
        dict_loss[epoch_index] = mean_loss
        with open(f"./log/{model_name}_loss.json", "w") as f:
            json.dump(dict_loss, f)
        # save the model
        torch.save(
            model.state_dict(),
            f"./model/{model_name}.pth",
        )
        epoch_index += 1
    model_infer.infer_model(model)
    total_iou = model_infer.get_infer_iou(0)
    dict_iou[epoch_index] = total_iou
    logging.info(f"Epoch {epoch_index}: Total IOU {total_iou:.4f}")
    with open(f"./log/{model_name}_iou.json", "w") as f:
        json.dump(dict_iou, f)
    if interval_epochs > 80:
        interval_epochs -= 10
