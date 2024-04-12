from tenLoader import TenVideoDataset, normalize, TenVideoTest, TenVideoInfer
from model import IterWholeFirst
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


model_name = "unet_10"
log_path = f"./log/{model_name}.log"
logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.info(f"Start training {model_name}.")

# Load the dataset
train_dataset = TenVideoDataset()
data_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
testset = TenVideoTest()
model_infer = TenVideoInfer(testset)

# Load the model
model = IterWholeFirst(
    add_xy_in_token=True,
    use_unet=True,
).cuda()

# Load the optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)

dict_loss = {}
dict_iou = {}
interval_epochs = 500
inter_num = 47
epoch_index = 0

for interval in range(inter_num):
    for e in range(interval_epochs):
        model.train()
        mean_loss = 0
        for (
            video_idx,
            pre_idx,
            fir_img,
            pre_img,
            cur_img,
            fir_bou,
            pre_bou,
            cur_bou,
        ) in data_loader:
            pre_idx = pre_idx.item()
            video_idx = video_idx.item()
            pre_bou = model_infer.get_boundary(video_idx, pre_idx)
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
    total_iou = model_infer.get_total_iou()
    dict_iou[epoch_index] = total_iou
    logging.info(f"Epoch {epoch_index}: Total IOU {total_iou:.4f}")
    with open(f"./log/{model_name}_iou.json", "w") as f:
        json.dump(dict_iou, f)
    if interval_epochs > 50:
        interval_epochs = 50
    if interval_epochs > 20:
        interval_epochs -= 10
