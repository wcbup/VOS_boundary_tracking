from tenLoader import TenVideoDataset, normalize, TenVideoTest, TenVideoInfer
from model import FeatupExtra
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
from polygon import RasLoss


model_name = "featConv_10"
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
test_testset = TenVideoTest("./10video/test/total_data.json")
test_model_infer = TenVideoInfer(test_testset)

# Load the model
model = FeatupExtra(
    nn.Sequential(
        nn.Conv2d(
            in_channels=384,
            out_channels=384,
            kernel_size=3,
            padding=1,
            stride=1,
        ),
        nn.ReLU(),
        nn.Conv2d(
            in_channels=384,
            out_channels=384,
            kernel_size=3,
            padding=1,
            stride=1,
        ),
        nn.ReLU(),
    )
).cuda()

# Load the optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# load the loss function
ras_loss = RasLoss().cuda()

dict_loss = {}
dict_iou = {}
test_dict_iou = {}
best_test_iou = 0
interval_epochs = 50
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
            cur_sgm = model_infer.test_set.get_item(video_idx, pre_idx + 1)[2]
            cur_sgm[cur_sgm > 0] = 1
            cur_sgm = torch.Tensor(cur_sgm).unsqueeze(0)
            for i in range(refine_num):
                loss += 0.8 ** (refine_num - i - 1) * ras_loss(
                    results[i], cur_sgm.cuda()
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
    test_model_infer.infer_model(model, 0)
    test_total_iou = test_model_infer.get_total_iou()
    if test_total_iou > best_test_iou:
        best_test_iou = test_total_iou
        torch.save(
            model.state_dict(),
            f"./model/{model_name}_best.pth",
        )
    total_iou = model_infer.get_total_iou()
    dict_iou[epoch_index] = total_iou
    test_dict_iou[epoch_index] = test_total_iou
    logging.info(
        f"Epoch {epoch_index}: Train Total IOU {total_iou:.4f}, Test Total IOU {test_total_iou:.4f}"
    )
    with open(f"./log/{model_name}_iou.json", "w") as f:
        json.dump(dict_iou, f)
    with open(f"./log/{model_name}_test_iou.json", "w") as f:
        json.dump(test_dict_iou, f)
    if interval_epochs > 20:
        interval_epochs -= 10
