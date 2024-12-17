import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from polygon import SoftPolygon, RasLoss
from cotracker import CotrackerLight, VideoLoss
import os
import logging
import json
from torchvision import datasets
from coco_pretrain import (
    StaticVideoDataset,
    ECSSD_dataset,
    FSS1000_dataset,
    HRSOD_dataset,
    BIG_dataset,
    DUTS_dataset,
)
import pathlib
import torch.optim as optim
from einops import repeat

model_name = "static_cotracker_light"
frame_num = 6
point_num = 16

# create the log directory
log_dir = f"./log/{model_name}"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_path = f"{log_dir}/{model_name}.log"
model_best_path = f"./model/{model_name}_best.pth"
model_last_path = f"./model/{model_name}_last.pth"
optimizer_path = f"./model/{model_name}_optimizer.pth"

logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.info(f"Preparing training {model_name}.")

# Load the dataset
ecssd_dataset = ECSSD_dataset()
fss_dataset = FSS1000_dataset()
hrsod_dataset = HRSOD_dataset()
big_dataset = BIG_dataset()
duts_dataset = DUTS_dataset()
dataset_list = []
dataset_list.append(ecssd_dataset)
dataset_list.append(fss_dataset)
for i in range(5):
    dataset_list.append(hrsod_dataset)
for i in range(5):
    dataset_list.append(big_dataset)
dataset_list.append(duts_dataset)
train_dataset = StaticVideoDataset(
    dataset_list,
    frame_num,
    point_num,
)
train_data_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=8,
)
logging.info(f"Train dataset length: {len(train_dataset)}")



# load the model
model = CotrackerLight(point_num).cuda()

# Load the loss function
ras_loss = RasLoss().cuda()
gt_rasterizer = SoftPolygon(1, "hard_mask").cuda()
video_loss = VideoLoss(ras_loss, gt_rasterizer).cuda()

# Load the optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)

loss_dict = {}
iou_train_dict = {}
best_iou = 0
epoch_num = 5400

logging.info(f"Start training {model_name}.")
for e in range(epoch_num):
    total_loss = 0
    total_iou = 0
    model.train()
    for imgs, masks, points in train_data_loader:
        optimizer.zero_grad()
        imgs = imgs.cuda()
        masks = masks.cuda()
        points = points.cuda()
        fir_points = points[:, 0]
        frame_num = masks.shape[1]
        init_points = repeat(fir_points, "b h w -> b f h w", f=frame_num)
        pred_points = model(imgs, init_points)
        target_masks = masks[:, 1:]
        loss, iou = video_loss(pred_points, target_masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_iou += iou.item()
    loss_dict[e] = total_loss / len(train_data_loader)
    iou_train_dict[e] = total_iou / len(train_data_loader)
    logging.info(
        f"Epoch {e}, Loss: {loss_dict[e]:.4f}, Train IoU: {iou_train_dict[e]:.4f}"
    )
    train_iou = iou_train_dict[e]
    if train_iou > best_iou:
        best_iou = train_iou
        torch.save(model.state_dict(), model_best_path)
        logging.info(f"Model saved at epoch {e}")

    # save the loss and iou
    with open(f"{log_dir}/loss.json", "w") as f:
        json.dump(loss_dict, f)
    with open(f"{log_dir}/iou_train.json", "w") as f:
        json.dump(iou_train_dict, f)
    torch.save(model.state_dict(), model_last_path)
    torch.save(optimizer.state_dict(), optimizer_path)
    logging.info(f"Model saved at epoch {e}")
logging.info(f"Finish training {model_name}.")
