import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from polygon import SoftPolygon, RasLoss
from cotracker import Cotracker, VideoLoss
import os
import logging
import json
from torchvision import datasets
from coco_pretrain import CocoVideoDataset
import pathlib
import torch.optim as optim
from einops import repeat

model_name = "coco_cotracker"

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
coco_train_img_path = pathlib.Path("./coco/train2017")
coco_train_ann_path = pathlib.Path("./coco/annotations/instances_train2017.json")
raw_train_dataset = datasets.CocoDetection(coco_train_img_path, coco_train_ann_path)
train_dataset = CocoVideoDataset(
    raw_train_dataset,
    dataset_size=10_000,
)
train_data_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0,
)
logging.info(f"Train dataset length: {len(train_dataset)}")

coco_val_img_path = pathlib.Path("./coco/val2017")
coco_val_ann_path = pathlib.Path("./coco/annotations/instances_val2017.json")
raw_val_dataset = datasets.CocoDetection(coco_val_img_path, coco_val_ann_path)
val_dataset = CocoVideoDataset(
    raw_val_dataset,
    dataset_size=None,
)
val_data_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0,
)
logging.info(f"Val dataset length: {len(val_dataset)}")

# load the model
model = Cotracker(32).cuda()

# Load the loss function
ras_loss = RasLoss().cuda()
gt_rasterizer = SoftPolygon(1, "hard_mask").cuda()
video_loss = VideoLoss(ras_loss, gt_rasterizer).cuda()

# Load the optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)

loss_dict = {}
iou_train_dict = {}
iou_val_dict = {}
best_val_iou = 0
epoch_num = 8

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
        init_points = repeat(fir_points, 'b h w -> b f h w', f=frame_num)
        pred_points = model(imgs, init_points)
        target_masks = masks[:, 1:]
        loss, iou = video_loss(pred_points, target_masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_iou += iou.item()
    loss_dict[e] = total_loss / len(train_data_loader)
    iou_train_dict[e] = total_iou / len(train_data_loader)
    logging.info(f"Epoch {e}, Loss: {loss_dict[e]:.4f}, Train IoU: {iou_train_dict[e]:.4f}")
    # Eval the model
    if e % 1 == 0 or e == epoch_num - 1:
        total_iou = 0
        model.eval()
        with torch.no_grad():
            for imgs, masks, points in val_data_loader:
                imgs = imgs.cuda()
                masks = masks.cuda()
                points = points.cuda()
                fir_points = points[:, 0]
                frame_num = masks.shape[1]
                init_points = repeat(fir_points, 'b h w -> b f h w', f=frame_num)
                pred_points = model(imgs, init_points)
                target_masks = masks[:, 1:]
                _, iou = video_loss(pred_points, target_masks)
                total_iou += iou.item()
        iou_val_dict[e] = total_iou / len(val_data_loader)
        logging.info(f"Epoch {e}, Val IoU: {iou_val_dict[e]:.4f}")
        if iou_val_dict[e] > best_val_iou:
            best_val_iou = iou_val_dict[e]
            torch.save(model.state_dict(), model_best_path)
            logging.info(f"New Best Model saved at epoch {e}")
        
    # save the loss and iou
    with open(f"{log_dir}/loss.json", "w") as f:
        json.dump(loss_dict, f)
    with open(f"{log_dir}/iou_train.json", "w") as f:
        json.dump(iou_train_dict, f)
    with open(f"{log_dir}/iou_val.json", "w") as f:
        json.dump(iou_val_dict, f)
    torch.save(model.state_dict(), model_last_path)
    torch.save(optimizer.state_dict(), optimizer_path)
    logging.info(f"Model saved at epoch {e}")
logging.info(f"Finish training {model_name}.")