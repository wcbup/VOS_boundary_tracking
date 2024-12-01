import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from polygon import SoftPolygon, RasLoss
from cotracker import Cotracker, DAVIS_Windows, CoWinEvaler, VideoLoss
import os
import logging
from Loader_17 import DAVIS_Rawset
from deform_video import DAVIS_withPoint
import json
from einops import repeat

model_name = "cotracker_win"
total_win_size = 40
small_win_size = 10
half_win_size = small_win_size // 2
sliding_win_num = (total_win_size - small_win_size) // half_win_size + 1

# create the log directory
log_dir = f"./log/{model_name}"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_path = f"{log_dir}/{model_name}.log"
model_path = f"./model/{model_name}_best.pth"

logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.info(f"Preparing training {model_name}.")

train_raw_set = DAVIS_Rawset(is_train=True)
train_point_set = DAVIS_withPoint(train_raw_set, 32, is_train=True)
train_video_set = DAVIS_Windows(train_point_set.raw_data_set, total_win_size)
train_video_loader = DataLoader(train_video_set, batch_size=1, shuffle=True)
logging.info(f"Train dataset length: {len(train_video_set)}")

val_raw_set = DAVIS_Rawset(is_train=False)
val_point_set = DAVIS_withPoint(val_raw_set, 32, is_train=False)
val_video_set = DAVIS_Windows(val_point_set.raw_data_set, total_win_size)
gt_rasterizer = SoftPolygon(1, "hard_mask").cuda()
co_win_evaler = CoWinEvaler(val_point_set.raw_data_set, gt_rasterizer, small_win_size)

# load the model
model = Cotracker(32).cuda()
optimizer = Adam(model.parameters(), lr=1e-4)

# Load the loss function
ras_loss = RasLoss().cuda()
video_loss = VideoLoss(ras_loss, gt_rasterizer).cuda()

loss_dict = {}
train_iou_dict = {}
eval_iou_dict = {}
epoch_num = 300
eval_interval = 15
best_val_iou = 0

for e in range(epoch_num):
    total_loss = 0
    total_iou = 0
    for imgs, masks, points in train_video_loader:
        fir_point = points[:, 0]
        cur_points = repeat(fir_point, "b p xy -> b f p xy", f=small_win_size)
        loss = 0
        iou = 0
        frame_num = len(imgs[0])
        sliding_win_num = (frame_num - small_win_size) // half_win_size + 1
        for i in range(0, sliding_win_num):
            cur_imgs = imgs[
                :, i * half_win_size : i * half_win_size + small_win_size
            ]
            cur_masks = masks[
                :, i * half_win_size : i * half_win_size + small_win_size
            ]
            optimizer.zero_grad()
            cur_imgs = cur_imgs.cuda()
            cur_masks = cur_masks.cuda()
            cur_points = cur_points.cuda()
            cur_pred_points = model(cur_imgs, cur_points)
            target_masks = cur_masks[:, 1:]
            tmp_loss, tmp_iou = video_loss(cur_pred_points, target_masks)
            tmp_loss.backward()
            optimizer.step()
            loss += tmp_loss.item()
            iou += tmp_iou.item()
            new_points = repeat(
                cur_pred_points[:, -1, -1], "b p xy -> b f p xy", f=half_win_size
            ).detach()
            old_points = cur_pred_points[:, -1, -half_win_size:].detach()
            cur_points = torch.cat([old_points, new_points], 1)
        if frame_num % half_win_size != 0:
            optimizer.zero_grad()
            cur_imgs = imgs[:, -small_win_size:]
            cur_imgs = cur_imgs.cuda()
            cur_masks = masks[:, -small_win_size:].cuda()
            left_len = frame_num - (sliding_win_num + 1) * half_win_size
            new_points = repeat(
                cur_pred_points[:, -1, -1], "b p xy -> b f p xy", f=left_len
            ).detach()
            old_point_num = small_win_size - left_len
            old_points = cur_pred_points[:, -1, -old_point_num:].detach()
            cur_points = torch.cat([old_points, new_points], 1)
            cur_points = cur_points.cuda()
            cur_pred_points = model(cur_imgs, cur_points)
            target_masks = cur_masks[:, 1:]
            tmp_loss, tmp_iou = video_loss(cur_pred_points, target_masks)
            tmp_loss.backward()
            optimizer.step()
            loss += tmp_loss.item()
            iou += tmp_iou.item()
        if frame_num % half_win_size == 0:
            loss /= sliding_win_num
            iou /= sliding_win_num
        else:
            loss /= sliding_win_num + 1
            iou /= sliding_win_num + 1
        total_loss += loss
        total_iou += iou
    total_loss /= len(train_video_loader)
    total_iou /= len(train_video_loader)
    logging.info(f"Epoch {e}, Loss: {total_loss:.4f}, Train IoU: {total_iou:.4f}")
    loss_dict[e] = total_loss
    train_iou_dict[e] = total_iou
    if e % eval_interval == 0 or e == epoch_num - 1:
        eval_iou = co_win_evaler.eval_all_videos(model, use_tqdm=False)
        logging.info(f"Epoch {e}, Eval IoU: {eval_iou:.4f}")
        eval_iou_dict[e] = eval_iou
        if eval_iou > best_val_iou:
            best_val_iou = eval_iou
            torch.save(model.state_dict(), model_path)
            logging.info(f"Model saved at epoch {e}")
    
    # save the loss and iou
    with open(f"{log_dir}/loss.json", "w") as f:
        json.dump(loss_dict, f)
    with open(f"{log_dir}/train_iou.json", "w") as f:
        json.dump(train_iou_dict, f)
    with open(f"{log_dir}/eval_iou.json", "w") as f:
        json.dump(eval_iou_dict, f)
    
logging.info("Training finished.")