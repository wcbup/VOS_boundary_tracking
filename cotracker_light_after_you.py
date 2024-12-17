import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from polygon import SoftPolygon, RasLoss
from cotracker import CotrackerLight, DAVIS_Video, CoEvaler, VideoLoss
import os
import logging
from Loader_17 import DAVIS_Rawset
from deform_video import DAVIS_withPoint
import json

model_name = "cotracker_light_after_you"
pretrain_model_path = "model/cotracker_light_you_best.pth"
frame_num = 6
point_num = 16

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
train_point_set = DAVIS_withPoint(train_raw_set, point_num, is_train=True)
train_video_set = DAVIS_Video(train_point_set.raw_data_set, frame_num)
train_video_loader = DataLoader(train_video_set, batch_size=1, shuffle=True)
logging.info(f"Train dataset length: {len(train_video_set)}")

val_raw_set = DAVIS_Rawset(is_train=False)
val_point_set = DAVIS_withPoint(val_raw_set, point_num, is_train=False)
val_video_set = DAVIS_Video(val_point_set.raw_data_set, frame_num)
gt_rasterizer = SoftPolygon(1, "hard_mask").cuda()
coevaler = CoEvaler(val_point_set.raw_data_set, gt_rasterizer)
logging.info(f"Val dataset length: {len(val_video_set)}")

# load the model
model = CotrackerLight(point_num).cuda()
model.load_state_dict(torch.load(pretrain_model_path))
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
    for total_imgs, total_masks, total_points in train_video_loader:
        optimizer.zero_grad()
        pred_points = model(total_imgs.cuda(), total_points.cuda())
        target_masks = total_masks[:, 1:]
        loss, iou = video_loss(pred_points, target_masks.cuda())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_iou += iou.item()
    loss_dict[e] = total_loss / len(train_video_loader)
    train_iou_dict[e] = total_iou / len(train_video_loader)
    logging.info(
        f"Epoch {e}, Loss: {loss_dict[e]:.4f}, Train IoU: {train_iou_dict[e]:.4f}"
    )
    if e % eval_interval == 0 or e == epoch_num - 1:
        avg_iou = coevaler.eval_all_video(model).item()
        eval_iou_dict[e] = avg_iou
        logging.info(f"Epoch {e}, Eval IoU: {eval_iou_dict[e]:.4f}")
        if avg_iou > best_val_iou:
            best_val_iou = avg_iou
            torch.save(model.state_dict(), model_path)
            logging.info(f"Model saved at epoch {e}")

    # save the loss and iou
    with open(f"{log_dir}/loss.json", "w") as f:
        json.dump(loss_dict, f)
    with open(f"{log_dir}/train_iou.json", "w") as f:
        json.dump(train_iou_dict, f)
    with open(f"{log_dir}/eval_iou.json", "w") as f:
        json.dump(eval_iou_dict, f)

logging.info(f"Training {model_name} finished.")
