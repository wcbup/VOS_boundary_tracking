from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from polygon import RasLoss, SoftPolygon
from Loader_17 import DAVIS_Rawset
import logging
import json
import os
from deform_video import DAVIS_withPoint, DeformVideo, VideoInferer
from deform_model import get_batch_average_bou_iou, deviation_loss

model_name = "def_davis_std"

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

# Load the dataset
train_raw_set = DAVIS_Rawset(is_train=True)
# train_raw_set.data_set = train_raw_set.data_set[:2]
train_dataset = DAVIS_withPoint(
    train_raw_set,
    is_train=True,
    point_num=64,
)
train_data_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=4,
)
logging.info(f"Train dataset length: {len(train_dataset)}")

val_raw_set = DAVIS_Rawset(is_train=False)
# val_raw_set.data_set = val_raw_set.data_set[:2]
val_dataset = DAVIS_withPoint(
    val_raw_set,
    is_train=False,
    point_num=64,
)

logging.info(f"Val dataset length: {len(val_dataset)}")

# load the model
model = DeformVideo().cuda()

# Load the loss function
ras_loss = RasLoss().cuda()
gt_rasterizer = SoftPolygon(1, "hard_mask").cuda()
val_inferer = VideoInferer(val_dataset, gt_rasterizer)

# Load the optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)

dif_loss_dict = {}
std_loss_dict = {}
iou_train_dict = {}
iou_val_dict = {}
best_val_iou = 0
eval_period = 1
epoch_num = 12

# Start training
logging.info(f"Start training {model_name}.")
for epoch in range(epoch_num):
    mean_dif_loss = 0
    mean_std_loss = 0
    train_mean_iou = 0
    model.train()
    for (
        video_idx,
        frame_idx,
        first_frame,
        previous_frame,
        current_frame,
    ) in train_data_loader:
        optimizer.zero_grad()
        fir_img, fir_mask, fir_point = first_frame
        pre_img, pre_mask, pre_point = previous_frame
        cur_img, cur_mask, cur_point = current_frame
        results = model(
            fir_img.cuda(),
            fir_point.cuda(),
            pre_img.cuda(),
            pre_point.cuda(),
            pre_mask.cuda(),
            cur_img.cuda(),
        )
        dif_loss = 0
        std_loss = 0
        for result in results:
            dif_loss += ras_loss(result, cur_mask.cuda())
            std_loss += deviation_loss(result)
        dif_loss /= len(results)
        std_loss /= len(results)
        total_loss = dif_loss + std_loss
        total_loss.backward()
        optimizer.step()
        mean_dif_loss += dif_loss.item()
        mean_std_loss += std_loss.item()
        iou = get_batch_average_bou_iou(results[-1], cur_mask.cuda(), gt_rasterizer)
        train_mean_iou += iou.item()
    mean_dif_loss /= len(train_data_loader)
    mean_std_loss /= len(train_data_loader)
    train_mean_iou /= len(train_data_loader)
    dif_loss_dict[epoch] = mean_dif_loss
    std_loss_dict[epoch] = mean_std_loss
    iou_train_dict[epoch] = train_mean_iou
    logging.info(
        f"Epoch {epoch}, dif loss: {mean_dif_loss:.4f}, std loss: {std_loss_dict:.4f}, train iou: {train_mean_iou:.4f}"
    )
    if epoch % eval_period == 0 or epoch == epoch_num - 1:
        val_mean_iou = 0
        model.eval()
        val_inferer.infer_all_videos(model)
        val_mean_iou = val_inferer.compute_all_videos_iou()
        iou_val_dict[epoch] = val_mean_iou
        logging.info(f"Epoch {epoch}, val iou: {val_mean_iou:.4f}")
        if val_mean_iou > best_val_iou:
            best_val_iou = val_mean_iou
            torch.save(model.state_dict(), model_path)
            logging.info(f"Save the best model to {model_path}")

    # Save the loss and iou
    with open(f"{log_dir}/dif_loss.json", "w") as f:
        json.dump(dif_loss_dict, f)
    with open(f"{log_dir}/std_loss.json", "w") as f:
        json.dump(std_loss_dict, f)
    with open(f"{log_dir}/iou_train.json", "w") as f:
        json.dump(iou_train_dict, f)
    with open(f"{log_dir}/iou_val.json", "w") as f:
        json.dump(iou_val_dict, f)

logging.info(f"Training {model_name} finished.")
