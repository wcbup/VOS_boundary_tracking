from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from polygon import RasLoss, SoftPolygon
import logging
import json
import os
from deform_video import DeformLightVideoPos
from deform_model import get_batch_average_bou_iou
from coco_pretrain import CocoPretrainDataset
import pathlib
from torchvision import datasets
import gc

model_name = "coco_def_light_more"
load_the_model = True

# create the log directory
log_dir = f"./log/{model_name}"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_path = f"{log_dir}/{model_name}.log"
model_best_path = f"./model/{model_name}_best.pth"
model_last_path = f"./model/{model_name}_last.pth"
adam_best_path = f"./model/{model_name}_adam_best.pth"
adam_last_path = f"./model/{model_name}_adam_last.pth"

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
train_dataset = CocoPretrainDataset(
    raw_train_dataset,
    dataset_size=20_000,
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
val_dataset = CocoPretrainDataset(
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
model = DeformLightVideoPos(offset_limit=56).cuda()

# Load the loss function
ras_loss = RasLoss().cuda()
gt_rasterizer = SoftPolygon(1, "hard_mask").cuda()

# Load the optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)

loss_dict = {}
iou_train_dict = {}
iou_val_dict = {}
best_val_iou = 0
eval_period = 1
epoch_num = 14
start_epoch = 0

if load_the_model:
    model.load_state_dict(torch.load(model_best_path))
    logging.info(f"Load the model from {model_best_path}")
    # Load the loss and iou
    with open(f"{log_dir}/loss.json", "r") as f:
        loss_dict = json.load(f)
    with open(f"{log_dir}/iou_train.json", "r") as f:
        iou_train_dict = json.load(f)
    with open(f"{log_dir}/iou_val.json", "r") as f:
        iou_val_dict = json.load(f)
    best_val_iou = max(iou_val_dict.values())
    start_epoch = int(max(loss_dict.keys())) + 1
    logging.info(f"Start from epoch {start_epoch}")
    optimizer.load_state_dict(torch.load(adam_best_path))


# Start training
logging.info(f"Start training {model_name}.")
for epoch in range(start_epoch, epoch_num):
    mean_loss = 0
    train_mean_iou = 0
    model.train()
    for batch in train_data_loader:
        optimizer.zero_grad()
        fir_img, fir_mask, fir_bou = (
            batch["first_img"].cuda(),
            batch["first_mask"].cuda(),
            batch["first_boundary"].cuda(),
        )
        pre_img, pre_mask, pre_bou = (
            batch["prev_img"].cuda(),
            batch["prev_mask"].cuda(),
            batch["prev_boundary"].cuda(),
        )
        cur_img, cur_mask, cur_bou = (
            batch["curr_img"].cuda(),
            batch["curr_mask"].cuda(),
            batch["curr_boundary"].cuda(),
        )
        pred_bou = model(
            fir_img,
            fir_bou,
            pre_img,
            pre_bou,
            pre_mask,
            cur_img,
        )
        loss = ras_loss(pred_bou, cur_mask)
        loss.backward()
        optimizer.step()
        mean_loss += loss.item()
        iou = get_batch_average_bou_iou(pred_bou, cur_mask, gt_rasterizer)
        train_mean_iou += iou.item()

    mean_loss /= len(train_data_loader)
    train_mean_iou /= len(train_data_loader)
    loss_dict[epoch] = mean_loss
    iou_train_dict[epoch] = train_mean_iou
    logging.info(
        f"Epoch {epoch}, train loss: {mean_loss:.4f}, train iou: {train_mean_iou:.4f}"
    )

    if epoch % eval_period == 0 or epoch == epoch_num - 1:
        val_mean_iou = 0
        model.eval()
        for batch in val_data_loader:
            fir_img, fir_mask, fir_bou = (
                batch["first_img"].cuda(),
                batch["first_mask"].cuda(),
                batch["first_boundary"].cuda(),
            )
            pre_img, pre_mask, pre_bou = (
                batch["prev_img"].cuda(),
                batch["prev_mask"].cuda(),
                batch["prev_boundary"].cuda(),
            )
            cur_img, cur_mask, cur_bou = (
                batch["curr_img"].cuda(),
                batch["curr_mask"].cuda(),
                batch["curr_boundary"].cuda(),
            )
            pred_bou = model(
                fir_img,
                fir_bou,
                pre_img,
                pre_bou,
                pre_mask,
                cur_img,
            )
            iou = get_batch_average_bou_iou(pred_bou, cur_mask, gt_rasterizer)
            val_mean_iou += iou.item()
        val_mean_iou /= len(val_data_loader)
        iou_val_dict[epoch] = val_mean_iou
        logging.info(f"Epoch {epoch}, val iou: {val_mean_iou:.4f}")

        if val_mean_iou > best_val_iou:
            best_val_iou = val_mean_iou
            torch.save(model.state_dict(), model_best_path)
            torch.save(optimizer.state_dict(), adam_best_path)
            logging.info(f"Save the best model to {model_best_path}")
        
    torch.save(model.state_dict(), model_last_path)
    torch.save(optimizer.state_dict(), adam_last_path)
    logging.info(f"Save the last model to {model_last_path}")
        
    # Save the loss and iou
    with open(f"{log_dir}/loss.json", "w") as f:
        json.dump(loss_dict, f)
    with open(f"{log_dir}/iou_train.json", "w") as f:
        json.dump(iou_train_dict, f)
    with open(f"{log_dir}/iou_val.json", "w") as f:
        json.dump(iou_val_dict, f)

logging.info(f"Finish training {model_name}.")