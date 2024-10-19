from deform_model import DeformLearnImage, get_batch_average_bou_iou
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from polygon import RasLoss, SoftPolygon
from Loader_17 import DAVIS_Rawset
from image_dataset import DAVIS_IMG_Dataset
import logging
import json
import os

model_name = "deform_img_davis"

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
train_dataset = DAVIS_IMG_Dataset(train_raw_set, is_train=True)
train_data_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=4,
)
logging.info(f"Train dataset length: {len(train_dataset)}")

val_raw_set = DAVIS_Rawset(is_train=False)
val_dataset = DAVIS_IMG_Dataset(val_raw_set, is_train=False)
val_data_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=4,
)
logging.info(f"Val dataset length: {len(val_dataset)}")

# load the model
model = DeformLearnImage().cuda()

# Load the loss function
ras_loss = RasLoss().cuda()
gt_rasterizer = SoftPolygon(1, "hard_mask").cuda()

# Load the optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)

loss_dict = {}
iou_train_dict = {}
iou_val_dict = {}
best_val_iou = 0
eval_period = 10
epoch_num = 1500

logging.info(f"Start training {model_name}.")
for epoch in range(epoch_num):
    mean_loss = 0
    train_mean_iou = 0
    model.train()
    for img, mask in train_data_loader:
        img = img.cuda()
        mask = mask.cuda()
        optimizer.zero_grad()
        loss = 0
        results = model(img, mask)
        for result in results:
            loss += ras_loss(result, mask)
        loss /= len(results)
        loss.backward()
        optimizer.step()
        mean_loss += loss.item()
        iou = get_batch_average_bou_iou(
            results[-1],
            mask,
            gt_rasterizer,
        )
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
        for img, mask in val_data_loader:
            img = img.cuda()
            mask = mask.cuda()
            result = model(img, mask)
            iou = get_batch_average_bou_iou(
                result,
                mask,
                gt_rasterizer,
            )
            val_mean_iou += iou.item()
        val_mean_iou /= len(val_data_loader)
        iou_val_dict[epoch] = val_mean_iou
        logging.info(f"Epoch {epoch}, val iou: {val_mean_iou:.4f}")
        if val_mean_iou > best_val_iou:
            best_val_iou = val_mean_iou
            torch.save(model.state_dict(), model_path)
            logging.info(f"Save the best model at epoch {epoch}")

    with open(f"{log_dir}/loss.json", "w") as f:
        json.dump(loss_dict, f)
    with open(f"{log_dir}/iou_train.json", "w") as f:
        json.dump(iou_train_dict, f)
    with open(f"{log_dir}/iou_val.json", "w") as f:
        json.dump(iou_val_dict, f)
logging.info(f"Finish training {model_name}.")
