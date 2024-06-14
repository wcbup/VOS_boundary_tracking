from DETR_model import DinoDetrMaskMul
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from polygon import RasLoss, SoftPolygon
from Loader_17 import DAVIS_Rawset, DAVIS_Infer, DAVIS_Dataset
import logging
import json
from MyLoss import deviation_loss, get_edges
from tenLoader import normalize

model_name = "DETR_davis_mb_std"
log_path = f"./log/{model_name}.log"
logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.info(f"Start training {model_name}")

# load the dataset
train_rawset = DAVIS_Rawset(is_train=True)
train_dataset = DAVIS_Dataset(train_rawset)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
train_infer = DAVIS_Infer(train_rawset)
val_rawset = DAVIS_Rawset(is_train=False)
val_infer = DAVIS_Infer(val_rawset)

# load the model
model = DinoDetrMaskMul(mul_before=True).cuda()

# load the loss function
ras_loss = RasLoss().cuda()

# load the optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)

train_total_loss_dict = {}
train_dif_loss_dict = {}
train_std_loss_dict = {}
train_iou_dict = {}
val_iou_dict = {}
best_val_iou = 0
interval_epochs = 15
interval_steps = 40
epoch_index = 0

for interval in range(interval_steps):
    for e in range(interval_epochs):
        model.train()
        mean_total_loss = 0
        mean_dif_loss = 0
        mean_std_loss = 0
        for (
            video_idx,
            frame_idx,
            fir_img,
            fir_sgm,
            pre_img,
            pre_sgm,
            cur_img,
            cur_sgm,
        ) in train_loader:
            optimizer.zero_grad()
            fir_img = fir_img.cuda()
            fir_sgm = fir_sgm.cuda()
            pre_img = pre_img.cuda()
            pre_sgm = pre_sgm.cuda()
            cur_img = cur_img.cuda()
            cur_sgm = cur_sgm.cuda()
            pred_bou = model(fir_img, fir_sgm, pre_img, pre_sgm, cur_img)
            dif_loss = ras_loss(pred_bou, cur_sgm)
            std_loss = deviation_loss(pred_bou)
            loss = 0.5 * dif_loss + 0.5 * std_loss
            loss.backward()
            optimizer.step()
            mean_total_loss += loss.item()
            mean_dif_loss += dif_loss.item()
            mean_std_loss += std_loss.item()
        mean_total_loss /= len(train_loader)
        mean_dif_loss /= len(train_loader)
        mean_std_loss /= len(train_loader)
        train_total_loss_dict[epoch_index] = mean_total_loss
        train_dif_loss_dict[epoch_index] = mean_dif_loss
        train_std_loss_dict[epoch_index] = mean_std_loss
        with open(f"./log/{model_name}_total_loss.json", "w") as f:
            json.dump(train_total_loss_dict, f)
        with open(f"./log/{model_name}_dif_loss.json", "w") as f:
            json.dump(train_dif_loss_dict, f)
        with open(f"./log/{model_name}_std_loss.json", "w") as f:
            json.dump(train_std_loss_dict, f)
        # save the model
        torch.save(
            model.state_dict(),
            f"./model/{model_name}.pth",
        )
        logging.info(f"Epoch {epoch_index}, Total Loss: {mean_total_loss:.4f}, Dif Loss: {mean_dif_loss:.4f}, Std Loss: {mean_std_loss:.4f}")
        epoch_index += 1
    train_infer.infer_model(model)
    train_iou = train_infer.get_total_iou()
    train_iou_dict[epoch_index] = train_iou
    val_infer.infer_model(model)
    val_iou = val_infer.get_total_iou()
    val_iou_dict[epoch_index] = val_iou
    if val_iou > best_val_iou:
        best_val_iou = val_iou
        torch.save(
            model.state_dict(),
            f"./model/{model_name}_best.pth",
        )
    logging.info(f"Epoch {epoch_index}, Train IoU: {train_iou}, Val IoU: {val_iou}")
    with open(f"./log/{model_name}_train_iou.json", "w") as f:
        json.dump(train_iou_dict, f)
    with open(f"./log/{model_name}_val_iou.json", "w") as f:
        json.dump(val_iou_dict, f)