from dataloader import CircleDataset, CircleRaw
from DETR_model import DinoDETR
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from polygon import RasLoss, SoftPolygon
from ModelInfer import ModelInfer
import logging
import json

model_name = "DETR_circle"

log_path = f"./log/{model_name}.log"
logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.info(f"Start training {model_name}.")

# Load the dataset
train_dataset = CircleDataset()
data_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=4,
)
raw_set = CircleRaw()
model_infer = ModelInfer(raw_set, is_detr=True)

# load the model
model = DinoDETR().cuda()

# Load the loss function
ras_loss = RasLoss().cuda()

# Load the optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)

loss_dict = {}
iou_dict = {}
interval_epochs = 150
inter_num = 42
epoch_index = 0

for interval in range(inter_num):
    for e in range(interval_epochs):
        model.train()
        mean_loss = 0
        for (
            pre_idx,
            fir_img,
            fir_sgm,
            fir_bou,
            pre_img,
            pre_sgm,
            pre_bou,
            cur_img,
            cur_sgm,
            cur_bou,
        ) in data_loader:
            pre_idx = pre_idx.item()
            pre_sgm = model_infer.get_segement(pre_idx)
            pre_sgm = pre_sgm.unsqueeze(0)
            optimizer.zero_grad()
            pred_bou = model(
                fir_img.cuda(),
                fir_sgm.cuda(),
                pre_img.cuda(),
                pre_sgm.cuda(),
                cur_img.cuda(),
            )
            loss = ras_loss(pred_bou, cur_sgm.cuda())
            loss.backward()
            optimizer.step()
            mean_loss += loss.item()
        mean_loss /= len(data_loader)
        loss_dict[epoch_index] = mean_loss
        with open(f"./log/{model_name}_loss.json", "w") as f:
            json.dump(loss_dict, f)
        # save the model
        torch.save(
            model.state_dict(),
            f"./model/{model_name}.pth",
        )
        logging.info(f"Epoch {epoch_index} loss: {mean_loss:.4f}")
        epoch_index += 1
    model_infer.infer_model(model)
    total_iou = model_infer.get_infer_iou(0)
    iou_dict[epoch_index] = total_iou.item()
    logging.info(f"Epoch {epoch_index} iou: {total_iou:.4f}")
    with open(f"./log/{model_name}_iou.json", "w") as f:
        json.dump(iou_dict, f)
    model_infer.show_infer_result(0)
    if interval_epochs > 50:
        interval_epochs = 50
    if interval_epochs > 20:
        interval_epochs -= 10
