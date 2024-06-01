from DETR_model import DinoDETR
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from polygon import RasLoss, SoftPolygon
from tenLoader import TenRawset, TenVideoInfer, TenDataset, OneHundredRawset
import logging
import json

model_name = "DETR_100"
torch.manual_seed(0)

log_path = f"./log/{model_name}.log"
logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.info(f"Start training {model_name}")

# Load the dataset
raw_set = OneHundredRawset()
train_dataset = TenDataset(raw_set)
data_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=4,
)
model_infer = TenVideoInfer(raw_set, is_detr=True)
test_infer = TenVideoInfer(TenRawset(False), is_detr=True)

# load the model
model = DinoDETR().cuda()

# Load the loss function
ras_loss = RasLoss().cuda()

# Load the optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)

loss_dict = {}
iou_dict = {}
test_iou_dict = {}
best_test_iou = 0
interval_epochs = 20
inter_num = 25
epoch_index = 0

for interval in range(inter_num):
    for e in range(interval_epochs):
        model.train()
        mean_loss = 0
        for (
            video_idx,
            pre_idx,
            fir_img,
            fir_bou,
            fir_sgm,
            pre_img,
            pre_bou,
            pre_sgm,
            cur_img,
            cur_bou,
            cur_sgm,
        ) in data_loader:
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
    model_infer.infer_model(model, 0)
    total_iou = model_infer.get_total_iou()
    iou_dict[epoch_index] = total_iou.item()
    test_infer.infer_model(model, 0)
    test_iou = test_infer.get_total_iou()
    test_iou_dict[epoch_index] = test_iou.item()
    if test_iou > best_test_iou:
        best_test_iou = test_iou
        torch.save(
            model.state_dict(),
            f"./model/{model_name}_best.pth",
        )
    logging.info(
        f"Epoch {epoch_index}: train iou: {total_iou:.4f}, test iou: {test_iou:.4f}"
    )
    with open(f"./log/{model_name}_iou.json", "w") as f:
        json.dump(iou_dict, f)
    with open(f"./log/{model_name}_test_iou.json", "w") as f:
        json.dump(test_iou_dict, f)
    # if interval_epochs > 50:
    #     interval_epochs = 50
    # if interval_epochs > 20:
    #     interval_epochs -= 10
