from dataloader import CircleDataset, CircleRaw
from model import FeatupExtra
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from polygon import RasLoss
from ModelInfer import ModelInfer
import logging
import json

model_name = "ras_featConv"

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
model_infer = ModelInfer(raw_set)

# load the model
extra_encoder = nn.Sequential(
    nn.Conv2d(
        in_channels=384,
        out_channels=384,
        kernel_size=3,
        padding=1,
        stride=1,
    ),
    nn.ReLU(),
    nn.Conv2d(
        in_channels=384,
        out_channels=384,
        kernel_size=3,
        padding=1,
        stride=1,
    ),
    nn.ReLU(),
)
model = FeatupExtra(extra_encoder).cuda()

# Load the loss function
ras_loss = RasLoss().cuda()

# Load the optimizer
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)

dict_loss = {}
dict_iou = {}
interval_epochs = 50
inter_num = 47
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
            pre_bou = model_infer.get_boundary(pre_idx)
            pre_bou = pre_bou.unsqueeze(0).cuda()
            optimizer.zero_grad()
            results = model(
                fir_img.cuda(),
                fir_bou.cuda(),
                pre_img.cuda(),
                cur_img.cuda(),
                pre_bou.cuda(),
            )
            refine_num = len(results)
            loss = 0
            for i in range(refine_num):
                loss += 0.8 ** (refine_num - i - 1) * ras_loss(
                    results[i], cur_sgm.cuda()
                )
            loss.backward()
            optimizer.step()
            mean_loss += loss.item()
        mean_loss /= len(data_loader)
        logging.info(f"Epoch {epoch_index} loss: {mean_loss:.4f}")
        dict_loss[epoch_index] = mean_loss
        with open(f"./log/{model_name}_loss.json", "w") as f:
            json.dump(dict_loss, f)
        # save the model
        torch.save(
            model.state_dict(),
            f"./model/{model_name}.pth",
        )
        epoch_index += 1
    model_infer.infer_model(model)
    total_iou = model_infer.get_infer_iou(0)
    dict_iou[epoch_index] = total_iou.item()
    logging.info(f"Epoch {epoch_index} iou: {total_iou:.4f}")
    with open(f"./log/{model_name}_iou.json", "w") as f:
        json.dump(dict_iou, f)
    if interval_epochs > 20:
        interval_epochs -= 10
