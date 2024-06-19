import torch
import matplotlib.pyplot as plt
from polygon import RasLoss, SoftPolygon
import json
from tenLoader import normalize
import torch.nn as nn
import torch.optim as optim
from MyLoss import get_edges, deviation_loss, total_len_loss
from preprocess_utensils import get_boundary_iou
import numpy as np
import time
import logging


class BoundarySampler(nn.Module):
    def __init__(
        self,
    ):
        super(BoundarySampler, self).__init__()
        self.boundary_points = nn.Parameter(
            torch.Tensor(
                [
                    [0.0, 0.0],
                    [0.0, 224.0],
                    [224.0, 224.0],
                    [224.0, 0.0],
                ]
            )
        )
        self.hard_polygon = SoftPolygon(0.01, mode="hard_mask")

    def forward(self):
        boundary = self.boundary_points
        boundary = torch.clamp(boundary, 0, 224)
        boundary = boundary.unsqueeze(0)
        return boundary

    def add_mid_points(self):
        # add mid points to the boundary
        boundary = self.boundary_points
        boundary_shift = torch.roll(boundary, 1, 0)
        # print(f"boundary: {boundary}")
        # print(f"boundary_shift: {boundary_shift}")
        mid_points = (boundary + boundary_shift) / 2
        # print(f"mid_points: {mid_points}")
        boundary_num = boundary.shape[0]
        new_boundary = torch.zeros(boundary_num * 2, 2).to(boundary.device)
        new_boundary[::2] = mid_points
        new_boundary[1::2] = boundary
        self.boundary_points = nn.Parameter(new_boundary)

    def get_numpy(self) -> np.ndarray:
        boundary = self.forward().squeeze(0).cpu().detach().numpy()
        return boundary

    def get_mask(self):
        boudary = self.forward()
        mask = self.hard_polygon(boudary, 224, 224)
        return mask.squeeze(0).cpu().detach().numpy()


def sample_one_frame(
    mask: torch.Tensor,
    max_point_num: int,
    use_std_loss: bool,
    fir_epoch_multi: int,
    epoch_num=100,
) -> dict:
    boundary_sampler = BoundarySampler().cuda()
    ras_loss = RasLoss().cuda()
    results = {}
    mask_batch = mask.unsqueeze(0).cuda()
    current_point_num = boundary_sampler.boundary_points.shape[0]
    while current_point_num <= max_point_num:
        optimizer = optim.Adam(boundary_sampler.parameters(), lr=1e-0)
        if use_std_loss:
            dif_weight = 0.5
            std_weight = 0.5
        else:
            dif_weight = 1.0
            std_weight = 0.0
        if current_point_num == 4:
            tmp_epoch_num = epoch_num * fir_epoch_multi
        else:
            tmp_epoch_num = epoch_num
        for epoch in range(tmp_epoch_num):
            optimizer.zero_grad()
            boundary_points = boundary_sampler()
            dif_loss = ras_loss(boundary_points, mask_batch)
            std_loss = deviation_loss(boundary_points)
            if std_loss < 0.01:
                total_loss = dif_weight * dif_loss
            else:
                total_loss = dif_weight * dif_loss + std_weight * std_loss
            total_loss.backward()
            optimizer.step()
        iou = get_boundary_iou(
            mask.cpu().detach().numpy(), boundary_sampler.get_numpy()
        )
        current_result = {}
        current_result["boundary"] = boundary_sampler.get_numpy().tolist()
        current_result["iou"] = iou
        results[current_point_num] = current_result
        boundary_sampler.add_mid_points()
        current_point_num = boundary_sampler.boundary_points.shape[0]
    torch.cuda.empty_cache()
    return results


def show_result(img: torch.Tensor, mask: torch.Tensor, result: dict):
    boundary = result["boundary"]
    tensor_boundary = torch.tensor(boundary, dtype=torch.float32).unsqueeze(0).cuda()
    boundary = np.array(boundary)
    hard_polygon = SoftPolygon(0.01, mode="hard_mask").cuda()
    iou = result["iou"]
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(normalize(img).permute(1, 2, 0).cpu().numpy())
    plt.imshow(mask.cpu().numpy(), alpha=0.5, cmap="gray")
    plt.axis("off")
    plt.title("ground truth")
    plt.subplot(1, 3, 2)
    plt.imshow(normalize(img).permute(1, 2, 0).cpu().numpy())
    plt.axis("off")
    plt.plot(
        boundary[:, 0],
        boundary[:, 1],
        "b-",
        # lw=5,
    )
    plt.scatter(
        boundary[:, 0],
        boundary[:, 1],
        c="r",
        s=10,
    )
    plt.title(f"IoU: {iou:.4f}")
    plt.subplot(1, 3, 3)
    pred_mask = hard_polygon(tensor_boundary, 224, 224)
    pred_mask = pred_mask.squeeze(0).cpu().detach().numpy()
    plt.imshow(pred_mask, cmap="spring", vmin=0, vmax=1)
    plt.axis("off")
    plt.title("predicted mask")
    plt.show()


def sample_one_video(
    video_data: list[tuple],
    max_point_num: int,
    use_std_loss: bool,
    fir_epoch_multi: int,
    epoch_num=100,
) -> dict:
    results = []
    for img, mask in video_data:
        result = sample_one_frame(
            mask,
            max_point_num,
            use_std_loss,
            fir_epoch_multi,
            epoch_num,
        )
        results.append(result)
    return results


def sample_save_dataset(
    video_dataset: list,
    start_idx: int,
    save_title: str,
    max_point_num: int,
    use_std_loss: bool,
    fir_epoch_multi: int,
    epoch_num:int,
):
    log_path = f"./log/{save_title}_{start_idx}_{max_point_num}_{"std" if use_std_loss else ""}_{epoch_num}_{fir_epoch_multi}.log"
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    logging.info(f"Start sampling {save_title}_{start_idx}_{max_point_num}_{"std" if use_std_loss else ""}_{epoch_num}.")
    save_path = f"{save_title}_{start_idx}_{max_point_num}_{"std" if use_std_loss else ""}_{epoch_num}_{fir_epoch_multi}.json"
    results = []
    logging.info(f"video num: {len(video_dataset)}")
    for video_idx, video_data in enumerate(video_dataset):
        result = sample_one_video(
            video_data,
            max_point_num,
            use_std_loss,
            fir_epoch_multi,
            epoch_num,
        )
        results.append(result)
        logging.info(f"video {video_idx} done")
    with open(save_path, "w") as f:
        json.dump(
            results,
            f,
        )


def load_results(
    start_idx: int,
    save_title: str,
    max_point_num: int,
    use_std_loss: bool,
    epoch_num=100,
):
    save_path = f"{save_title}_{start_idx}_{max_point_num}_{"std" if use_std_loss else ""}_{epoch_num}.json"
    with open(save_path, "r") as f:
        results = json.load(f)

    def numericalize_dict(d: dict):
        result = {}
        for key, value in d.items():
            result[int(key)] = value
        return result

    # numericalize the results
    for video_result in results:
        for frame_idx, frame_result in enumerate(video_result):
            video_result[frame_idx] = numericalize_dict(frame_result)

    return results
