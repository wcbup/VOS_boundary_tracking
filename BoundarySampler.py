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
        init_boundary=None,
    ):
        super(BoundarySampler, self).__init__()
        if init_boundary is not None:
            self.boundary_points = nn.Parameter(init_boundary)
        else:
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

    def get_mask(self, hard_polygon: nn.Module):
        boudary = self.forward()
        mask = hard_polygon(boudary, 224, 224)
        return mask.squeeze(0).cpu().detach().numpy()

    def get_extreme_4_points(self):
        boundary = self.forward().squeeze(0)
        x_min = torch.min(boundary[:, 0])
        x_max = torch.max(boundary[:, 0])
        y_min = torch.min(boundary[:, 1])
        y_max = torch.max(boundary[:, 1])
        return torch.Tensor(
            [
                [x_min, y_min],
                [x_min, y_max],
                [x_max, y_max],
                [x_max, y_min],
            ],
        ).to(boundary.device)

def get_polygon_iou(
    polygon: torch.Tensor,
    mask: torch.Tensor,
    hard_polygon: nn.Module,
) -> float:
    if mask.sum() == 0:
        return 1
    polygon_batch = polygon.unsqueeze(0)
    mask_batch = mask.unsqueeze(0)
    ras_mask = hard_polygon(
        polygon_batch,
        mask_batch.shape[1],
        mask_batch.shape[2],
    )
    ras_mask = ras_mask.squeeze(0)
    ras_mask[ras_mask == -1] = 0
    interction = torch.sum(ras_mask * mask)
    union = torch.sum(ras_mask) + torch.sum(mask) - interction
    iou = interction / union
    return iou.item()

def sample_one_frame(
    mask: torch.Tensor,
    max_point_num: int,
    use_std_loss: bool,
    fir_epoch_multi: int,
    hard_polygon: nn.Module,
    epoch_num: int,
    ras_loss: nn.Module,
    boundary_sampler: BoundarySampler,
) -> dict:
    results = {}
    mask_batch = mask.unsqueeze(0).cuda()
    current_point_num = boundary_sampler.boundary_points.shape[0]
    if use_std_loss:
        dif_weight = 0.5
        std_weight = 0.5
    else:
        dif_weight = 1.0
        std_weight = 0.0
    while current_point_num <= max_point_num:
        optimizer = optim.Adam(boundary_sampler.parameters(), lr=1)
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
        iou = get_polygon_iou(
            boundary_sampler().squeeze(0),
            mask_batch.squeeze(0),
            hard_polygon,
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
    hard_polygon: nn.Module,
    epoch_num: int,
    ras_loss: nn.Module,
    min_threshold: int,
) -> list:
    results = []
    boundary_sampler = BoundarySampler().cuda()
    for frame_idx, (img, mask) in enumerate(video_data):
        if mask.sum() < min_threshold:
            tmp_fir_epoch_multi = fir_epoch_multi * 5
        else:
            tmp_fir_epoch_multi = fir_epoch_multi
        if frame_idx == 0:
            results.append(
                sample_one_frame(
                    mask=mask,
                    max_point_num=max_point_num,
                    use_std_loss=use_std_loss,
                    fir_epoch_multi=tmp_fir_epoch_multi,
                    hard_polygon=hard_polygon,
                    epoch_num=epoch_num,
                    ras_loss=ras_loss,
                    boundary_sampler=boundary_sampler,
                ),
            )
        else:
            refine_result = sample_one_frame(
                mask=mask,
                max_point_num=max_point_num,
                use_std_loss=use_std_loss,
                fir_epoch_multi=1,
                hard_polygon=hard_polygon,
                epoch_num=epoch_num,
                ras_loss=ras_loss,
                boundary_sampler=boundary_sampler,
            )
            new_boundary_sampler = BoundarySampler().cuda()
            new_result = sample_one_frame(
                mask=mask,
                max_point_num=max_point_num,
                use_std_loss=use_std_loss,
                fir_epoch_multi=tmp_fir_epoch_multi,
                hard_polygon=hard_polygon,
                epoch_num=epoch_num,
                ras_loss=ras_loss,
                boundary_sampler=new_boundary_sampler,
            )
            if refine_result[256]["iou"] > new_result[256]["iou"]:
                results.append(refine_result)
            else:
                results.append(new_result)
                boundary_sampler = new_boundary_sampler
        boundary_sampler = BoundarySampler(boundary_sampler.get_extreme_4_points()).cuda()
    return results


def sample_save_dataset(
    video_dataset: list,
    start_idx: int,
    save_title: str,
    max_point_num: int,
    use_std_loss: bool,
    fir_epoch_multi: int,
    epoch_num: int,
    min_threshold: int,
):
    title = f"{save_title}_{start_idx}_{max_point_num}_{"std" if use_std_loss else ""}_{epoch_num}_{fir_epoch_multi}_{min_threshold}"
    log_path = f"./log/{title}.log"
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    logging.info(f"Start sampling {title}.")

    save_path = f"./sample_results/{title}.json"

    results = []
    logging.info(f"video num: {len(video_dataset)}")
    ras_loss = RasLoss().cuda()
    hard_polygon = SoftPolygon(0.01, mode="hard_mask").cuda()
    for video_idx, video_data in enumerate(video_dataset):
        result = sample_one_video(
            video_data=video_data,
            max_point_num=max_point_num,
            use_std_loss=use_std_loss,
            fir_epoch_multi=fir_epoch_multi,
            hard_polygon=hard_polygon,
            epoch_num=epoch_num,
            ras_loss=ras_loss,
            min_threshold=min_threshold,
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
