import torch


def get_edges(pred_bou: torch.Tensor) -> torch.Tensor:
    pred_bou_shift = torch.roll(pred_bou, 1, 1)
    pred_bou_offset = torch.square(pred_bou - pred_bou_shift)
    pred_edges = torch.sum(pred_bou_offset, dim=2)
    pred_edges = torch.abs(pred_edges)
    pred_edges = torch.sqrt(pred_edges)
    return pred_edges


def deviation_loss(pred_bou: torch.Tensor, max_coord=224) -> torch.Tensor:
    pred_bou = pred_bou / max_coord
    pred_edges = get_edges(pred_bou)
    pred_edge_mean = torch.mean(pred_edges, 1, keepdim=True)
    pred_edge_offset = pred_edges - pred_edge_mean
    pred_edge_deviation = torch.mean(
        torch.square(pred_edge_offset),
        1,
        keepdim=False,
    )
    pred_edge_deviation = torch.abs(pred_edge_deviation)
    pred_edge_deviation = torch.sqrt(pred_edge_deviation)
    return pred_edge_deviation.mean()

def total_len_loss(pred_bou: torch.Tensor, max_coord=224) -> torch.Tensor:
    pred_bou = pred_bou / max_coord
    pred_edges = get_edges(pred_bou)
    return pred_edges.mean()