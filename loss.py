import torch


def chamer_distance_loss(x, y):
    def bi_chamfer_distance(x, y):
        def chamfer_distance(a: torch.Tensor, b: torch.Tensor) -> float:
            total_min_dist = 0
            for i in range(a.shape[0]):
                total_min_dist += (b - a[i]).pow(2).sum(1).min(0)[0]
            return total_min_dist / a.shape[0]

        return chamfer_distance(x, y) + chamfer_distance(y, x)

    result = torch.tensor(0.0).to(x.device)
    for i in range(x.shape[0]):
        result += bi_chamfer_distance(x[i], y[i])

    return result / x.shape[0]


def order_loss(x: torch.Tensor, y: torch.Tensor):
    def my_order_loss(x, y):
        min_loss = (x - y).abs().sum()
        for shift in range(1, x.shape[0]):
            # loss = (x - torch.roll(y, shifts=shift, dims=0)).pow(2).sum().sqrt()
            loss = (x - torch.roll(y, shifts=shift, dims=0)).abs().sum()
            min_loss = torch.min(min_loss, loss)
        return min_loss / x.shape[0]
        # return min_loss

    result = torch.tensor(0.0).to(x.device)
    for i in range(x.shape[0]):
        result += my_order_loss(x[i], y[i])

    return result / x.shape[0]
