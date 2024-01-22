import torch


def l1_wiring_cost(
    attn: torch.Tensor, weight: torch.Tensor, scale: bool = True
) -> torch.Tensor:
    cost = (attn * weight).sum(dim=(-2, -1)).mean()
    if scale:
        cost = cost / weight.max()
    return cost
