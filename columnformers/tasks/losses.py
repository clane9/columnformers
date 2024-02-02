import torch
from torch import nn


class WiringCost(nn.Module):
    weight: torch.Tensor

    def __init__(self, geometry: torch.Tensor, lambd: float = 0.01, p: float = 1.0):
        super().__init__()
        self.lambd = lambd
        self.p = p

        weight = geometry**p
        weight = weight / weight.mean()
        self.register_buffer("weight", weight)

    def forward(self, attn: torch.Tensor) -> torch.Tensor:
        # attn assumed to be non-negative with rows summing to 1
        return self.lambd * (self.weight * attn).sum(dim=-1).mean()

    def extra_repr(self) -> str:
        return f"lambd={self.lambd}, p={self.p}"
