import torch
import torch.nn.functional as F
from torch import nn


class L1WiringCost(nn.Module):
    def __init__(self, dist: torch.Tensor, lambd: float = 0.01, p: float = 1.0):
        super().__init__()
        self.lambd = lambd
        self.p = p

        weight = dist**p
        self.register_buffer("weight", weight)

    def forward(self, attn: torch.Tensor) -> torch.Tensor:
        # attn assumed to be >= 0 and rows sum to 1
        return self.lambd * (self.weight * attn).sum(dim=-1).mean()

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, lambd={self.lambd}, p={self.p}"


class CrossEntropyWiringCost(nn.Module):
    def __init__(self, dist: torch.Tensor, lambd: float = 0.01, sigma: float = 2.0):
        super().__init__()
        self.lambd = lambd
        self.sigma = sigma

        weight = -0.5 * dist**2 / sigma**2
        weight = -F.log_softmax(weight, dim=-1)
        self.register_buffer("weight", weight)

    def forward(self, attn: torch.Tensor) -> torch.Tensor:
        # attn assumed to be >= 0 and rows sum to 1
        return self.lambd * (self.weight * attn).sum(dim=-1).mean()

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, lambd={self.lambd}, sigma={self.sigma}"
