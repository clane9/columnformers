from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class L1WiringCost(nn.Module):
    """
    Distance weighted L1 wiring cost. Distance can be raised to a power to penalize
    long-range connections more strongly.
    """

    def __init__(
        self,
        geo_embed: torch.Tensor,
        in_geo_embed: Optional[torch.Tensor] = None,
        lambd: float = 0.01,
        p: float = 1.0,
    ):
        super().__init__()
        self.lambd = lambd
        self.p = p

        if in_geo_embed is None:
            in_geo_embed = geo_embed
        dist = torch.cdist(geo_embed, in_geo_embed)

        weight = dist**p
        self.register_buffer("weight", weight)

    def forward(self, attn: torch.Tensor) -> torch.Tensor:
        # attn assumed to be >= 0 and rows sum to 1
        return self.lambd * (self.weight * attn).sum(dim=-1).mean()

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, lambd={self.lambd}, p={self.p}"


class CrossEntropyWiringCost(nn.Module):
    """
    Cross-entropy wiring cost, penalizing `- (A * W.log()).sum()` where `W` is based on
    gaussian weighted probabilities. `W = softmax(0.5 * D ** 2 / sigma ** 2)`.

    Similar to the distance-weighted L1 wiring cost with `p=2`, but does not decay all
    the way to 0 for distance 0 or penalize long connections quite as strongly.
    """

    def __init__(
        self,
        geo_embed: torch.Tensor,
        in_geo_embed: Optional[torch.Tensor] = None,
        lambd: float = 0.01,
        sigma: float = 2.0,
    ):
        super().__init__()
        self.lambd = lambd
        self.sigma = sigma

        if in_geo_embed is None:
            in_geo_embed = geo_embed
        dist = torch.cdist(geo_embed, in_geo_embed)

        weight = -0.5 * dist**2 / sigma**2
        weight = -F.log_softmax(weight, dim=-1)
        self.register_buffer("weight", weight)

    def forward(self, attn: torch.Tensor) -> torch.Tensor:
        # attn assumed to be >= 0 and rows sum to 1
        return self.lambd * (self.weight * attn).sum(dim=-1).mean()

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, lambd={self.lambd}, sigma={self.sigma}"
