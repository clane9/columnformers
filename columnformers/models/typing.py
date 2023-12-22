from typing import Optional, Tuple

import torch
from torch import nn


class Columnformer(nn.Module):
    """
    Abstract Columnformer interface.
    """

    dist: torch.Tensor
    seq_len: int
    embed_dim: int
    inner_dim: int
    depth: int
    sheet: nn.Module

    def forward(
        self, x: torch.Tensor, depth: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def wiring_cost(self, attn: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
