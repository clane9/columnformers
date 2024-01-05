from typing import Dict, Optional, Tuple

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
        """
        Forward pass for depth steps. Returns output features (B, N, C) and attention
        matrices (B, N, N).
        """
        raise NotImplementedError

    def wiring_cost(self, attn: torch.Tensor) -> torch.Tensor:
        """
        Attention wiring cost regularization.
        """
        raise NotImplementedError


class TaskModel(nn.Module):
    """
    Abstract task model interface
    """

    def forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute model forward pass and loss. Returns the loss tensor and a state dict.
        """
        raise NotImplementedError
