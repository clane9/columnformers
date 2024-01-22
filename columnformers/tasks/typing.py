from typing import Dict, Tuple

import torch
from torch import nn


class Task(nn.Module):
    """
    Abstract task interface.
    """

    def forward(
        self, model: nn.Module, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute model forward pass and loss. Returns the loss tensor and a state dict.
        """
        raise NotImplementedError
