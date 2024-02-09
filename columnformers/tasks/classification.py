from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .losses import WiringCost


class ImageClassification(nn.Module):
    def __init__(self, wiring_cost: Optional[WiringCost] = None):
        super().__init__()
        if wiring_cost is None:
            self.register_module("wiring_cost", None)
        else:
            self.wiring_cost = wiring_cost

    def forward(
        self, model: nn.Module, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        images = batch["image"]
        target = batch["label"]

        output = model(images)
        output, state = output if isinstance(output, tuple) else (output, {})

        class_loss = F.cross_entropy(output, target)
        if self.wiring_cost is not None:
            wiring_loss = self.wiring_cost(state["attns"])
        else:
            wiring_loss = torch.zeros_like(class_loss)
        loss = class_loss + wiring_loss

        state = {
            "image": images,
            "label": target,
            **state,
            "output": output,
            "class_loss": class_loss,
            "wiring_loss": wiring_loss,
        }
        return loss, state
