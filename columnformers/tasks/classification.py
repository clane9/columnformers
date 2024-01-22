from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .losses import l1_wiring_cost


class ImageClassification(nn.Module):
    def __init__(self, wiring_lambd: float = 0.0):
        super().__init__()
        self.wiring_lambd = wiring_lambd

    def forward(
        self, model: nn.Module, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        images = batch["image"]
        target = batch["label"]

        output = model(images)
        output, state = output if isinstance(output, tuple) else (output, {})

        class_loss = F.cross_entropy(output, target)
        if self.wiring_lambd > 0:
            wiring_loss = self.wiring_lambd * l1_wiring_cost(
                state["attns"], model.geometry
            )
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

    def extra_repr(self) -> str:
        return f"wiring_lambd={self.wiring_lambd}"
