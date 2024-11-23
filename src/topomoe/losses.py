from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn

State = Dict[str, torch.Tensor]


class VisionTask(nn.Module):
    def __init__(self, losses: Dict[str, nn.Module]):
        super().__init__()
        self.losses = nn.ModuleDict(losses)

    def forward(self, model: nn.Module, batch: State) -> Tuple[torch.Tensor, State]:
        images = batch["image"]
        output, state = model(images)

        loss_dict = {}
        for name, loss_fn in self.losses.items():
            loss_dict[name] = loss_fn(batch, output, state)

        total_loss = sum(loss_dict.values())

        state = {
            **batch,
            "output": output,
            **state,
            **loss_dict,
        }
        return total_loss, state


class CrossEntropyLoss(nn.Module):
    def forward(self, batch: State, output: torch.Tensor, state: State) -> torch.Tensor:
        target = batch["label"]
        return F.cross_entropy(output, target)
