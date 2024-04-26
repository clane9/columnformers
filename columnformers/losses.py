import math
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


class L1WiringCost(nn.Module):
    weight: torch.Tensor

    def __init__(self, geometry: torch.Tensor, lambd: float = 0.1, p: float = 1.0):
        super().__init__()
        self.lambd = lambd
        self.p = p

        weight = geometry**p
        weight = weight / weight.mean()
        self.register_buffer("weight", weight)

    def forward(self, batch: State, output: torch.Tensor, state: State) -> torch.Tensor:
        attn = state.get("attn")
        if attn is None:
            return torch.zeros((), device=output.device, dtype=output.dtype)
        loss = self.lambd * (self.weight * attn).abs().sum(dim=-1).mean()
        return loss

    def extra_repr(self) -> str:
        return f"lambd={self.lambd}, p={self.p}"


class TVMixtureLoss(nn.Module):
    # todo: not sure if this idea is the best. tv loss on coefficients, which are
    # softmaxed within each layer. maybe local cross entropy could be better?

    def __init__(self, lambd: float = 0.001):
        super().__init__()
        self.lambd = lambd

    def forward(self, batch: State, output: torch.Tensor, state: State) -> torch.Tensor:
        coef = state.get("coef")
        if coef is None:
            return torch.zeros((), device=output.device, dtype=output.dtype)

        coef = torch.cat([c for c in coef if c is not None], dim=1)
        N, E = coef.shape
        H = math.isqrt(N)
        coef = coef.reshape(H, H, E)
        loss = (self.lambd / E) * (
            (coef[1:] - coef[:-1]).abs().sum()
            + (coef[:, 1:] - coef[:, :-1]).abs().sum()
        )
        return loss

    def extra_repr(self) -> str:
        return f"lambd={self.lambd}"


class CrossEntropyLoss(nn.Module):
    def forward(self, batch: State, output: torch.Tensor, state: State) -> torch.Tensor:
        target = batch["label"]
        return F.cross_entropy(output, target)
