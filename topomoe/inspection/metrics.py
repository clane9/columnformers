import re
from typing import Dict

import torch
from timm.utils.metrics import accuracy
from torch import nn

from .registry import register_metric


@register_metric
class Accuracy(nn.Module):
    name = "accuracy"

    def forward(self, state: Dict[str, torch.Tensor]):
        output = state.get("output")
        target = state.get("target")
        if output is not None and target is not None:
            acc1 = accuracy(output.detach(), target.detach())[0]
            return {self.name: acc1}
        return {}


@register_metric
class AttentionEntropy(nn.Module):
    name = "attn_entropy"
    pattern = re.compile(r"\.attn")

    def forward(self, state: Dict[str, torch.Tensor]):
        metrics = {}
        for k, v in state.items():
            if self.pattern.search(k) and v is not None:
                metrics[f"{self.name}-{k}"] = attention_entropy(v.detach())
        return metrics


def attention_entropy(attn: torch.Tensor, eps: float = 1e-8):
    """
    Entropy of the attention matrix.

    Reference:
        https://github.com/apple/ml-sigma-reparam
    """
    return -(attn * (attn + eps).log()).sum(dim=-1).mean()
