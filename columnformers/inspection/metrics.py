from typing import Dict

import torch
from timm.utils.metrics import accuracy
from torch import nn

from .registry import register_metric


@register_metric("attn_entropy")
class AttentionEntropy(nn.Module):
    def forward(self, state: Dict[str, torch.Tensor]):
        attns = state.get("attns")
        if attns is None:
            return float("nan")
        return attention_entropy(attns).item()


def attention_entropy(attn: torch.Tensor, eps: float = 1e-8):
    """
    Entropy of the attention matrix.

    Reference:
        https://github.com/apple/ml-sigma-reparam
    """
    return -(attn * (attn + eps).log()).sum(dim=-1).mean()


@register_metric("accuracy")
class Accuracy(nn.Module):
    def forward(self, state: Dict[str, torch.Tensor]):
        output = state.get("output")
        target = state.get("label")
        if output is None or target is None:
            return float("nan")
        return accuracy(output, target)[0].item()
