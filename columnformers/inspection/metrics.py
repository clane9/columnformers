from typing import Dict

import torch
from timm.utils.metrics import accuracy

from .registry import register_metric


@register_metric("attn_entropy")
class AttentionEntropy:
    def __call__(self, state: Dict[str, torch.Tensor]):
        attn = state.get("attn")
        if attn is None:
            return float("nan")
        return attention_entropy(attn).item()


def attention_entropy(attn: torch.Tensor):
    """
    Entropy of the attention matrix.

    Reference:
        https://github.com/apple/ml-sigma-reparam
    """
    return -(attn * attn.log()).sum(dim=-1).mean()


@register_metric("accuracy")
class Accuracy:
    def __call__(self, state: Dict[str, torch.Tensor]):
        output = state.get("output")
        target = state.get("label")
        if output is None or target is None:
            return float("nan")
        return accuracy(output, target)[0].item()
