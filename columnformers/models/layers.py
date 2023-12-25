import torch
from timm.models.layers import trunc_normal_
from torch import nn


class GlobalAveragePool(nn.Module):
    """
    Global average pooling over a sequence.

    Args:
        pool_len: Length of the sequence, N.

    Shape:
        - Input: (B, N, C)
        - Output: (B, C)
    """

    def __init__(self, pool_len: int):
        super().__init__()
        self.pool_len = pool_len

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.shape[1] >= self.pool_len, "input length must be >= pool_len"
        input = input[:, -self.pool_len :]
        output = input.mean(dim=1)
        return output

    def extra_repr(self) -> str:
        return f"{self.pool_len}"


class SpatialPool(nn.Module):
    """
    Pool a sequence of features with a learned attention weight per class.

    Args:
        pool_len: Length of the sequence, N.
        num_classes: Number of classes, K.
        drop: Dropout probability.

    Shape:
        - Input: (B, N, C)
        - Output: (B, K, C)
    """

    def __init__(self, pool_len: int, num_classes: int, drop: float = 0.0):
        super().__init__()
        self.pool_len = pool_len
        self.num_classes = num_classes

        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()
        self.weight = nn.Parameter(torch.empty(num_classes, pool_len))
        self.reset_parameters()

    def reset_parameters(self):
        trunc_normal_(self.weight, std=0.02)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.shape[1] >= self.pool_len, "input length must be >= pool_len"
        input = input[:, -self.pool_len :]
        attn = torch.softmax(self.weight, dim=1)
        attn = self.drop(attn)
        output = attn @ input
        return output

    def extra_repr(self) -> str:
        return f"{self.pool_len}, {self.num_classes}"
