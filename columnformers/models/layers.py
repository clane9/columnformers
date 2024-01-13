from typing import Callable

import torch
import torch.nn.functional as F
from timm.layers import trunc_normal_
from torch import nn

Layer = Callable[..., nn.Module]


class UntiedLinear(nn.Module):
    """
    Linear layer with untied weights across the sequence.

    References:
        nn.Linear
        ViT linear init
    """

    def __init__(
        self,
        seq_len: int,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty((seq_len, out_features, in_features)))
        if bias:
            self.bias = nn.Parameter(torch.empty(seq_len, out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        trunc_normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = torch.einsum("bnc,ndc->bnd", input, self.weight)
        if self.bias is not None:
            output = output + self.bias
        return output

    def extra_repr(self) -> str:
        return (
            f"{self.seq_len}, {self.in_features}, {self.out_features}, "
            f"bias={self.bias is not None}"
        )


class UntiedLayerNorm(nn.Module):
    """
    Layer norm with untied weights across the sequence.
    """

    def __init__(
        self,
        seq_len: int,
        dim: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.empty(seq_len, dim))
            self.bias = nn.Parameter(torch.empty(seq_len, dim))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = F.layer_norm(input, (self.dim,), eps=self.eps)
        if self.elementwise_affine:
            input = input * self.weight + self.bias
        return input

    def extra_repr(self) -> str:
        return (
            f"{self.seq_len}, {self.dim}, eps={self.eps}, "
            f"elementwise_affine={self.elementwise_affine}"
        )
