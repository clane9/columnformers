from typing import Callable, List

import torch
import torch.nn.functional as F
from timm.layers import trunc_normal_
from torch import nn

Layer = Callable[..., nn.Module]


class UntiedLinear(nn.Module):
    """
    Linear layer with untied weights across the sequence.
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


class MixtureLinear(nn.Module):
    """
    Mixture of linear layers. The linear weights for each token in the sequence are
    computed as a linear combination of the weights in the mixture.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        self.weight = nn.Parameter(torch.empty((out_features, in_features, rank)))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, rank))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        trunc_normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor, coef: torch.Tensor) -> torch.Tensor:
        # input: (B, N, C)
        # coef: (N, R)
        # Nb, this implementation for some reason uses significantly fewer flops
        # compared to equivalent alternatives (e.g. einsum) for some reason.
        weight = (coef @ self.weight.transpose(1, 2)).transpose(0, 1)
        if self.bias is not None:
            bias = coef @ self.bias.t()
        output = torch.einsum("bnc,ndc->bnd", input, weight)
        if self.bias is not None:
            output = output + bias
        return output

    def extra_repr(self) -> str:
        return (
            f"{self.in_features}, {self.out_features}, {self.rank}, "
            f"bias={self.bias is not None}"
        )


class MixtureCoefficients(nn.Module):
    def __init__(
        self,
        seq_len: int,
        rank: int = 16,
        softmax: bool = True,
        temp_scale: bool = True,
    ):
        super().__init__()
        assert not temp_scale or softmax, "temp_scale requires softmax"

        self.seq_len = seq_len
        self.rank = rank
        self.softmax = softmax
        self.temp_scale = temp_scale

        self.weight = nn.Parameter(torch.empty((seq_len, rank)))
        # scale is a per-token scaling in log-space following CLIP
        if self.temp_scale:
            self.scale = nn.Parameter(torch.empty((seq_len)))
        else:
            self.register_parameter("scale", None)
        self.reset_parameters()

    def reset_parameters(self):
        trunc_normal_(self.weight, std=self.rank**-0.5)
        if self.temp_scale:
            nn.init.zeros_(self.scale)

    def forward(self) -> torch.Tensor:
        coef = self.weight.clone()
        if self.temp_scale:
            scale = torch.clamp(torch.exp(self.scale), max=100)
            coef = scale[:, None] * F.normalize(coef, dim=1)
        if self.softmax:
            coef = coef.softmax(dim=1)
        return coef

    def extra_repr(self) -> str:
        return (
            f"{self.seq_len}, {self.rank}, softmax={self.softmax}, "
            f"temp_scale={self.temp_scale}"
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

    def no_weight_decay(self) -> List[str]:
        # Nb, not excluded by default since 2d
        return ["weight", "bias"]

    def extra_repr(self) -> str:
        return (
            f"{self.seq_len}, {self.dim}, eps={self.eps}, "
            f"elementwise_affine={self.elementwise_affine}"
        )


class SpatialPool(nn.Module):
    """
    Pool a sequence of features with a learned attention weight per class.

    Args:
        seq_len: Length of the sequence, N.
        num_classes: Number of classes, K.
        drop: Dropout probability.

    Shape:
        - Input: (B, N, C)
        - Output: (B, K, C)
    """

    def __init__(self, seq_len: int, num_classes: int, drop: float = 0.0):
        super().__init__()
        self.seq_len = seq_len
        self.num_classes = num_classes

        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()
        self.weight = nn.Parameter(torch.empty(num_classes, seq_len))
        self.reset_parameters()

    def reset_parameters(self):
        trunc_normal_(self.weight, std=0.2)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        attn = torch.softmax(self.weight, dim=1)
        attn = self.drop(attn)
        output = attn @ input
        return output

    def extra_repr(self) -> str:
        return f"{self.seq_len}, {self.num_classes}"


def init_weights(module: nn.Module):
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, "init_weights"):
        module.init_weights()
