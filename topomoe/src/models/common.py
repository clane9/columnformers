import logging
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
from torch import nn
from timm.layers import trunc_normal_
from timm.layers.helpers import to_2tuple, to_3tuple

from topomoe.src.utils import filter_kwargs

State = Dict[str, torch.Tensor]
Layer = Callable[..., nn.Module]


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: Union[bool, Tuple[bool, bool, bool]] = False,
        linear_layer: Layer = nn.Linear,
        q_linear_layer: Optional[Layer] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        if q_linear_layer is None:
            q_linear_layer = linear_layer

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        qkv_biases = to_3tuple(qkv_bias)

        # nb, query weight/bias can break symmetry between branches
        # this is why we decouple it from the other layers
        self.q = q_linear_layer(dim, num_heads * self.head_dim, bias=qkv_biases[0])
        self.k = linear_layer(dim, num_heads * self.head_dim, bias=qkv_biases[1])
        self.v = linear_layer(dim, dim, bias=qkv_biases[2])
        self.proj = linear_layer(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, State]:
        if context is None:
            context = x
        B, N, C = x.shape
        M = context.size(1)
        nh, d = self.num_heads, self.head_dim
        q = self.q(x).reshape(B, N, nh, d).permute(0, 2, 1, 3)
        k = self.k(context).reshape(B, M, nh, d).permute(0, 2, 1, 3)
        v = self.v(context).reshape(B, M, nh, d).permute(0, 2, 1, 3)

        # Nb, no flash attention bc we need the attention matrix
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, {"attn": attn}


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        linear_layer: Layer = nn.Linear,
        act_layer: Optional[Layer] = nn.GELU,
        bias: Union[bool, Tuple[bool, bool]] = True,
        drop: Union[float, Tuple[float, float]] = 0.0,
    ):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        biases = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        act_layer = nn.Identity if act_layer is None else act_layer

        self.fc1 = linear_layer(in_features, hidden_features, bias=biases[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = linear_layer(hidden_features, out_features, bias=biases[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


def init_weights(module: nn.Module):
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def to_list(x, length):
    if not isinstance(x, (list, tuple)):
        x = [x] * length
    elif len(x) == 1:
        x = x * length
    elif len(x) != length:
        raise ValueError(f"Length of x {len(x)} doesn't match target length {length}")
    return x


def model_factory(
    cls: type, params: Dict[str, Any], defaults: Dict[str, Any], **kwargs
):
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    kwargs, extra_args = filter_kwargs(cls, kwargs)
    if extra_args:
        logging.warning("Extra kwargs to %s: %s", cls.__name__, extra_args)
    kwargs = {**defaults, **kwargs}
    model = cls(**params, **kwargs)
    return model
