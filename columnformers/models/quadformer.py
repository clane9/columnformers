"""
The primate visual cortex has a branching organization. Early layers are characterized
by retinotopic mapping, narrow receptive fields, and shared orientation-tuned filters. As
you proceed through the hierarchy, receptive fields widen and tuning becomes more
specialized. The shared trunk splits into independent streams (e.g. dorsal, ventral),
which ultimately terminate in multiple highly specialized areas with near global
receptive field.

The quadformer aims to be a highly oversimplified model of this branching architecture.
It combines the basic architecture of the transformer with the pooling from convnets and
branching from mixtures of experts. The architecture consists of a sequence of stages.
After the first stage, later stages start by pooling the input and "branching", i.e.
tiling the pooled input. Each branch occupies a block in the patch grid, which are then
processed by independent attention and mlp modules. The block attention modules pull in
relevant information from other patches within the same block as well as from other
blocks. At the start of each stage, attention is also able to pull in higher resolution
information from the pre-pooled input. The Mlp modules then compute branch-specific
features.

With this architecture, we will see what kinds of specialization the independent
branches learn, as well as if there is an accuracy / flops benefit to having
specialized branches compared to classic ViTs.
"""

import logging
import math
from functools import partial
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from timm.layers import PatchEmbed, trunc_normal_
from timm.layers.helpers import to_2tuple, to_3tuple
from torch import nn

from columnformers.utils import filter_kwargs

from .registry import register_model

State = Dict[str, torch.Tensor]
Layer = Callable[..., nn.Module]


class QuadPool(nn.Module):
    """
    Input is a grid of BxB blocks. Separately in each block, do 2x2 average pooling
    followed by 2x2 tiling of the pooled block.
    """

    def __init__(self, block: int):
        super().__init__()
        self.block = block

    def forward(self, input: torch.Tensor):
        N, L, C = input.shape
        B = self.block
        H = math.isqrt(L) // B
        assert L == (B * H) ** 2
        assert H % 2 == 0

        # extract each block, shape (batch, *, block_height, block_height)
        input = rearrange(
            input,
            "n (b1 h1 b2 h2) c -> n (b1 b2 c) h1 h2",
            b1=B,
            h1=H,
            b2=B,
            h2=H,
        )

        # 2x2 average pooling
        output = F.avg_pool2d(input, kernel_size=2, stride=2)
        # repeat pooled input 4x
        output = output.unsqueeze(1).repeat(1, 4, 1, 1, 1)

        # reshape back to original
        # new_block = 2 * block, new_block_height = block_height / 2
        # shape (batch, seq_len, dim)
        output = rearrange(
            output,
            "n (a1 a2) (b1 b2 c) h1 h2 -> n (b1 a1 h1 b2 a2 h2) c",
            a1=2,
            b1=B,
            h1=H // 2,
            a2=2,
            b2=B,
            h2=H // 2,
        )
        return output

    def extra_repr(self) -> str:
        return f"{self.block}"


class BlockLinear(nn.Module):
    """
    Input is a grid of BxB blocks. Independent linear layer for each block.
    """

    def __init__(
        self,
        block: int,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ):
        super().__init__()
        self.block = block
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.empty((block * block, out_features, in_features))
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(block * block, out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        trunc_normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        N, L, C = input.shape
        B = self.block
        H = math.isqrt(L) // B
        assert L == (B * H) ** 2

        # pull out blocks, push in batch dim for batch matmul
        input = rearrange(
            input,
            "n (b1 h1 b2 h2) c -> (b1 b2) (n h1 h2) c",
            b1=B,
            h1=H,
            b2=B,
            h2=H,
        )
        # (blocks, *, in_dim) -> (blocks, *, out_dim)
        output = input @ self.weight.transpose(1, 2)

        if self.bias is not None:
            output = output + self.bias.unsqueeze(1)

        # reshape back to original
        output = rearrange(
            output,
            "(b1 b2) (n h1 h2) c -> n (b1 h1 b2 h2) c",
            b1=B,
            h1=H,
            b2=B,
            h2=H,
        )
        return output

    def extra_repr(self) -> str:
        return (
            f"{self.block}, {self.in_features}, {self.out_features}, "
            f"bias={self.bias is not None}"
        )


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


class Block(nn.Module):
    def __init__(
        self,
        dim: int = 384,
        block: int = 1,
        pool: bool = False,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: Union[bool, Tuple[bool, bool, bool]] = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        act_layer: Layer = nn.GELU,
    ):
        super().__init__()
        assert not pool or block % 2 == 0, "block must be divisible by 2"

        # Nb, norm layer always shared
        norm_layer = nn.LayerNorm
        if block > 1:
            linear_layer = partial(BlockLinear, block)
        else:
            linear_layer = nn.Linear

        self.norm1 = norm_layer(dim)
        self.pool = QuadPool(block // 2) if pool else nn.Identity()
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            # only q is decoupled across blocks, k, v, proj are shared
            q_linear_layer=linear_layer,
            linear_layer=nn.Linear,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            linear_layer=linear_layer,
            act_layer=act_layer,
            drop=proj_drop,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, State]:
        x = self.norm1(x)
        # pool inputs and branch
        # input -> 2x2 pooling -> 2x2 tiling
        # independently in each block, resulting in a quad tree like branching structure
        pooled = self.pool(x)

        # attention with queries from pooled input and keys/values from full input
        # independent weights per block
        # Nb, the pooled input to each block is identical, but the independent weights
        # should break the symmetry
        x, attn_state = self.attn(pooled, x)
        # residual on top of pooled input
        # the query/residual is the main path; all other information is pulled in
        # selectively via attention
        x = x + pooled

        # standard mlp, but independent weights per block
        x = x + self.mlp(self.norm2(x))

        state = {**attn_state, "features": x}
        return x, state


class Quadformer(nn.Module):
    def __init__(
        self,
        img_size: int = 128,
        patch_size: int = 16,
        in_chans: int = 3,
        depth: int = 12,
        pool_stages: Tuple[int, ...] = (4, 8),
        embed_dim: int = 384,
        num_heads: int = 8,
        qkv_bias: Union[bool, Tuple[bool, bool, bool]] = True,
        mlp_ratio: Union[float, List[float]] = 4.0,
        mlp_conserve: bool = False,
        act_layer: Layer = nn.GELU,
        attn_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        global_pool: Literal["", "avg"] = "avg",
        num_classes: int = 100,
        drop_rate: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.global_pool = global_pool
        num_patches = (img_size // patch_size) ** 2

        blocks = np.ones(depth, dtype=np.int64)
        for idx in pool_stages:
            blocks[idx:] *= 2
        blocks = blocks.tolist()

        mlp_ratio = _to_list(mlp_ratio, depth)
        if mlp_conserve:
            mlp_ratio = [r / (b * b) for r, b in zip(mlp_ratio, blocks)]

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.pos_embed = nn.Parameter(torch.empty(1, num_patches, embed_dim))

        self.blocks = nn.ModuleList(
            Block(
                dim=embed_dim,
                block=blocks[ii],
                pool=ii in pool_stages,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio[ii],
                qkv_bias=qkv_bias,
                attn_drop=attn_drop_rate,
                proj_drop=proj_drop_rate,
                act_layer=act_layer,
            )
            for ii in range(depth)
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.head_drop = nn.Dropout(drop_rate)
        if num_classes:
            self.head = nn.Linear(embed_dim, num_classes)
        else:
            self.head = nn.Identity()

        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(_init_weights)

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, State]:
        x = self.patch_embed(x)
        x = x + self.pos_embed

        states = []
        keys = set()
        for block in self.blocks:
            x, state = block(x)

            states.append(state)
            keys.update(state.keys())

        state = {key: torch.stack([s.get(key) for s in states], dim=1) for key in keys}
        return x, state

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        if self.global_pool == "avg":
            x = x.mean(dim=1)
        x = self.head_drop(x)
        x = self.head(x)
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, State]:
        x, state = self.forward_features(x)
        x = self.forward_head(x)
        return x, state


def _init_weights(module: nn.Module):
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _to_list(x, length):
    if not isinstance(x, (list, tuple)):
        x = [x] * length
    elif len(x) == 1:
        x = x * length
    return x


def _create_quadformer(params: Dict[str, Any], defaults: Dict[str, Any], **kwargs):
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    kwargs, extra_args = filter_kwargs(Quadformer, kwargs)
    if extra_args:
        logging.warning("Extra kwargs to Quadformer: %s", extra_args)
    kwargs = {**defaults, **kwargs}
    model = Quadformer(**params, **kwargs)
    return model


@register_model
def quadformer_tiny_patch16_128(**kwargs):
    params = {
        "img_size": 128,
        "patch_size": 16,
        "in_chans": 3,
        "depth": 6,
        "embed_dim": 384,
    }
    defaults = {
        "pool_stages": (2, 4),
        "num_heads": 6,
    }
    model = _create_quadformer(params, defaults, **kwargs)
    return model
