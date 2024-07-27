"""
Quad-tree MoE transformer (QuadMoE)

The QuadMoE is a simplified baseline for the TopoMoE transformer. Like the
transformer, it consists of a series of stages consisting of pooling and topographic MoE
blocks. But rather than learn the pooling and expert assignment maps, we hand-design
them. The representation map at each stage is the same size. To pool from one stage to
the next, we do 2x2 average pooling and then tile the result as a 2x2 grid. In later
stages, this is applied recursively for each block from the prior stage. Experts are
then statically assigned to the corresponding blocks in the map grid. For increasing
stages, the assignment maps resemble the branches of a quad tree.
"""

import math
from functools import partial
from typing import List, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from einops import rearrange
from timm.layers import PatchEmbed, trunc_normal_
from torch import nn

from .common import Attention, Layer, Mlp, State, init_weights, model_factory, to_list
from .registry import register_model


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


class Block(nn.Module):
    def __init__(
        self,
        block: int = 1,
        dim: int = 384,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: Union[bool, Tuple[bool, bool, bool]] = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        act_layer: Layer = nn.GELU,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        linear_layer = partial(BlockLinear, block) if block > 1 else nn.Linear
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            # only q is decoupled across blocks, k, v, proj are shared
            linear_layer=nn.Linear,
            q_linear_layer=linear_layer,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            linear_layer=linear_layer,
            act_layer=act_layer,
            drop=proj_drop,
        )

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, State]:
        # x: pooled input
        # context: full resolution input

        # attention with queries from pooled input and keys/values from full input
        # independent weights per block
        # Nb, the pooled input to each block is identical, but the independent weights
        # should break the symmetry
        attend, attn_state = self.attn(
            self.norm1(x),
            self.norm1(context) if context is not None else None,
        )
        # residual on top of pooled input
        # the query/residual is the main path; all other information is pulled in
        # selectively via attention
        x = x + attend

        # standard mlp, but independent weights per block
        x = x + self.mlp(self.norm2(x))

        state = {**attn_state, "features": x}
        return x, state


class Stage(nn.Module):
    def __init__(
        self,
        in_block: int = 1,
        pool: bool = False,
        depth: int = 1,
        dim: int = 384,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: Union[bool, Tuple[bool, bool, bool]] = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        act_layer: Layer = nn.GELU,
    ):
        super().__init__()
        block = in_block * 2 if pool else in_block
        if pool:
            self.pool = QuadPool(in_block)
        else:
            self.register_module("pool", None)
        self.blocks = nn.ModuleList(
            Block(
                block=block,
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                act_layer=act_layer,
            )
            for ii in range(depth)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, State]:
        # pool inputs and branch
        # input -> 2x2 pooling -> 2x2 tiling
        # independently in each block, resulting in a quad tree like branching structure
        if self.pool:
            x, context = self.pool(x), x
        else:
            context = None

        state = {}
        for ii, block in enumerate(self.blocks):
            x, block_state = block(x, context)
            state.update({f"blocks.{ii}.{k}": v for k, v in block_state.items()})
            context = None
        return x, state


class QuadMoETransformer(nn.Module):
    def __init__(
        self,
        img_size: int = 128,
        patch_size: int = 16,
        in_chans: int = 3,
        depths: Tuple[int, ...] = (4, 4, 4),
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

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.pos_embed = nn.Parameter(torch.empty(1, num_patches, embed_dim))

        stages = []
        mlp_ratio = to_list(mlp_ratio, len(depths))

        for ii, (depth, ratio) in enumerate(zip(depths, mlp_ratio)):
            # do pooling and double blocks after first stage
            pool = ii > 0
            in_block = 1 if ii == 0 else 2 ** (ii - 1)
            block = in_block * 2 if pool else in_block

            if mlp_conserve:
                ratio = ratio / (block * block)

            stage = Stage(
                in_block=in_block,
                pool=pool,
                depth=depth,
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=ratio,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop_rate,
                proj_drop=proj_drop_rate,
                act_layer=act_layer,
            )
            stages.append(stage)

        self.stages = nn.ModuleList(stages)

        self.norm = nn.LayerNorm(embed_dim)
        self.head_drop = nn.Dropout(drop_rate)
        if num_classes:
            self.head = nn.Linear(embed_dim, num_classes)
        else:
            self.head = nn.Identity()

        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(init_weights)

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, State, State]:
        x = self.patch_embed(x)
        x = x + self.pos_embed

        losses = {}
        state = {}
        for ii, stage in enumerate(self.stages):
            x, stage_state = stage(x)
            state.update({f"stages.{ii}.{k}": v for k, v in stage_state.items()})

        return x, losses, state

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        if self.global_pool == "avg":
            x = x.mean(dim=1)
        x = self.head_drop(x)
        x = self.head(x)
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, State, State]:
        x, losses, state = self.forward_features(x)
        x = self.forward_head(x)
        return x, losses, state


@register_model
def quadmoe_tiny_1s_patch16_128(**kwargs):
    params = {
        "img_size": 128,
        "patch_size": 16,
        "in_chans": 3,
        "depths": (6,),
        "embed_dim": 384,
    }
    defaults = {"num_heads": 6}
    model = model_factory(QuadMoETransformer, params, defaults, **kwargs)
    return model


@register_model
def quadmoe_tiny_2s_patch16_128(**kwargs):
    params = {
        "img_size": 128,
        "patch_size": 16,
        "in_chans": 3,
        "depths": (3, 3),
        "embed_dim": 384,
    }
    defaults = {"num_heads": 6}
    model = model_factory(QuadMoETransformer, params, defaults, **kwargs)
    return model


@register_model
def quadmoe_tiny_3s_patch16_128(**kwargs):
    params = {
        "img_size": 128,
        "patch_size": 16,
        "in_chans": 3,
        "depths": (2, 2, 2),
        "embed_dim": 384,
    }
    defaults = {"num_heads": 6}
    model = model_factory(QuadMoETransformer, params, defaults, **kwargs)
    return model
