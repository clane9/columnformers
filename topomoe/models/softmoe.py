"""
References:
    https://arxiv.org/abs/2308.00951
    https://github.com/google-research/vmoe/blob/main/vmoe/projects/soft_moe/router.py
    https://github.com/google-research/vmoe/blob/main/vmoe/nn/vit_moe.py
"""

from functools import partial
from typing import List, Literal, Tuple, Union

import torch
import torch.nn.functional as F
from einops import rearrange
from timm.layers import PatchEmbed, trunc_normal_
from torch import nn

from .common import Attention, Layer, Mlp, State, init_weights, model_factory, to_list
from .registry import register_model


class SoftRouter(nn.Module):
    """
    Compute soft routing logits.

    Slots are total slots, i.e. experts * slots_per_expert.
    """

    def __init__(self, slots: int, dim: int, noise_std: float = 0.0):
        super().__init__()
        self.slots = slots
        self.dim = dim
        self.noise_std = noise_std
        self.weight = nn.Parameter(torch.empty((slots, dim)))
        self.scale = nn.Parameter(torch.empty(()))
        self.reset_parameters()

    def reset_parameters(self):
        trunc_normal_(self.weight, std=0.02)
        nn.init.ones_(self.scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, L, C)
        # logits: (N, S, L)
        x = F.normalize(x, dim=-1)
        weight = self.scale * F.normalize(self.weight, dim=-1)
        logits = weight @ x.transpose(1, 2)

        if self.training and self.noise_std > 0:
            x = x + self.noise_std * torch.randn_like(x)
        return logits

    def no_weight_decay(self) -> List[str]:
        return ["weight", "scale"]

    def extra_repr(self) -> str:
        return f"{self.slots}, {self.dim}, noise_std={self.noise_std}"


class ExpertLinear(nn.Module):
    """
    Independent linear layer for each expert. Input shape should be (batch, slots, dim).
    """

    def __init__(
        self,
        num_experts: int,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.empty((num_experts, out_features, in_features))
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(num_experts, out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        trunc_normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        N, S, C = input.shape
        slots_per_expert = S // self.num_experts
        input = rearrange(
            input, "n (e s) c -> e (n s) c", e=self.num_experts, s=slots_per_expert
        )
        output = input @ self.weight.transpose(1, 2)
        output = rearrange(output, "e (n s) c -> n (e s) c", n=N, s=slots_per_expert)
        return output

    def extra_repr(self) -> str:
        return (
            f"{self.num_experts}, {self.in_features}, {self.out_features}, "
            f"bias={self.bias is not None}"
        )


class Block(nn.Module):
    def __init__(
        self,
        num_experts: int = 16,
        slots_per_expert: int = 16,
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
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            linear_layer=nn.Linear,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.norm2 = nn.LayerNorm(dim)

        if num_experts > 1:
            self.router = SoftRouter(num_experts * slots_per_expert, dim)
            linear_layer = partial(ExpertLinear, num_experts)
        else:
            self.register_module("router", None)
            linear_layer = nn.Linear

        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            linear_layer=linear_layer,
            act_layer=act_layer,
            drop=proj_drop,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, State]:
        attend, attn_state = self.attn(self.norm1(x))
        x = x + attend

        # softmoe mlp path
        # logits: (N, S, L)
        if self.router is not None:
            z = self.norm2(x)
            logits = self.router(z)
            dispatch = logits.softmax(dim=-1)
            combine = logits.transpose(1, 2).softmax(dim=-1)

            z = dispatch @ z
            z = self.mlp(z)
            z = combine @ z
            x = x + z
        else:
            x = x + self.mlp(self.norm2(x))
            dispatch = None

        state = {**attn_state, "features": x, "dispatch": dispatch}
        return x, state


class Stage(nn.Module):
    """
    Nb, this extra layer of abstraction is not necessary for soft moe, but included for
    consistency with other models.
    """

    def __init__(
        self,
        depth: int = 1,
        num_experts: int = 16,
        slots_per_expert: int = 16,
        dim: int = 384,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: Union[bool, Tuple[bool, bool, bool]] = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        act_layer: Layer = nn.GELU,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            Block(
                num_experts=num_experts,
                slots_per_expert=slots_per_expert,
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
        state = {}
        for ii, block in enumerate(self.blocks):
            x, block_state = block(x)
            state.update({f"blocks.{ii}.{k}": v for k, v in block_state.items()})
        return x, state


class SoftMoETransformer(nn.Module):
    def __init__(
        self,
        img_size: int = 128,
        patch_size: int = 16,
        in_chans: int = 3,
        depths: Tuple[int, ...] = (4, 4, 4),
        num_experts: Tuple[int, ...] = (1, 4, 16),
        slots_per_token: Union[int, List[int]] = 1,
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

        slots_per_token = to_list(slots_per_token, len(num_experts))
        slots_per_expert = [
            (slots * num_patches) // experts
            for slots, experts in zip(slots_per_token, num_experts)
        ]
        assert min(slots_per_expert) > 0, "can't have more experts than slots"

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.pos_embed = nn.Parameter(torch.empty(1, num_patches, embed_dim))

        stages = []
        mlp_ratio = to_list(mlp_ratio, len(depths))

        for ii, (depth, experts, slots, ratio) in enumerate(
            zip(depths, num_experts, slots_per_expert, mlp_ratio)
        ):
            if mlp_conserve:
                ratio = ratio / experts

            stage = Stage(
                depth=depth,
                num_experts=experts,
                slots_per_expert=slots,
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
def softmoe_tiny_1s_patch16_128(**kwargs):
    params = {
        "img_size": 128,
        "patch_size": 16,
        "in_chans": 3,
        "depths": (6,),
        "widths": 8,
        "embed_dim": 384,
    }
    defaults = {
        "num_experts": (1,),
        "num_heads": 6,
    }
    model = model_factory(SoftMoETransformer, params, defaults, **kwargs)
    return model


@register_model
def softmoe_tiny_2s_patch16_128(**kwargs):
    params = {
        "img_size": 128,
        "patch_size": 16,
        "in_chans": 3,
        "depths": (3, 3),
        "widths": 8,
        "embed_dim": 384,
    }
    defaults = {
        "num_experts": (1, 4),
        "num_heads": 6,
    }
    model = model_factory(SoftMoETransformer, params, defaults, **kwargs)
    return model


@register_model
def softmoe_tiny_3s_patch16_128(**kwargs):
    params = {
        "img_size": 128,
        "patch_size": 16,
        "in_chans": 3,
        "depths": (2, 2, 2),
        "widths": 8,
        "embed_dim": 384,
    }
    defaults = {
        "num_experts": (1, 4, 16),
        "num_heads": 6,
    }
    model = model_factory(SoftMoETransformer, params, defaults, **kwargs)
    return model
