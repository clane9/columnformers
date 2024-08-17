"""
Topographic mixture of experts vision transformer.
"""

from functools import partial
from typing import List, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from timm.layers import PatchEmbed, trunc_normal_
from torch import nn

from . import wiring
from .common import Attention, Layer, Mlp, State, init_weights, model_factory, to_list
from .registry import register_model


class SoftPool(nn.Module):
    """
    Pooling/branching/routing of tokens between stages following Soft MoE router.

    The stage grid embedding is used as a set of learned queries, one per grid position.
    The keys can be either the input tokens themselves (dynamic, just like Soft MoE) or
    the grid embedding for the previous stage (static).
    """

    def __init__(
        self,
        slots: int,
        dim: int,
        static: bool = False,
        in_grid_embed: Optional[nn.Parameter] = None,
    ):
        super().__init__()
        assert (
            not static or in_grid_embed is not None
        ), "in_grid_embed required for static pooling"
        self.slots = slots
        self.static = static
        self.grid_embed = nn.Parameter(torch.empty(slots, dim))
        if static:
            self.in_grid_embed = in_grid_embed
        else:
            self.register_parameter("in_grid_embed", None)
        self.scale = nn.Parameter(torch.empty(()))
        self.reset_parameters()

    def reset_parameters(self):
        trunc_normal_(self.grid_embed, std=0.02)
        nn.init.ones_(self.scale)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, State]:
        key = self.in_grid_embed if self.static else x
        # l2 normalization with learned scale following soft moe.
        key = F.normalize(key, dim=-1)
        query = self.scale * F.normalize(self.grid_embed, dim=-1)
        pool = (query @ key.transpose(-1, -2)).softmax(dim=-1)
        x = pool @ x
        return x, {"pool": pool, "grid_embed": query}

    def no_weight_decay(self) -> List[str]:
        return ["scale"]

    def extra_repr(self) -> str:
        return f"{self.slots}, static={self.static}"


class TopoMaps(nn.Module):
    """
    Mapping of experts to token grid positions following Soft MoE router.
    """

    def __init__(self, experts: int, grid_embed: nn.Parameter):
        super().__init__()
        self.experts = experts
        self.grid_embed = grid_embed
        self.expert_embed = nn.Parameter(torch.empty((experts, grid_embed.size(-1))))
        self.scale = nn.Parameter(torch.empty(()))
        self.reset_parameters()

    def reset_parameters(self):
        trunc_normal_(self.expert_embed, std=0.02)
        nn.init.ones_(self.scale)

    def forward(self) -> Tuple[torch.Tensor, State]:
        # l2 normalization with learned scale following soft moe.
        query = F.normalize(self.grid_embed, dim=-1)
        key = self.scale * F.normalize(self.expert_embed, dim=-1)
        maps = (query @ key.t()).softmax(dim=-1)
        state = {"maps": maps, "expert_embed": key}
        return maps, state

    def no_weight_decay(self) -> List[str]:
        return ["scale"]

    def extra_repr(self) -> str:
        return f"{self.experts}"


class TopoLinear(nn.Module):
    """
    Topographic mixture of linear layers. The linear weights for each token in the
    sequence are computed as a convex combination of the weights in the mixture. The
    combination probabilities are given by a set of topographic expert maps.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        maps: TopoMaps,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.maps = maps
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features, maps.experts))
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, maps.experts))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        trunc_normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # input: (batch, seq_len, in_features)
        maps, _ = self.maps()  # (seq_len, experts)
        # Nb, this implementation for some reason uses significantly fewer flops
        # compared to equivalent alternatives (e.g. einsum, batch matmul) for some
        # reason.
        weight = (maps @ self.weight.transpose(1, 2)).transpose(0, 1)
        if self.bias is not None:
            bias = maps @ self.bias.t()
        output = torch.einsum("bnc,ndc->bnd", input, weight)
        if self.bias is not None:
            output = output + bias
        return output

    def extra_repr(self) -> str:
        return f"{self.in_features}, {self.out_features}, bias={self.bias is not None}"


class Block(nn.Module):
    def __init__(
        self,
        maps: Optional[TopoMaps] = None,
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
        linear_layer = partial(TopoLinear, maps=maps) if maps else nn.Linear
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            # only q is decoupled across blocks, k, v, proj are shared
            # interestingly this has some similarity with how modern llms have more
            # query heads than keys
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
        # this way attention acts like adaptive dynamic pooling
        # should we have separate norm for the context?
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
        in_grid_embed: nn.Parameter,
        pool: bool = False,
        seq_len: Optional[int] = None,
        depth: int = 1,
        num_experts: int = 16,
        dim: int = 384,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: Union[bool, Tuple[bool, bool, bool]] = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        act_layer: Layer = nn.GELU,
        static_pool: bool = False,
        in_wiring_cost: Optional[nn.Module] = None,
        wiring_cost: Optional[nn.Module] = None,
    ):
        super().__init__()
        in_seq_len = in_grid_embed.size(-2)
        assert (
            seq_len is None or seq_len == in_seq_len or pool
        ), "changing seq_len requires pool=true"
        self.seq_len = seq_len or in_seq_len

        if pool:
            # Pool from input tokens to current token space.
            # Note that nothing says the sequence lengths between stages need to be the
            # same size. By letting the sizes be different, we can model the different
            # area sizes in visual cortex. Similar to TDANN. Also similar to how Soft
            # MoE gets best performance with a lot of slots and a lot of experts.
            self.pool = SoftPool(
                self.seq_len,
                dim,
                static=static_pool,
                in_grid_embed=in_grid_embed,
            )
            self.grid_embed = self.pool.grid_embed
        else:
            self.register_module("pool", None)
            self.grid_embed = in_grid_embed

        if num_experts > 1:
            # Topographic mapping of experts to tokens. Shared across all blocks.
            self.maps = TopoMaps(num_experts, self.grid_embed)
        else:
            self.register_module("maps", None)

        self.blocks = nn.ModuleList(
            Block(
                maps=self.maps,
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

        self.register_module("in_wiring_cost", in_wiring_cost)
        self.register_module("wiring_cost", wiring_cost)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, State, State]:
        # Pooling tokens to tokens, like Soft MoE pooling tokens to slots. Importantly
        # though, without some special init or regularization, the pooling can
        # arbitrarily shuffle tokens. We can use some wiring cost to promote spatial
        # regularity. But even without that, we can still visualize the implicit spatial
        # topography by doing a 2D embedding of the pooling position embeddings.
        if self.pool:
            pooled, pool_state = self.pool(x)
            x, context = pooled, x
            pool = pool_state["pool"]
        else:
            pool = context = None
            pool_state = {}

        state = {}
        for ii, block in enumerate(self.blocks):
            x, block_state = block(x, context)
            state.update({f"blocks.{ii}.{k}": v for k, v in block_state.items()})
            context = None

        losses = {}
        if self.in_wiring_cost is not None:
            if pool is not None:
                losses["pool.wiring_cost"] = self.in_wiring_cost(pool)

            # input attention cross-attends over the input tokens, has shape
            # (seq_len, in_seq_len)
            in_attn = state["blocks.0.attn"]
            losses["blocks.0.attn.wiring_cost"] = self.in_wiring_cost(in_attn)

        if self.wiring_cost is not None:
            for ii in range(1, len(self.blocks)):
                attn = state[f"blocks.{ii}.attn"]
                losses[f"blocks.{ii}.attn.wiring_cost"] = self.wiring_cost(attn)

        state.update(pool_state)
        if self.maps is not None:
            _, maps_state = self.maps()
            state.update(maps_state)

        return x, losses, state


class TopoMoETransformer(nn.Module):
    def __init__(
        self,
        img_size: int = 128,
        patch_size: int = 16,
        in_chans: int = 3,
        depths: Tuple[int, ...] = (4, 4, 4),
        widths: Optional[Tuple[int, ...]] = None,
        stage_pools: Optional[Tuple[bool, ...]] = None,
        num_experts: Tuple[int, ...] = (1, 4, 16),
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
        static_pool: bool = False,
        wiring_lambd: float = 0.0,
        wiring_sigma: float = 2.0,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.global_pool = global_pool
        num_patches = (img_size // patch_size) ** 2
        widths = widths or img_size // patch_size

        widths = to_list(widths, len(depths))
        mlp_ratio = to_list(mlp_ratio, len(depths))
        stage_pools = stage_pools or (False,) + len(depths) * (True,)

        if wiring_lambd > 0:
            geo_embeds = wiring.geo_embedding(widths)
            wiring_cost_layer = partial(
                wiring.CrossEntropyWiringCost, lambd=wiring_lambd, sigma=wiring_sigma
            )
        else:
            geo_embeds = wiring_cost_layer = None

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.pos_embed = nn.Parameter(torch.empty(num_patches, embed_dim))

        stages = []
        in_grid_embed = self.pos_embed
        for ii, (depth, experts, width, pool, ratio) in enumerate(
            zip(depths, num_experts, widths, stage_pools, mlp_ratio)
        ):
            if mlp_conserve:
                ratio = ratio / experts

            wiring_cost = in_wiring_cost = None
            if wiring_lambd > 0:
                # no input wiring cost from patches to first stage
                # this could be used to learn the initial mapping from the retina
                if ii > 0:
                    in_wiring_cost = wiring_cost_layer(
                        geo_embeds[ii], geo_embeds[ii - 1]
                    )
                wiring_cost = wiring_cost_layer(geo_embeds[ii])

            stage = Stage(
                in_grid_embed=in_grid_embed,
                pool=pool,
                seq_len=width * width,
                depth=depth,
                num_experts=experts,
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=ratio,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop_rate,
                proj_drop=proj_drop_rate,
                act_layer=act_layer,
                static_pool=static_pool,
                in_wiring_cost=in_wiring_cost,
                wiring_cost=wiring_cost,
            )

            # update grid embedding for next stage to the one from the previous stage
            in_grid_embed = stage.grid_embed
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
            x, stage_losses, stage_state = stage(x)
            losses.update({f"stages.{ii}.{k}": v for k, v in stage_losses.items()})
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
def topomoe_tiny_1s_patch16_128(**kwargs):
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
    model = model_factory(TopoMoETransformer, params, defaults, **kwargs)
    return model


@register_model
def topomoe_tiny_2s_patch16_128(**kwargs):
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
    model = model_factory(TopoMoETransformer, params, defaults, **kwargs)
    return model


@register_model
def topomoe_tiny_3s_patch16_128(**kwargs):
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
    model = model_factory(TopoMoETransformer, params, defaults, **kwargs)
    return model
