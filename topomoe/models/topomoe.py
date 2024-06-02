"""
The primate visual cortex has a topographic branching organization. Early layers are
characterized by retinotopic mapping, narrow receptive fields, and shared
orientation-tuned filters. As you proceed through the hierarchy, receptive fields widen
and tuning becomes more specialized. The shared trunk splits into independent streams
(e.g. dorsal, ventral), which ultimately terminate in multiple highly specialized areas
with near global receptive field.

The topographic MoE transformer aims to model this high-level architecture. The model
consists of a series of stages, each with its own independent topographic
representational map. For example, for an input patch grid of 8x8, the first stage might
have a map of size 8x8 matching the input, the second stage could have a larger map of
size 16x16, and the final stage could have a map size of 32x32.

Each stage starts by pooling the output of the prior stage. The pooling strategy closely
follows the routing approach from Soft MoE. The pooling weights, shape (seq_len,
in_seq_len) are computed by taking dot products between two position embeddings: the one
for the prior stage (or patch input for the first stage) and the one for the current
stage. Then we apply softmax as usual. This way, tokens pool from input tokens at
similar positions, where position is defined by the learned position embeddings.
Basically, it is associative pooling based on position. Effectively then, the position
embeddings capture the implicit topography of each map. Furthermore, we can constrain
the implicit topography by applying eg wiring cost regularization.

After pooling, we apply a series of topographic MoE transformer blocks. The
representation map is partitioned into areas and each area is assigned a unique expert.
The mapping of experts to positions uses the same topographic mapping approach as above.
Each expert has a learned "position" in the position embedding space. And we associate
experts to tokens based on the correlation between the expert embedding and
the token position embeddings.

In the attention module, the query weights are independent for each expert, so that each
expert can learn to select for unique information. But the key/value/projection weights
are shared. This is done for parsimony, since I don't see a clear reason to decouple the
key/value/proj weights. It also has some similarity with modern LLM architectures that
use more query heads than key/value heads.

In addition, for the first block in the stage, queries come from the pooled input and
keys/values come from the full input. In this way, the first attention block effectively
does cross attention from the previous stage output. This attention can also be seen as
dynamic content-dependent pooling, in constrast to static position based pooling.

The Mlp is a standard Mlp except for the independent weights for each expert. The per expert Mlp
hidden dimension can be divided by the number of experts (preserving parameters,
reducing flops) or left alone (increasing parameters, preserving flops).

With this architecture, we will see what kinds of topography the pooling layers learn.
For example, whether it arranges positions retinotopically, how it allocates its
available token capacity to different regions of the image, how it manages the
transition from retinotopic mapping to semantic mapping. We will also see how experts
are arranged in the representation maps, how the arrangement relates to the pooling
topography (e.g. does each expert get a view of the full input, or perhaps do you get
experts that specialize for the center vs the periphery). Together, by modeling the key
branching topographic structure of the visual cortex we hope to better understand how
topography emerges, why it emerges, what kinds of specialized computations emerge, and
what specialization is good for.
"""

import logging
from functools import partial
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from timm.layers import PatchEmbed, trunc_normal_
from timm.layers.helpers import to_2tuple, to_3tuple
from torch import nn

from columnformers.utils import filter_kwargs

from .registry import register_model

State = Dict[str, torch.Tensor]
Layer = Callable[..., nn.Module]


class TopoMaps(nn.Module):
    """
    Topographic mapping of token positions to "slots".

    Slots can be the tokens in a new grid (when pooling) or experts for topographic
    expert mapping.

    If token wise, do softmax over tokens, output is (slots, tokens). Otherwise softmax
    over slots, output is (tokens, slots).

    Closely follows the Soft-MoE router, but uses fixed position embeddings rather than
    dynamic input embeddings to route based on.
    """

    def __init__(
        self,
        slots: int,
        pos_embed: nn.Parameter,
        token_wise: bool = False,
    ):
        super().__init__()
        self.slots = slots
        self.token_wise = token_wise
        # save a reference to the shared position embedding
        self.pos_embed = pos_embed
        self.weight = nn.Parameter(torch.empty((slots, pos_embed.size(-1))))
        self.scale = nn.Parameter(torch.empty(()))
        self.reset_parameters()

    def reset_parameters(self):
        trunc_normal_(self.weight, std=0.02)
        nn.init.ones_(self.scale)

    def forward(self) -> torch.Tensor:
        # l2 normalize weights and embeddings following soft moe. apparently this helps
        # with training stability at large scale and doesn't hurt at small scale.
        pos_embed = F.normalize(self.pos_embed, dim=-1)
        weight = self.scale * F.normalize(self.weight, dim=-1)
        if self.token_wise:
            # mapping slots <- tokens, shape (slots, tokens)
            # like soft moe dispatch
            coef = weight @ pos_embed.t()
        else:
            # mapping tokens <- slots, shape (tokens, slots)
            # like soft moe combine
            coef = pos_embed @ weight.t()
        coef = coef.softmax(dim=-1)
        return coef

    def no_weight_decay(self) -> List[str]:
        return ["weight", "scale"]

    def extra_repr(self) -> str:
        return f"{self.slots}, token_wise={self.token_wise}"


class TopoLinear(nn.Module):
    """
    Topographic mixture of linear layers. The linear weights for each token in the
    sequence are computed as a convex combination of the weights in the mixture. The
    combination probabilities are given by a set of topographic maps.

    In the language of Soft MoE, expert weights are "combined" into each token.
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
        self.weight = nn.Parameter(torch.empty((out_features, in_features, maps.slots)))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, maps.slots))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        trunc_normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # input: (batch, seq_len, in_features)
        maps = self.maps()  # (seq_len, experts)
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
        self, x: torch.Tensor, pooled: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, State]:
        x = self.norm1(x)
        pooled = self.norm1(pooled) if pooled is not None else x

        # attention with queries from pooled input and keys/values from full input
        # this way attention acts like adaptive dynamic pooling
        x, attn_state = self.attn(pooled, x)
        # residual on top of pooled input
        # the query/residual is the main path; all other information is pulled in
        # selectively via attention
        x = x + pooled

        # standard mlp, but independent weights per block
        x = x + self.mlp(self.norm2(x))

        state = {**attn_state, "features": x}
        return x, state


class Stage(nn.Module):
    def __init__(
        self,
        in_pos_embed: nn.Parameter,
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
    ):
        super().__init__()
        in_seq_len = in_pos_embed.size(-2)
        assert (
            seq_len is None or seq_len == in_seq_len or pool
        ), "changing seq_len requires pool=true"
        self.seq_len = seq_len or in_seq_len

        if pool:
            # Pool from input tokens to current token space using a topographic map.
            # Note that nothing says the sequence lengths between stages need to be the
            # same size. By letting the sizes be different, we can model the different
            # area sizes in visual cortex. Similar to how in TDANN, the size of areas
            # increases as you go up the hierarchy. Also similar to how in Soft MoE they
            # get best performance when there are more slots than tokens.
            self.pool = TopoMaps(self.seq_len, in_pos_embed, token_wise=True)
            # Use pooling weights as new position embedding. This position embedding
            # goes into the expert assignment maps, as well as subsequent stages.
            # Doing it this way should hopefully help connect the maps together better.
            self.pos_embed = self.pool.weight
        else:
            self.register_module("pool", None)
            self.pos_embed = in_pos_embed

        if num_experts > 1:
            # Topographic mapping of experts to tokens. Shared across all blocks.
            self.maps = TopoMaps(num_experts, self.pos_embed)
        else:
            self.register_module("maps", None)

        self.blocks = nn.ModuleList(
            Block(
                # do pooling only on first block of the stage
                pool=self.pool if ii == 0 else None,
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, State]:
        # Pooling tokens to tokens, much like soft moe pooling tokens to slots.
        # Importantly though, without some special init or regularization, the pooling
        # can arbitrarily shuffle tokens. We may be able to use some wiring cost to
        # promote regularity. But even without that, we can still visualize the implicit
        # spatial topography by doing a 2D embedding of the pooling position embeddings.
        if self.pool:
            pool = self.pool()
            pooled = pool @ x
        else:
            # dummy pool for state
            pool = torch.eye(self.seq_len, dtype=x.dtype, device=x.device)
            pool = pooled = None

        state = {}
        for ii, block in enumerate(self.blocks):
            x, block_state = block(x, pooled=pooled)
            state.update({f"block.{ii}.{k}": v for k, v in block_state.items()})

        # add position embedding, pooling, and expert maps to state
        state["pos_embed"] = self.pos_embed.clone()
        state["pool"] = pool
        state["maps"] = self.maps() if self.maps else None
        return x, state


class TopoMoETransformer(nn.Module):
    def __init__(
        self,
        img_size: int = 128,
        patch_size: int = 16,
        in_chans: int = 3,
        depths: Tuple[int, ...] = (4, 4, 4),
        num_experts: Tuple[int, ...] = (1, 4, 16),
        seq_len: Optional[Union[int, Tuple[int, ...]]] = None,
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

        self.pos_embed = pos_embed = nn.Parameter(torch.empty(num_patches, embed_dim))
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        stages = []
        seq_len = _to_list(seq_len, len(depths))
        mlp_ratio = _to_list(mlp_ratio, len(depths))

        for ii, (depth, experts, seq, ratio) in enumerate(
            zip(depths, num_experts, seq_len, mlp_ratio)
        ):
            if mlp_conserve:
                ratio = ratio / num_experts

            # do pooling for stages after first or if initial sequence length doesn't
            # match the patch grid.
            pool = ii > 0 or seq != seq_len

            stage = Stage(
                in_pos_embed=pos_embed,
                pool=pool,
                seq_len=seq,
                depth=depth,
                num_experts=experts,
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=ratio,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop_rate,
                proj_drop=proj_drop_rate,
                act_layer=act_layer,
            )

            # update pos embedding for next stage to the one from the previous stage
            pos_embed = stage.pos_embed
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
        self.apply(_init_weights)

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, State]:
        x = self.patch_embed(x)
        x = x + self.pos_embed

        state = {}
        for ii, stage in enumerate(self.stages):
            x, stage_state = stage(x)
            state.update({f"stages.{ii}.{k}": v for k, v in stage_state.items()})

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
    elif len(x) != length:
        raise ValueError(f"Length of x {len(x)} doesn't match target length {length}")
    return x


def _create_model(
    cls: type, params: Dict[str, Any], defaults: Dict[str, Any], **kwargs
):
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    kwargs, extra_args = filter_kwargs(cls, kwargs)
    if extra_args:
        logging.warning("Extra kwargs to %s: %s", cls.__name__, extra_args)
    kwargs = {**defaults, **kwargs}
    model = cls(**params, **kwargs)
    return model


@register_model
def topomoe_tiny_2s_patch16_128(**kwargs):
    params = {
        "img_size": 128,
        "patch_size": 16,
        "in_chans": 3,
        "depths": (3, 3),
        "seq_len": 64,
        "embed_dim": 384,
    }
    defaults = {
        "num_experts": (1, 4),
        "num_heads": 6,
    }
    model = _create_model(TopoMoETransformer, params, defaults, **kwargs)
    return model


@register_model
def topomoe_tiny_3s_patch16_128(**kwargs):
    params = {
        "img_size": 128,
        "patch_size": 16,
        "in_chans": 3,
        "depths": (2, 2, 2),
        "seq_len": 64,
        "embed_dim": 384,
    }
    defaults = {
        "num_experts": (1, 4, 16),
        "num_heads": 6,
    }
    model = _create_model(TopoMoETransformer, params, defaults, **kwargs)
    return model
