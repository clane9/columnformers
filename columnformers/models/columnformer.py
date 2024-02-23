from functools import partial
from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from timm.layers import trunc_normal_
from timm.layers.helpers import to_2tuple, to_3tuple
from torch import nn

from .layers import Layer, MixtureLinear, UntiedLayerNorm, UntiedLinear


class Attention(nn.Module):
    """
    Multi-head attention with options for:

        - untied weights across the sequence
        - learned attention bias (optionally per head)
        - low qk dim
        - no value/projection following (He & Hofmann, 2023)

    References:
        timm vision_transformer
        https://arxiv.org/abs/2311.01906
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        untied: bool = False,
        seq_len: Optional[int] = None,
        bias: bool = False,
        head_bias: bool = False,
        qkv_bias: Union[bool, Tuple[bool, bool, bool]] = False,
        qk_head_dim: Optional[int] = None,
        no_vp: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        assert not no_vp or proj_drop == 0, "no_vp incompatible with proj_drop"
        assert not (bias or untied) or seq_len, "seq_len required for bias or untied"
        assert not head_bias or bias, "head_bias requires bias"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_head_dim = qk_head_dim or self.head_dim
        self.scale = self.qk_head_dim**-0.5
        qkv_biases = to_3tuple(qkv_bias)
        linear_layer = partial(UntiedLinear, seq_len) if untied else nn.Linear

        if bias:
            self.bias = nn.Parameter(torch.zeros(seq_len, seq_len))
        else:
            self.register_parameter("bias", None)
        if head_bias:
            self.head_bias = nn.Parameter(torch.zeros(num_heads, seq_len, seq_len))
        else:
            self.register_parameter("head_bias", None)

        self.q = linear_layer(dim, num_heads * self.qk_head_dim, bias=qkv_biases[0])
        self.k = linear_layer(dim, num_heads * self.qk_head_dim, bias=qkv_biases[1])
        if no_vp:
            self.v = nn.Identity()
            self.proj = nn.Identity()
        else:
            self.v = linear_layer(dim, dim, bias=qkv_biases[2])
            self.proj = linear_layer(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        nh, qkd, d = self.num_heads, self.qk_head_dim, self.head_dim
        q = self.q(x).reshape(B, N, nh, qkd).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, nh, qkd).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, nh, d).permute(0, 2, 1, 3)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if self.bias is not None:
            attn = attn + self.bias
        if self.head_bias is not None:
            attn = attn + self.head_bias
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

    def init_bias(self, bias: torch.Tensor):
        assert self.bias is not None
        self.bias.data.copy_(bias)

    def extra_repr(self) -> str:
        return f"bias={self.bias is not None}, head_bias={self.head_bias is not None}"


class Selection(nn.Module):
    """
    Multi-head static feature-based "selection".

    Treats inputs as keys/values and uses a static as opposed to dyanmic query.
    """

    def __init__(
        self,
        seq_len: int,
        dim: int,
        num_heads: int = 8,
        bias: bool = True,
        head_bias: bool = True,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        assert seq_len, "seq_len required for selection"
        assert not head_bias or bias, "head_bias requires bias"

        self.dim = dim
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.query = nn.Parameter(torch.empty(num_heads, seq_len, self.head_dim))
        if bias:
            self.bias = nn.Parameter(torch.empty(seq_len, seq_len))
        else:
            self.register_parameter("bias", None)
        if head_bias:
            self.head_bias = nn.Parameter(torch.empty(num_heads, seq_len, seq_len))
        else:
            self.register_parameter("head_bias", None)
        self.attn_drop = nn.Dropout(attn_drop)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        trunc_normal_(self.query, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        if self.head_bias is not None:
            nn.init.zeros_(self.head_bias)

    def forward(self, x: torch.Tensor):
        B, N, C = x.shape
        x = x.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        q = self.scale * self.query
        attn = q @ x.transpose(-2, -1)
        if self.bias is not None:
            attn = attn + self.bias
        if self.head_bias is not None:
            attn = attn + self.head_bias
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ x
        x = x.transpose(1, 2).reshape(B, N, C)
        return x, attn

    def init_bias(self, bias: torch.Tensor):
        self.bias.data.copy_(bias)

    def extra_repr(self) -> str:
        return (
            f"{self.seq_len}, {self.dim}, num_heads={self.num_heads}, "
            f"bias={self.bias is not None}, head_bias={self.head_bias is not None}"
        )


class Mixing(nn.Module):
    """
    Multi-head static feature "mixing" similar to ConvNext and MLP-Mixer.
    """

    def __init__(
        self,
        seq_len: int,
        dim: int,
        num_heads: int = 8,
        softmax: bool = True,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        assert seq_len, "seq_len required for selection"

        self.dim = dim
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.softmax = softmax

        self.bias = nn.Parameter(torch.empty(seq_len, seq_len))
        self.head_bias = nn.Parameter(torch.empty(num_heads, seq_len, seq_len))
        self.attn_drop = nn.Dropout(attn_drop)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        trunc_normal_(self.bias, std=0.02)
        nn.init.zeros_(self.head_bias)

    def forward(self, x: torch.Tensor):
        B, N, C = x.shape
        x = x.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn = self.bias + self.head_bias
        if self.softmax:
            attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ x
        x = x.transpose(1, 2).reshape(B, N, C)

        # expand to be consistent with other attention mechanisms
        attn = attn.unsqueeze(0)
        return x, attn

    def init_bias(self, bias: torch.Tensor):
        self.bias.data.copy_(bias)

    def extra_repr(self) -> str:
        return (
            f"{self.seq_len}, {self.dim}, num_heads={self.num_heads}, "
            f"softmax={self.softmax}"
        )


class Mlp(nn.Module):
    """
    Mlp module with option for untied weights across the sequence.

    References:
        timm Mlp
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        seq_len: Optional[int] = None,
        untied: bool = False,
        act_layer: Optional[Layer] = nn.GELU,
        bias: Union[bool, Tuple[bool, bool]] = True,
        drop: Union[float, Tuple[float, float]] = 0.0,
    ):
        super().__init__()
        assert not untied or seq_len, "seq_len required for untied"

        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        biases = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        if untied:
            linear_layer = partial(UntiedLinear, seq_len)
        else:
            linear_layer = nn.Linear
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


class MixtureMlp(nn.Module):
    """
    Mixture of MLP experts. The MLP weights for each token in the sequence are computed
    as a linear combination of the individual expert weights. A bit similar to model
    soups. The linear combination coefficients are static per token in the sequence.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        seq_len: Optional[int] = None,
        rank: int = 16,
        act_layer: Optional[Layer] = nn.GELU,
        bias: Union[bool, Tuple[bool, bool]] = True,
        drop: Union[float, Tuple[float, float]] = 0.0,
    ):
        super().__init__()
        assert seq_len is not None, "seq_len required for MixtureMLP"

        self.rank = rank
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        biases = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        act_layer = nn.Identity if act_layer is None else act_layer

        self.fc1 = MixtureLinear(
            in_features, hidden_features, rank=rank, bias=biases[0]
        )
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = MixtureLinear(
            hidden_features, out_features, rank=rank, bias=biases[1]
        )
        self.drop2 = nn.Dropout(drop_probs[1])

        self.coef = nn.Parameter(torch.empty(seq_len, rank))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = (0.02 * self.rank**-0.5) ** 0.5
        trunc_normal_(self.coef, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: what about softmax coefficients?
        x = self.fc1(x, self.coef)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x, self.coef)
        x = self.drop2(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        untied: Union[bool, Tuple[bool, bool, bool]] = False,
        seq_len: Optional[int] = None,
        mlp_rank: Optional[int] = None,
        attn_mode: Literal["classic", "selection", "mixing", "linmixing"] = "classic",
        skip_attn: bool = True,
        attn_bias: bool = False,
        attn_head_bias: bool = False,
        qkv_bias: Union[bool, Tuple[bool, bool, bool]] = False,
        qk_head_dim: Optional[int] = None,
        no_vp: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        act_layer: Layer = nn.GELU,
    ):
        super().__init__()
        self.skip_attn = skip_attn
        untied_norm, untied_attn, untied_mlp = to_3tuple(untied)
        norm_layer = partial(UntiedLayerNorm, seq_len) if untied_norm else nn.LayerNorm

        self.norm1 = norm_layer(dim)
        if attn_mode == "classic":
            self.attn = Attention(
                dim=dim,
                num_heads=num_heads,
                untied=untied_attn,
                seq_len=seq_len,
                bias=attn_bias,
                head_bias=attn_head_bias,
                qkv_bias=qkv_bias,
                qk_head_dim=qk_head_dim,
                no_vp=no_vp,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
            )
        elif attn_mode == "selection":
            self.attn = Selection(
                seq_len=seq_len,
                dim=dim,
                num_heads=num_heads,
                bias=attn_bias,
                head_bias=attn_head_bias,
                attn_drop=attn_drop,
            )
        elif attn_mode == "mixing":
            self.attn = Mixing(
                seq_len=seq_len,
                dim=dim,
                num_heads=num_heads,
                attn_drop=attn_drop,
            )
        elif attn_mode == "linmixing":
            self.attn = Mixing(
                seq_len=seq_len,
                dim=dim,
                num_heads=num_heads,
                softmax=False,
                attn_drop=attn_drop,
            )
        else:
            raise ValueError(f"Unknown attn_mode {attn_mode}")
        self.norm2 = norm_layer(dim)
        if mlp_rank is None:
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=int(dim * mlp_ratio),
                seq_len=seq_len,
                untied=untied_mlp,
                act_layer=act_layer,
                drop=proj_drop,
            )
        else:
            self.mlp = MixtureMlp(
                in_features=dim,
                hidden_features=int(dim * mlp_ratio),
                seq_len=seq_len,
                rank=mlp_rank,
                act_layer=act_layer,
                drop=proj_drop,
            )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pooled, attn = self.attn(self.norm1(x))
        x = x + pooled if self.skip_attn else pooled
        x = x + self.mlp(self.norm2(x))
        return x, attn

    def extra_repr(self) -> str:
        return f"skip_attn={self.skip_attn}"


class Columnformer(nn.Module):
    geometry: Optional[torch.Tensor]

    def __init__(
        self,
        embed_dim: int = 384,
        depth: int = 12,
        recurrent: bool = False,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        untied: Union[bool, Tuple[bool, bool, bool]] = False,
        seq_len: Optional[int] = None,
        mlp_rank: Optional[Union[int, List[int]]] = None,
        attn_mode: Literal["classic", "selection", "mixing", "linmixing"] = "classic",
        skip_attn: bool = True,
        attn_bias: bool = False,
        attn_head_bias: bool = False,
        qkv_bias: Union[bool, Tuple[bool, bool, bool]] = True,
        qk_head_dim: Optional[int] = None,
        no_vp: bool = False,
        attn_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        act_layer: Layer = nn.GELU,
        geometry: Optional[torch.Tensor] = None,
        init_local_attn: bool = False,
        local_attn_sigma: float = 2.0,
    ):
        super().__init__()
        assert (
            not init_local_attn or geometry is not None
        ), "geometry required for local attention"
        assert (
            not init_local_attn or attn_bias
        ), "attn_bias required for local attention"
        if geometry is not None:
            assert seq_len, "seq_len required for geometry"
            assert geometry.shape == (seq_len, seq_len), "invalid geometry shape"

        self.embed_dim = embed_dim
        self.depth = depth
        self.recurrent = recurrent
        self.seq_len = seq_len
        self.init_local_attn = init_local_attn
        self.local_attn_sigma = local_attn_sigma

        num_blocks = 1 if recurrent else depth
        mlp_ranks = mlp_rank if isinstance(mlp_rank, list) else [mlp_rank]
        if len(mlp_ranks) == 1:
            mlp_ranks = num_blocks * mlp_ranks

        self.blocks = nn.ModuleList(
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                untied=untied,
                seq_len=seq_len,
                mlp_rank=mlp_ranks[ii],
                attn_mode=attn_mode,
                skip_attn=skip_attn,
                attn_bias=attn_bias,
                attn_head_bias=attn_head_bias,
                qkv_bias=qkv_bias,
                qk_head_dim=qk_head_dim,
                no_vp=no_vp,
                attn_drop=attn_drop_rate,
                proj_drop=proj_drop_rate,
                act_layer=act_layer,
            )
            for ii in range(num_blocks)
        )

        self.register_buffer("geometry", geometry)
        self.init_weights()

    def init_weights(self):
        if self.init_local_attn:
            attn_bias = gaussian_local_attn_bias(
                self.geometry, sigma=self.local_attn_sigma
            )
            for block in self.blocks:
                block.attn.init_bias(attn_bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if self.seq_len and x.shape[1] < self.seq_len:
            x = F.pad(x, (0, 0, 0, self.seq_len - x.shape[1]))

        features = []
        attns = []
        for step in range(self.depth):
            x, attn = self.blocks[0 if self.recurrent else step](x)
            features.append(x)
            attns.append(attn)
        features = torch.stack(features, dim=1)  # B, depth, N, D
        attns = torch.stack(attns, dim=1)  # B, depth, nh, N, N

        state = {"features": features, "attns": attns}
        return x, state

    def extra_repr(self) -> str:
        return (
            f"depth={self.depth}, recurrent={self.recurrent}, "
            f"geometry={None if self.geometry is None else tuple(self.geometry.shape)}, "
            f"init_local_attn={self.init_local_attn}, "
            f"local_attn_sigma={self.local_attn_sigma}"
        )


def gaussian_local_attn_bias(
    dist: torch.Tensor, sigma: float = 2.0, min: Optional[float] = -8.0
):
    attn_bias = -(dist**2) / (2 * sigma**2)
    if min is not None:
        attn_bias = torch.clamp(attn_bias, min=min)
    return attn_bias
