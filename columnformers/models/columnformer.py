from functools import partial
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from timm.layers import trunc_normal_
from timm.layers.helpers import to_2tuple, to_3tuple
from torch import nn

from .layers import Layer, UntiedLayerNorm, UntiedLinear
from .registry import register_model


class Attention(nn.Module):
    """
    Multi-head attention with options for:

        - untied weights across the sequence
        - learned attention bias
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

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_head_dim = qk_head_dim or self.head_dim
        self.scale = self.head_dim**-0.5
        qkv_biases = to_3tuple(qkv_bias)
        linear_layer = partial(UntiedLinear, seq_len=seq_len) if untied else nn.Linear

        if bias:
            self.bias = nn.Parameter(torch.zeros(seq_len, seq_len))
        else:
            self.register_parameter("bias", None)

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
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

    def extra_repr(self) -> str:
        return f"bias={self.bias is not None}"


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
        untied: bool = False,
        seq_len: Optional[int] = None,
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
        linear_layer = partial(UntiedLinear, seq_len=seq_len) if untied else nn.Linear
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
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        untied: Union[bool, Tuple[bool, bool, bool]] = False,
        seq_len: Optional[int] = None,
        skip_attn: bool = True,
        attn_bias: bool = False,
        qkv_bias: Union[bool, Tuple[bool, bool, bool]] = False,
        qk_head_dim: Optional[int] = None,
        no_vp: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        act_layer: Layer = nn.GELU,
    ):
        super().__init__()
        self.skip_attn = skip_attn
        untied_attn, untied_mlp, untied_norm = to_3tuple(untied)
        norm_layer = (
            partial(UntiedLayerNorm, seq_len=seq_len) if untied_norm else nn.LayerNorm
        )

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            untied=untied_attn,
            seq_len=seq_len,
            bias=attn_bias,
            qkv_bias=qkv_bias,
            qk_head_dim=qk_head_dim,
            no_vp=no_vp,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            untied=untied_mlp,
            seq_len=seq_len,
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
    def __init__(
        self,
        embed_dim: int,
        depth: int = 12,
        recurrent: bool = False,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        untied: Union[bool, Tuple[bool, bool, bool]] = False,
        seq_len: Optional[int] = None,
        skip_attn: bool = True,
        attn_bias: bool = False,
        qkv_bias: Union[bool, Tuple[bool, bool, bool]] = True,
        qk_head_dim: Optional[int] = None,
        no_vp: bool = False,
        attn_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        act_layer: Layer = nn.GELU,
    ):
        super().__init__()
        self.depth = depth
        self.recurrent = recurrent
        self.seq_len = seq_len

        num_blocks = 1 if recurrent else depth
        self.blocks = nn.ModuleList(
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                untied=untied,
                seq_len=seq_len,
                skip_attn=skip_attn,
                attn_bias=attn_bias,
                qkv_bias=qkv_bias,
                qk_head_dim=qk_head_dim,
                no_vp=no_vp,
                attn_drop=attn_drop_rate,
                proj_drop=proj_drop_rate,
                act_layer=act_layer,
            )
            for _ in range(num_blocks)
        )

        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if self.seq_len and x.shape[1] < self.seq_len:
            x = F.pad(x, (0, 0, 0, self.seq_len - x.shape[1]))

        features = []
        attns = []
        for step in range(self.depth):
            x, attn = self.blocks[0 if self.recurrent else step](x)
            features.append(x)
            attns.append(attn)

        state = {"features": features, "attns": attns}
        return x, state

    def extra_repr(self) -> str:
        return f"depth={self.depth}, recurrent={self.recurrent}"


def init_weights(module: nn.Module):
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, "init_weights"):
        module.init_weights()


@register_model
def transformer_tiny(**kwargs):
    kwargs = {"num_heads": 6, **kwargs}
    model = Columnformer(
        embed_dim=384, depth=6, recurrent=False, untied=False, seq_len=None, **kwargs
    )
    return model


@register_model
def columnformer_ff_tiny_n64(**kwargs):
    kwargs = {
        "num_heads": 1,
        "mlp_ratio": 1 / 6.0,
        "untied": True,
        "qk_head_dim": 64,
        "no_vp": True,
        **kwargs,
    }
    model = Columnformer(embed_dim=384, depth=6, recurrent=False, seq_len=64, **kwargs)
    return model


@register_model
def columnformer_rnn_tiny_n384(**kwargs):
    kwargs = {
        "num_heads": 1,
        "mlp_ratio": 1 / 6.0,
        "untied": True,
        "qk_head_dim": 64,
        "no_vp": True,
        **kwargs,
    }
    model = Columnformer(embed_dim=384, depth=6, recurrent=True, seq_len=384, **kwargs)
    return model
