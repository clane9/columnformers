import logging
from typing import Any, Dict, Literal, Optional, Tuple

import torch
from timm.layers import PatchEmbed, trunc_normal_
from torch import nn

from columnformers.utils import filter_kwargs

from .columnformer import Columnformer
from .geometry import multilayer_geometry
from .layers import SpatialPool, init_weights
from .registry import register_model


class VisionColumnformer(nn.Module):
    def __init__(
        self,
        encoder: Columnformer,
        img_size: int = 128,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 100,
        global_pool: Literal["", "avg", "spatial"] = "avg",
        output_len: Optional[int] = None,
        pos_embed: bool = True,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_chans * patch_size**2
        self.embed_dim = encoder.embed_dim
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.output_len = output_len or self.num_patches

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=self.embed_dim,
        )
        if pos_embed:
            self.pos_embed = nn.Parameter(
                torch.empty(1, self.num_patches, self.embed_dim)
            )
        else:
            self.register_parameter("pos_embed", None)

        self.encoder = encoder
        self.norm = nn.LayerNorm(self.embed_dim)

        if self.global_pool == "spatial":
            self.pool = SpatialPool(self.output_len, self.num_classes)
        else:
            self.register_module("pool", None)
        self.head_drop = nn.Dropout(drop_rate)
        if num_classes:
            self.head = nn.Linear(self.embed_dim, num_classes)
        else:
            self.head = nn.Identity()

        self.apply(init_weights)

    def init_weights(self):
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)

    def forward_features(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x, state = self.encoder(x)
        x = self.norm(x)
        return x, state

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, -self.output_len :]
        if self.global_pool == "spatial":
            x = self.pool(x)  # B, K, C
        elif self.global_pool == "avg":
            x = x.mean(dim=1)
        x = self.head_drop(x)
        if self.global_pool == "spatial":
            x = torch.sum(x * self.head.weight, dim=2) + self.head.bias
        else:
            x = self.head(x)
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x, state = self.forward_features(x)
        x = self.forward_head(x)
        return x, state

    @property
    def geometry(self) -> Optional[torch.Tensor]:
        return self.encoder.geometry

    def extra_repr(self) -> str:
        return f"pos_embed={self.pos_embed is not None}"


def _create_vision_columnformer(
    widths: Optional[Tuple[int, ...]] = None,
    encoder_params: Optional[Dict[str, Any]] = None,
    encoder_defaults: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
    defaults: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> VisionColumnformer:
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    if widths:
        depth_offset = kwargs.pop("depth_offset", 2.0)
        geometry = multilayer_geometry(widths, depth_offset=depth_offset)
    else:
        geometry = None

    encoder_kwargs, _ = filter_kwargs(Columnformer, kwargs)
    kwargs = {k: v for k, v in kwargs.items() if k not in encoder_kwargs}
    kwargs, extra_args = filter_kwargs(VisionColumnformer, kwargs)
    if extra_args:
        logging.warning("Extra kwargs to VisionColumnformer: %s", extra_args)

    encoder_kwargs = {**encoder_defaults, **encoder_kwargs}
    encoder = Columnformer(**encoder_params, **encoder_kwargs, geometry=geometry)

    kwargs = {**defaults, **kwargs}
    model = VisionColumnformer(encoder=encoder, **params, **kwargs)
    return model


@register_model
def vision_transformer_tiny_patch16_128(**kwargs):
    encoder_params = {
        "embed_dim": 384,
        "depth": 6,
        "recurrent": False,
        "seq_len": 64,
    }
    encoder_defaults = {"num_heads": 6}
    params = {"img_size": 128, "patch_size": 16}
    defaults = {}
    model = _create_vision_columnformer(
        widths=(8,),
        encoder_params=encoder_params,
        encoder_defaults=encoder_defaults,
        params=params,
        defaults=defaults,
        **kwargs,
    )
    return model


@register_model
def vision_moemixer_tiny_patch16_128(**kwargs):
    encoder_params = {
        "embed_dim": 384,
        "depth": 6,
        "recurrent": False,
        "seq_len": 64,
    }
    encoder_defaults = {
        "attn_mode": "linmixing",
        "mlp_mode": "moe",
        "norm_mode": "classic",
        "num_heads": 6,
        "moe_experts": [1, 1, 2, 2, 4, 4],
        "moe_conserve": True,
    }
    params = {"img_size": 128, "patch_size": 16}
    defaults = {}
    model = _create_vision_columnformer(
        widths=(8,),
        encoder_params=encoder_params,
        encoder_defaults=encoder_defaults,
        params=params,
        defaults=defaults,
        **kwargs,
    )
    return model


@register_model
def vision_columnformer_ff_tiny_patch16_128(**kwargs):
    encoder_params = {
        "embed_dim": 384,
        "depth": 6,
        "recurrent": False,
        "seq_len": 64,
    }
    encoder_defaults = {
        "attn_mode": "untied",
        "mlp_mode": "untied",
        "norm_mode": "untied",
        "num_heads": 1,
        "mlp_ratio": 1 / 6.0,
        "attn_bias": True,
        "qk_head_dim": 64,
        "no_vp": True,
        "init_local_attn": True,
    }
    params = {"img_size": 128, "patch_size": 16}
    defaults = {}
    model = _create_vision_columnformer(
        widths=(8,),
        encoder_params=encoder_params,
        encoder_defaults=encoder_defaults,
        params=params,
        defaults=defaults,
        **kwargs,
    )
    return model


@register_model
def vision_columnformer_r_tiny_patch16_128(**kwargs):
    encoder_params = {
        "embed_dim": 384,
        "depth": 6,
        "recurrent": True,
        "seq_len": 384,
    }
    encoder_defaults = {
        "attn_mode": "untied",
        "mlp_mode": "untied",
        "norm_mode": "untied",
        "num_heads": 1,
        "mlp_ratio": 1 / 6.0,
        "skip_attn": False,
        "attn_bias": True,
        "qk_head_dim": 64,
        "no_vp": True,
        "init_local_attn": True,
    }
    params = {"img_size": 128, "patch_size": 16}
    defaults = {}
    model = _create_vision_columnformer(
        widths=6 * (8,),
        encoder_params=encoder_params,
        encoder_defaults=encoder_defaults,
        params=params,
        defaults=defaults,
        **kwargs,
    )
    return model


@register_model
def vision_tut_tiny_patch16_128(**kwargs):
    encoder_params = {
        "embed_dim": 384,
        "depth": 6,
        "recurrent": True,
        "seq_len": 384,
    }
    encoder_defaults = {
        "attn_mode": "classic",
        "mlp_mode": "moe",
        "norm_mode": "classic",
        "num_heads": 6,
        "mlp_ratio": 1.0,
        "moe_experts": 24,
        "moe_conserve": False,
        "attn_bias": True,
    }
    params = {"img_size": 128, "patch_size": 16}
    defaults = {}
    model = _create_vision_columnformer(
        widths=6 * (8,),
        encoder_params=encoder_params,
        encoder_defaults=encoder_defaults,
        params=params,
        defaults=defaults,
        **kwargs,
    )
    return model
