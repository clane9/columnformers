import logging
from typing import Any, Dict, Literal, Optional, Tuple

import torch
import torch.nn.functional as F
from timm.layers import PatchEmbed
from torch import nn

from columnformers.utils import filter_kwargs

from .columnformer import Columnformer
from .geometry import multilayer_geometry
from .layers import MixtureCoefficients, SpatialPool, init_weights
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

    def forward_features(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x = self.patch_embed(x)
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


def _create_vision_columnformer(
    encoder_params: Optional[Dict[str, Any]] = None,
    encoder_defaults: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
    defaults: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> VisionColumnformer:
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    encoder_kwargs, _ = filter_kwargs(Columnformer, kwargs)
    kwargs = {k: v for k, v in kwargs.items() if k not in encoder_kwargs}
    kwargs, extra_args = filter_kwargs(VisionColumnformer, kwargs)
    if extra_args:
        logging.warning("Extra kwargs to VisionColumnformer: %s", extra_args)

    encoder_kwargs = {**encoder_defaults, **encoder_kwargs}
    encoder = Columnformer(**encoder_params, **encoder_kwargs)

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
        "geometry": multilayer_geometry(8),
    }
    encoder_defaults = {"num_heads": 6}
    params = {"img_size": 128, "patch_size": 16}
    defaults = {}
    model = _create_vision_columnformer(
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
        "geometry": multilayer_geometry(8),
    }
    encoder_defaults = {
        "attn_mode": "linmixing",
        "mlp_mode": "moe",
        "norm_mode": "classic",
        "num_heads": 6,
        "num_experts": [1, 1, 2, 2, 4, 4],
        "mlp_conserve": True,
    }
    params = {"img_size": 128, "patch_size": 16}
    defaults = {}
    model = _create_vision_columnformer(
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
        "geometry": multilayer_geometry(8),
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
        "geometry": multilayer_geometry(6 * (8,)),
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
        "geometry": multilayer_geometry(6 * (8,)),
    }
    encoder_defaults = {
        "attn_mode": "classic",
        "mlp_mode": "moe",
        "norm_mode": "classic",
        "num_heads": 6,
        "mlp_ratio": 1.0,
        "num_experts": 24,
        "mlp_conserve": False,
        "attn_bias": True,
    }
    params = {"img_size": 128, "patch_size": 16}
    defaults = {}
    model = _create_vision_columnformer(
        encoder_params=encoder_params,
        encoder_defaults=encoder_defaults,
        params=params,
        defaults=defaults,
        **kwargs,
    )
    return model


@register_model
def vision_tut_ff_tiny_patch16_128(**kwargs):
    encoder_params = {
        "embed_dim": 384,
        "depth": 6,
        "recurrent": True,
        "seq_len": 384,
        "geometry": multilayer_geometry(6 * (8,)),
        # Direct connections to each layer.
        # Note that these indices are into `cat([input, x], dim=1)`, so the first layer
        # gets input, second layer gets first layer output, etc.
        "direct_edges": torch.arange(384),
    }
    encoder_defaults = {
        "attn_mode": "classic",
        "mlp_mode": "moe",
        "norm_mode": "classic",
        "num_heads": 6,
        "mlp_ratio": 1.0,
        "num_experts": 24,
        "mlp_conserve": False,
        "attn_bias": True,
    }
    params = {"img_size": 128, "patch_size": 16}
    defaults = {}
    model = _create_vision_columnformer(
        encoder_params=encoder_params,
        encoder_defaults=encoder_defaults,
        params=params,
        defaults=defaults,
        **kwargs,
    )
    return model


@register_model
def vision_transformer_r_tiny_patch16_128(**kwargs):
    encoder_params = {
        "attn_mode": "moe",
        "mlp_mode": "moe",
        "norm_mode": "moe",
        "embed_dim": 384,
        "depth": 6,
        "recurrent": True,
        "seq_len": 384,
        "num_experts": 6,
        "mlp_conserve": False,
        "moe_temp_scale": False,
        "geometry": multilayer_geometry(6 * (8,)),
        "direct_edges": torch.arange(384),
    }
    encoder_defaults = {"num_heads": 6}
    params = {"img_size": 128, "patch_size": 16}
    defaults = {}
    model = _create_vision_columnformer(
        encoder_params=encoder_params,
        encoder_defaults=encoder_defaults,
        params=params,
        defaults=defaults,
        **kwargs,
    )

    # manually set coefficients
    # weight: (N, E) pre-softmax
    coef: MixtureCoefficients = model.encoder.blocks[0].coef
    indices = torch.arange(384, device=coef.weight.device) // 64
    weight = F.one_hot(indices, coef.rank).log()
    coef.weight.data.copy_(weight)
    coef.weight.requires_grad_(False)
    return model
