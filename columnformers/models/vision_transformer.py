import logging
from typing import Literal

from timm.models.vision_transformer import VisionTransformer

from .registry import register_model


@register_model
def vision_transformer_patch16_128(
    num_classes: int = 100,
    global_pool: Literal["", "avg", "token"] = "token",
    embed_dim: int = 384,
    depth: int = 12,
    num_heads: int = 6,
    drop_rate: float = 0.0,
    proj_drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    **kwargs,
) -> VisionTransformer:
    if kwargs:
        logging.warning("Extra unused kwargs: %s", kwargs)

    model = VisionTransformer(
        img_size=128,
        patch_size=16,
        num_classes=num_classes,
        global_pool=global_pool,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        drop_rate=drop_rate,
        proj_drop_rate=proj_drop_rate,
        attn_drop_rate=attn_drop_rate,
    )
    return model
