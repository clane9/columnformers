import logging
from typing import Dict, Literal, Optional, Tuple

import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from timm.models.vision_transformer import VisionTransformer as ViT
from torch import nn

from .layers import GlobalAveragePool, SpatialPool
from .registry import create_model, register_model
from .typing import Columnformer


class VisionColumnformer(nn.Module):
    def __init__(
        self,
        encoder: Columnformer,
        img_size: int = 128,
        patch_size: int = 16,
        output_len: Optional[int] = 256,
        num_classes: int = 100,
        global_pool: Literal["avg", "spatial"] = "avg",
        drop_rate: float = 0.0,
        wiring_lambd: float = 0.01,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.output_len = output_len or encoder.seq_len
        self.num_classes = num_classes
        self.wiring_lambd = wiring_lambd
        self.global_pool = global_pool
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = 3 * patch_size**2
        self.seq_len = encoder.seq_len
        self.embed_dim = encoder.embed_dim

        self.to_patches = Rearrange(
            "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=patch_size, p2=patch_size
        )
        # Shared patch embedding
        self.patch_embed = nn.Linear(self.patch_dim, self.embed_dim)
        self.pad = nn.ZeroPad2d((0, 0, 0, self.seq_len - self.num_patches))
        # Columnformer encoder
        self.encoder = encoder
        # Output feature pooling
        if self.global_pool == "spatial":
            # Learned spatial pooling per class
            self.pool = SpatialPool(self.output_len, self.num_classes)
        else:
            self.pool = GlobalAveragePool(self.output_len)
        # Shared head
        self.fc_norm = nn.LayerNorm(self.embed_dim)
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.embed_dim, self.num_classes)

    def forward_features(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # images: B, C, H, W

        # patch embedding
        # patches: B, N, P
        # input: B, N, C
        patches = self.to_patches(images)
        input = self.patch_embed(patches)

        # pad input to seq_len
        # NOTE: we assume that the first num_patches columns are input columns
        input = self.pad(input)

        # features: B, N, C
        # attn: B, N, N
        features, attn = self.encoder(input)
        return features, attn

    def forward_head(self, features: torch.Tensor) -> torch.Tensor:
        # features: B, N, C

        # pooled: B, K, C if spatial or B, C otherwise
        pooled = self.pool(features)
        pooled = self.head_drop(self.fc_norm(pooled))

        # output: B, K
        if self.global_pool == "spatial":
            output = (pooled * self.head.weight).sum(dim=-1) + self.head.bias
        else:
            output = self.head(pooled)
        return output

    def forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        images = batch["image"]
        targets = batch["label"]

        features, attn = self.forward_features(images)
        output = self.forward_head(features)

        ce_loss = F.cross_entropy(output, targets)
        wiring_cost = self.wiring_lambd * self.encoder.wiring_cost(attn)
        loss = ce_loss + wiring_cost

        state = {
            "image": images,
            "label": targets,
            "features": features,
            "attn": attn,
            "output": output,
            "ce_loss": ce_loss,
            "wiring_cost": wiring_cost,
            "loss": loss,
        }
        return loss, state


class VisionTransformer(nn.Module):
    def __init__(self, encoder: ViT):
        super().__init__()
        self.encoder = encoder

    def forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        images = batch["image"]
        targets = batch["label"]

        features = self.encoder.forward_features(images)
        output = self.encoder.forward_head(features)
        loss = F.cross_entropy(output, targets)

        state = {
            "image": images,
            "label": targets,
            "features": features,
            "output": output,
            "loss": loss,
        }
        return loss, state


@register_model
def vision_columnformer_multilayer_patch16_128(
    version: str = "v1",
    num_classes: int = 100,
    global_pool: Literal["avg", "spatial"] = "avg",
    drop_rate: float = 0.0,
    wiring_lambd: float = 0.01,
    layer_widths: Tuple[int, ...] = (12, 16),
    embed_dim: int = 384,
    depth: int = 6,
    **kwargs,
):
    layer_widths = (8,) + layer_widths
    # output read out from last layer
    output_len = layer_widths[-1] ** 2

    encoder = create_model(
        f"columnformer_multilayer_{version}",
        layer_widths=layer_widths,
        embed_dim=embed_dim,
        depth=depth,
        **kwargs,
    )
    model = VisionColumnformer(
        encoder=encoder,
        img_size=128,
        patch_size=16,
        output_len=output_len,
        num_classes=num_classes,
        global_pool=global_pool,
        drop_rate=drop_rate,
        wiring_lambd=wiring_lambd,
    )
    return model


@register_model
def vision_transformer_patch16_128(
    num_classes: int = 100,
    global_pool: Literal["avg", "token"] = "token",
    embed_dim: int = 384,
    depth: int = 12,
    num_heads: int = 6,
    drop_rate: float = 0.0,
    proj_drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    **kwargs,
):
    if kwargs:
        logging.warning("Extra unused kwargs: %s", kwargs)

    encoder = ViT(
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
    model = VisionTransformer(encoder)
    return model
