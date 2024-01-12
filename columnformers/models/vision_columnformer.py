from typing import Dict, Literal, Optional, Tuple

import torch
from einops.layers.torch import Rearrange
from timm.layers import trunc_normal_
from torch import nn

from columnformers.typing import Columnformer

from .registry import create_model, register_model


class VisionColumnformer(nn.Module):
    def __init__(
        self,
        encoder: Columnformer,
        img_size: int = 128,
        patch_size: int = 16,
        output_len: Optional[int] = 256,
        num_classes: int = 100,
        global_pool: Literal["", "avg", "spatial"] = "avg",
        use_fc_norm: bool = True,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.output_len = output_len or encoder.seq_len
        self.num_classes = num_classes
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
            self.register_module("pool", None)
        # Shared head
        self.fc_norm = nn.LayerNorm(self.embed_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()
        if num_classes:
            self.head = nn.Linear(self.embed_dim, num_classes)
        else:
            self.head = nn.Identity()

    def forward_features(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # patch embedding
        patches = self.to_patches(images)
        input = self.patch_embed(patches)

        # pad input to seq_len
        # NOTE: we assume that the first num_patches columns are input columns
        input = self.pad(input)

        features, state = self.encoder(input)
        return features, state

    def forward_head(self, features: torch.Tensor) -> torch.Tensor:
        # pool output features
        # pooled: B, M, C
        pooled = features[:, -self.output_len :]
        if self.global_pool == "spatial":
            # pooled: B, K, C
            pooled = self.pool(pooled)
        elif self.global_pool == "avg":
            # pooled: B, C
            pooled = pooled.mean(dim=1)
        pooled = self.head_drop(self.fc_norm(pooled))

        # predict head
        if self.global_pool == "spatial":
            output = torch.sum(pooled * self.head.weight, dim=-1) + self.head.bias
        else:
            output = self.head(pooled)
        return output

    def forward(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        features, state = self.forward_features(images)
        output = self.forward_head(features)
        return output, state


class SpatialPool(nn.Module):
    """
    Pool a sequence of features with a learned attention weight per class.

    Args:
        seq_len: Length of the sequence, N.
        num_classes: Number of classes, K.
        drop: Dropout probability.

    Shape:
        - Input: (B, N, C)
        - Output: (B, K, C)
    """

    def __init__(self, seq_len: int, num_classes: int, drop: float = 0.0):
        super().__init__()
        self.seq_len = seq_len
        self.num_classes = num_classes

        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()
        self.weight = nn.Parameter(torch.empty(num_classes, seq_len))
        self.reset_parameters()

    def reset_parameters(self):
        trunc_normal_(self.weight, std=0.2)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        attn = torch.softmax(self.weight, dim=1)
        attn = self.drop(attn)
        output = attn @ input
        return output

    def extra_repr(self) -> str:
        return f"{self.seq_len}, {self.num_classes}"


@register_model
def vision_columnformer_patch16_128(
    version: str = "v1",
    num_classes: int = 100,
    global_pool: Literal["", "avg", "spatial"] = "avg",
    drop_rate: float = 0.0,
    layer_widths: Tuple[int, ...] = (12, 16),
    **kwargs,
):
    # prepend 8x8 input columns
    layer_widths = (8,) + tuple(layer_widths)
    # output read out from last layer
    output_len = layer_widths[-1] ** 2

    encoder = create_model(
        f"columnformer_multilayer_{version}",
        layer_widths=layer_widths,
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
    )
    return model
