from typing import Dict, Literal, Optional, Tuple

import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import nn

from .layers import GlobalAveragePool, SpatialPool
from .typing import Columnformer


class ImageClassification(nn.Module):
    def __init__(
        self,
        encoder: Columnformer,
        img_size: int = 128,
        patch_size: int = 16,
        output_len: Optional[int] = 256,
        num_classes: int = 1000,
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

    def _forward_features(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # images: B, C, H, W
        # patches: B, N, P
        # input: B, N, C
        # features: B, N, C
        # attn: B, N, N

        # patch embedding
        patches = self.to_patches(images)
        input = self.patch_embed(patches)

        # pad input to seq_len
        # NOTE: we assume that the first num_patches columns are input columns
        input = self.pad(input)

        features, attn = self.encoder(input)
        return features, attn

    def _forward_head(self, features: torch.Tensor) -> torch.Tensor:
        # features: B, N, C
        # pooled: B, K, C if spatial or B, C otherwise
        # output: B, K

        pooled = self.pool(features)
        pooled = self.head_drop(self.fc_norm(pooled))
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

        features, attn = self._forward_features(images)
        output = self._forward_head(features)

        ce_loss = F.cross_entropy(output, targets)
        wiring_cost = self.wiring_lambd * self.encoder.wiring_cost(attn)
        loss = ce_loss + wiring_cost

        state = {
            "features": features,
            "attn": attn,
            "output": output,
            "ce_loss": ce_loss,
            "wiring_cost": wiring_cost,
            "loss": loss,
        }
        return loss, state
