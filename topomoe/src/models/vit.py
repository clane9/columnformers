import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
from . import wiring
from .common import Attention, Layer, Mlp, State, init_weights, model_factory, to_list
from .registry import register_model


class ViTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, drop_rate=0.0):
        super(ViTBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=drop_rate,
            proj_drop=drop_rate,
        )
        self.drop_path = nn.Dropout(drop_rate)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            drop=drop_rate,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        state = {}

        
        x_norm = self.norm1(x)
        attn_output, attn_state = self.attn(x_norm)
        x = x + self.drop_path(attn_output)
        state["attention_weights"] = attn_state["attn"]  # Store attention weights

       
        x_norm = self.norm2(x)
        mlp_output = self.mlp(x_norm)
        x = x + self.drop_path(mlp_output)

        return x, state


class ViTWithStateAndLoss(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        num_classes=100,
    ):
        super(ViTWithStateAndLoss, self).__init__()
        
        
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)
        self.num_classes = num_classes
        
        
        self.blocks = nn.ModuleList([
            ViTBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_rate=drop_rate)
            for _ in range(depth)
        ])
        
       
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, self.num_classes)  
        self.apply(init_weights)  

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        state = {}
        losses = {}  # Populate losses?
        
        
        for i, block in enumerate(self.blocks):
            x, block_state = block(x)
            state[f"block_{i}"] = block_state  # Collecting intermediate state from each block

        x = self.norm(x)
        return x, losses, state

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        x = x.mean(dim=1)  
        return self.head(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        x, losses, state = self.forward_features(x)
        x = self.forward_head(x)
        return x, losses, state



@register_model
def vit_base_patch16_128(**kwargs):
    params = {
        "img_size": 128,
        "patch_size": 16,
        "in_chans": 3,
        "embed_dim": 384,
        "depth": 12,         #  Vary number of transformer blocks based on size
        "num_heads": 6,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "drop_rate": 0.0,
        "num_classes": 100,  
    }
    

    model = ViTWithStateAndLoss(
        img_size=params["img_size"],
        patch_size=params["patch_size"],
        in_chans=params["in_chans"],
        embed_dim=params["embed_dim"],
        depth=params["depth"],
        num_heads=params["num_heads"],
        mlp_ratio=params["mlp_ratio"],
        qkv_bias=params["qkv_bias"],
        drop_rate=params["drop_rate"],
        num_classes=params["num_classes"]
    )
    return model
