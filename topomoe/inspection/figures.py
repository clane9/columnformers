import math
import re
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from torch import nn

from .registry import register_figure

plt.switch_backend("Agg")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 10
plt.style.use("ggplot")


@register_figure("feat_corr_maps")
class FeatureCorrMaps(nn.Module):
    pattern = re.compile(r"\.features")

    def __init__(self, num_examples: int = 4):
        super().__init__()
        self.num_examples = num_examples

    def forward(self, state: Dict[str, torch.Tensor]) -> Dict[str, Figure]:
        figures = {}
        for k, v in state.items():
            if self.pattern.search(k) and v is not None:
                # B, N, C
                feat = F.normalize(v.detach(), dim=-1)
                feat_corr = feat @ feat.transpose(-2, -1)
                feat_corr = feat_corr.unsqueeze(1)  # add dummy head dim
                f = attn_grid(
                    feat_corr,
                    num_examples=self.num_examples,
                    title=f"{k} feat corr maps",
                )
                figures[k] = f
        return figures

    def extra_repr(self) -> str:
        return f"num_examples={self.num_examples}"


@register_figure("pool_maps")
class PoolMaps(nn.Module):
    pattern = re.compile(r"\.pool")

    def forward(self, state: Dict[str, torch.Tensor]) -> Dict[str, Figure]:
        figures = {}
        for k, v in state.items():
            if self.pattern.search(k) and v is not None:
                # N, M
                pool = v.detach()
                pool = pool.view((1, 1, *pool.size()))  # add dummy batch head dim
                f = attn_grid(pool, num_examples=1, title=f"{k} pool maps")
                figures[k] = f
        return figures


@register_figure("attn_maps")
class AttentionMaps(nn.Module):
    pattern = re.compile(r"\.attn")

    def __init__(self, num_examples: int = 4):
        super().__init__()
        self.num_examples = num_examples

    def forward(self, state: Dict[str, torch.Tensor]) -> Dict[str, Figure]:
        figures = {}
        for k, v in state.items():
            if self.pattern.search(k) and v is not None:
                attn = v.detach()
                f = attn_grid(
                    attn, num_examples=self.num_examples, title=f"{k} attn maps"
                )
                figures[k] = f
        return figures

    def extra_repr(self) -> str:
        return f"num_examples={self.num_examples}"


def attn_grid(
    attn: torch.Tensor,
    num_examples: int = 4,
    head: Optional[int] = 0,
    as_maps: bool = True,
    pad: int = 1,
    plotw: float = 3.0,
    ploth: float = 3.0,
    title: Optional[str] = None,
):
    num_examples = min(num_examples, len(attn))
    # B, nh, N, M
    assert attn.ndim == 4

    attn = attn.detach()[:num_examples]
    attn = attn.mean(dim=1) if head is None else attn[:, head]

    N, M = attn.shape[-2:]
    HN, HM = math.isqrt(N), math.isqrt(M)
    assert N == HN * HN and M == HM * HM, "Expected square sequence lengths"

    # rearrange to form grid of maps with padding
    if as_maps:
        attn = rearrange(
            attn,
            "b (h1 w1) (h2 w2) -> b h1 w1 h2 w2",
            h1=HN,
            w1=HN,
            h2=HM,
            w2=HM,
        )
        attn = F.pad(attn, 4 * (pad,), value=float("nan"))
        attn = rearrange(attn, "b h1 w1 h2 w2 -> b (h1 h2) (w1 w2)")
        ploth = ploth * (HN * HM) / 64
        plotw = plotw * (HN * HM) / 64
    else:
        ploth = ploth * (HN * HN) / 64
        plotw = plotw * (HM * HM) / 64

    nr = 1
    nc = num_examples
    if title:
        ploth = ploth + 0.1
    f, axs = plt.subplots(nr, nc, figsize=(nc * plotw, nr * ploth), squeeze=False)

    for jj in range(nc):
        plt.sca(axs[0, jj])
        imshow(attn[jj])

    if title is not None:
        plt.suptitle(title, fontsize="medium")
    plt.tight_layout(pad=0.5)
    return f


@register_figure("image_grid")
class ImageGrid(nn.Module):
    def __init__(self, num_examples: int = 32):
        super().__init__()
        self.num_examples = num_examples

    def forward(self, state: Dict[str, torch.Tensor]) -> Dict[str, Figure]:
        images = state.get("image")
        if images is None:
            return {}
        return {"image": image_grid(images, num_examples=self.num_examples)}

    def extra_repr(self) -> str:
        return f"num_examples={self.num_examples}"


def image_grid(
    images: torch.Tensor,
    num_examples: int = 32,
    num_col: int = 8,
    plotw: float = 3.0,
    ploth: float = 3.0,
):
    num_examples = min(num_examples, len(images))
    assert images.ndim == 4 and images.shape[1] == 3
    assert num_examples % num_col == 0

    images = images.detach()[:num_examples]
    images = (images - images.min()) / (images.max() - images.min())

    nr = num_examples // num_col
    nc = num_col
    f, axs = plt.subplots(nr, nc, figsize=(nc * plotw, nr * ploth), squeeze=False)
    axs = axs.flatten()

    idx = 0
    for ii in range(num_examples):
        plt.sca(axs[idx])
        imshow(images[ii])
        idx += 1

    plt.tight_layout(pad=0.5)
    return f


def imshow(img: torch.Tensor, cmap: str = "turbo", colorbar: bool = False, **kwargs):
    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()

    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
        cmap = None
    kwargs = {"interpolation": "nearest", "cmap": cmap, **kwargs}
    plt.imshow(img, **kwargs)

    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    if colorbar:
        cb = plt.colorbar(fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=8)
