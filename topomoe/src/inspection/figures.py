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

plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 10
plt.style.use("ggplot")


@register_figure
class FeatureCorrMaps(nn.Module):
    name = "feat_corr_maps"
    pattern = re.compile(r"\.features")

    def __init__(self, num_examples: int = 4):
        super().__init__()
        self.num_examples = num_examples

    def forward(self, state: Dict[str, torch.Tensor]) -> Dict[str, Figure]:
        figures = {}
        for k, v in state.items():
            if self.pattern.search(k) and v is not None:
                # B, N, C -> B, N, N
                feat = F.normalize(v.detach(), dim=-1)
                feat_corr = feat @ feat.transpose(-2, -1)
                feat_corr = feat_corr
                f = plot_maps(
                    feat_corr,
                    num_examples=self.num_examples,
                    title=f"{k} feat corr maps",
                )
                figures[f"{self.name}-{k}"] = f
        return figures

    def extra_repr(self) -> str:
        return f"num_examples={self.num_examples}"


@register_figure
class PoolMaps(nn.Module):
    name = "pool_maps"
    pattern = re.compile(r"\.pool")

    def __init__(self, num_examples: int = 4):
        super().__init__()
        self.num_examples = num_examples

    def forward(self, state: Dict[str, torch.Tensor]) -> Dict[str, Figure]:
        figures = {}
        for k, v in state.items():
            if self.pattern.search(k) and v is not None:
                pool = v.detach()
                # N, M -> B, N, M
                if pool.ndim == 2:
                    pool = pool.unsqueeze(0)
                f = plot_maps(
                    pool, num_examples=self.num_examples, title=f"{k} pool maps"
                )
                figures[f"{self.name}-{k}"] = f
        return figures

    def extra_repr(self) -> str:
        return f"num_examples={self.num_examples}"


@register_figure
class ExpertMaps(nn.Module):
    name = "expert_maps"
    pattern = re.compile(r"\.maps")

    def forward(self, state: Dict[str, torch.Tensor]) -> Dict[str, Figure]:
        figures = {}
        for k, v in state.items():
            if self.pattern.search(k) and v is not None:
                # N, E -> B, N, E
                maps = v.detach().unsqueeze(0)
                f = plot_maps(
                    maps, num_examples=1, as_grid=False, title=f"{k} expert maps"
                )
                figures[f"{self.name}-{k}"] = f
        return figures


@register_figure
class AttentionMaps(nn.Module):
    name = "attn_maps"
    pattern = re.compile(r"\.attn")

    def __init__(self, head: int = 0, num_examples: int = 4):
        super().__init__()
        self.head = head
        self.num_examples = num_examples

    def forward(self, state: Dict[str, torch.Tensor]) -> Dict[str, Figure]:
        figures = {}
        for k, v in state.items():
            if self.pattern.search(k) and v is not None:
                # B, nh, N, M -> B, N, M
                attn = v.detach()[:, self.head]
                f = plot_maps(
                    attn, num_examples=self.num_examples, title=f"{k} attn maps"
                )
                figures[f"{self.name}-{k}"] = f
        return figures

    def extra_repr(self) -> str:
        return f"head={self.head}, num_examples={self.num_examples}"


@torch.no_grad()
def plot_maps(
    maps: torch.Tensor,
    num_examples: int = 4,
    as_grid: bool = True,
    pad: int = 1,
    ploth: float = 3.0,
    title: Optional[str] = None,
):
    num_examples = min(num_examples, len(maps))
    # B, N, M
    assert maps.ndim == 3

    maps = maps[:num_examples]

    N, M = maps.shape[-2:]
    HN, HM = math.isqrt(N), math.isqrt(M)
    assert N == HN * HN and (
        not as_grid or M == HM * HM
    ), "Expected square sequence lengths"

    # rearrange to form grid of maps with padding
    if as_grid:
        maps = rearrange(
            maps,
            "b (h1 w1) (h2 w2) -> b h1 w1 h2 w2",
            h1=HN,
            w1=HN,
            h2=HM,
            w2=HM,
        )
        maps = F.pad(maps, 4 * (pad,), value=float("nan"))
        maps = rearrange(maps, "b h1 w1 h2 w2 -> b (h1 h2) (w1 w2)")
        plotw = ploth
    else:
        maps = rearrange(maps, "b (h1 w1) m -> b m h1 w1", h1=HN, w1=HN)
        maps = F.pad(maps, 2 * (pad,), value=float("nan"))
        maps = rearrange(maps, "b m h1 w1 -> b h1 (m w1)")
        plotw = M * ploth

    nr = 1
    nc = num_examples
    if title:
        ploth = ploth + 0.1
    f, axs = plt.subplots(nr, nc, figsize=(nc * plotw, nr * ploth), squeeze=False)

    for jj in range(nc):
        plt.sca(axs[0, jj])
        imshow(maps[jj])

    if title is not None:
        plt.suptitle(title, fontsize="medium")
    plt.tight_layout(pad=0.5)
    return f


@register_figure
class ImageGrid(nn.Module):
    name = "image_grid"

    def __init__(self, num_examples: int = 32):
        super().__init__()
        self.num_examples = num_examples

    def forward(self, state: Dict[str, torch.Tensor]) -> Dict[str, Figure]:
        images = state.get("image")
        if images is None:
            return {}
        return {self.name: image_grid(images, num_examples=self.num_examples)}

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
