from typing import Dict

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import nn

from .registry import register_figure

matplotlib.use("Agg")
plt.rcParams["figure.dpi"] = 150
plt.style.use("ggplot")


@register_figure("attn_feat_corr")
class AttentionFeatureCorrelation(nn.Module):
    def __init__(self, step: int = -1):
        super().__init__()
        self.step = step

    def forward(self, state: Dict[str, torch.Tensor]):
        attns = state.get("attns")
        features = state.get("features")
        if attns is None or features is None:
            return None

        # depth, B, nh, N, N
        attn = attns.detach()[self.step].mean(dim=1)
        # depth, B, N, D
        feat = features.detach()[self.step]
        return attn_feat_corr(attn, feat)

    def extra_repr(self) -> str:
        return f"step={self.step}"


def attn_feat_corr(
    attn: torch.Tensor,
    feat: torch.Tensor,
    num_examples: int = 16,
    num_col: int = 4,
    plotw: float = 3.4,
    ploth: float = 3.0,
):
    num_examples = min(num_examples, len(attn))
    assert attn.ndim == feat.ndim == 3
    assert num_examples % num_col == 0

    attn = attn.detach()[:num_examples]
    feat = feat.detach()[:num_examples]
    norm_feat = F.normalize(feat, dim=2)
    feat_corr = norm_feat @ norm_feat.transpose(1, 2)

    nr = num_examples // num_col
    nc = 2 * num_col
    f, axs = plt.subplots(nr, nc, figsize=(nc * plotw, nr * ploth))
    axs = axs.flatten()

    idx = 0
    for ii in range(num_examples):
        plt.sca(axs[idx])
        imshow(attn[ii], colorbar=True)
        if ii < num_col:
            plt.title("Attn", fontsize=8)
        idx += 1

        plt.sca(axs[idx])
        imshow(feat_corr[ii], colorbar=True)
        if ii < num_col:
            plt.title("Feat corr", fontsize=8)
        idx += 1

    plt.tight_layout(pad=0.5)
    return f


@register_figure("image_grid")
class ImageGrid(nn.Module):
    def forward(self, state: Dict[str, torch.Tensor]):
        images = state.get("image")
        if images is None:
            return None
        return image_grid(images)


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
    f, axs = plt.subplots(nr, nc, figsize=(nc * plotw, nr * ploth))
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
