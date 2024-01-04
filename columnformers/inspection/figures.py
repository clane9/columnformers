from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

from .registry import register_figure

plt.rcParams["figure.dpi"] = 150
plt.style.use("ggplot")


@register_figure("attn_feat_corr")
class AttentionFeatureCorrelation:
    def __call__(self, state: Dict[str, torch.Tensor]):
        attn = state.get("attn")
        features = state.get("features")
        if attn is None or features is None:
            return None
        return attn_feat_corr(attn, features)


def attn_feat_corr(
    attn: torch.Tensor,
    features: torch.Tensor,
    num_examples: int = 16,
    num_col: int = 4,
    plotw: float = 3.4,
    ploth: float = 3.0,
):
    num_examples = min(num_examples, len(attn))
    assert attn.ndim == features.ndim == 3
    assert num_examples % num_col == 0

    attn = attn.detach()[:num_examples]
    features = features.detach()[:num_examples]

    norm_features = F.normalize(features, dim=2)
    feat_corr = norm_features @ norm_features.transpose(1, 2)

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
class ImageGrid:
    def __call__(self, state: Dict[str, torch.Tensor]):
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
