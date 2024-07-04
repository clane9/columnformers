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


@register_figure("attn_grid")
class AttentionGrid(nn.Module):
    def __init__(
        self,
        pattern: str = r"(\.pool|\.attn)",
        as_maps: bool = True,
    ):
        super().__init__()
        self.pattern = re.compile(pattern)
        self.as_maps = as_maps

    def forward(self, state: Dict[str, torch.Tensor]) -> Dict[str, Figure]:
        figures = {}
        for k, v in state.items():
            if self.pattern.search(k) and v is not None:
                # add batch, head dim for pool maps
                if v.ndim == 2:
                    v = v.view((1, 1, *v.size()))
                f = attn_grid(v, as_maps=self.as_maps, title=k)
                figures[f"{k}.attn_grid"] = f
        return figures

    def extra_repr(self) -> str:
        return f"pattern='{self.pattern.pattern}', as_maps={self.as_maps}"


def attn_grid(
    attn: torch.Tensor,
    num_examples: int = 3,
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
