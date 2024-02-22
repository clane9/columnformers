from pathlib import Path
from typing import Dict

import pytest
import torch

from columnformers.inspection import create_figure

OUTDIR = Path("test_results/test_figures")
OUTDIR.mkdir(parents=True, exist_ok=True)


@pytest.fixture(scope="module")
def train_state() -> Dict[str, torch.Tensor]:
    state = {
        "image": torch.randn(32, 3, 128, 128),
        "features": torch.randn(32, 6, 64, 384),
        "attns": torch.softmax(torch.randn(32, 6, 6, 64, 64), dim=-1),
    }
    return state


@pytest.mark.parametrize(
    "name",
    [
        "image_grid",
        "attn_grid",
        "feat_corr_grid",
        "image_attn_maps",
    ],
)
def test_figures(name: str, train_state: Dict[str, torch.Tensor]):
    figure_fun = create_figure(name)
    fig = figure_fun(train_state)
    fig.savefig(OUTDIR / f"{name}.png")


if __name__ == "__main__":
    pytest.main([__file__])
