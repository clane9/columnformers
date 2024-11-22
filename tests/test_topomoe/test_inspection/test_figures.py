import shutil
from pathlib import Path
from typing import Dict

import pytest
import torch
from matplotlib import pyplot as plt

from topomoe.inspection import create_figure

OUTDIR = Path(__file__).parent / "test_results"
if OUTDIR.exists():
    shutil.rmtree(OUTDIR)
OUTDIR.mkdir(parents=True)


@pytest.fixture(scope="module")
def train_state() -> Dict[str, torch.Tensor]:
    state = {
        "stages.0.pool": torch.softmax(torch.randn(32, 6, 100, 64), dim=-1),
        "stages.0.blocks.1.attn": torch.softmax(torch.randn(32, 6, 100, 100), dim=-1),
    }
    return state


def test_attn_grid(train_state: Dict[str, torch.Tensor]):
    figure_fun = create_figure("attn_maps")
    outdir = OUTDIR / "attn_maps"
    outdir.mkdir(exist_ok=True)
    figures = figure_fun(train_state)
    for name, fig in figures.items():
        fig.savefig(outdir / f"{name}.png")
    plt.close("all")
