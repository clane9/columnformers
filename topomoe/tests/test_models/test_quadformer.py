import logging

import pytest
import torch
from fvcore.nn import FlopCountAnalysis

from topomoe.models.quadformer import Quadformer

CONFIGS = {
    "quadformer_2stage": {
        "depths": (3, 3),
        "embed_dim": 384,
        "num_heads": 6,
    },
    "quadformer_3stage": {
        "depths": (2, 2, 2),
        "embed_dim": 384,
        "num_heads": 6,
    },
}


@pytest.mark.parametrize(
    "config",
    [
        "quadformer_2stage",
        "quadformer_3stage",
    ],
)
def test_model(config: str):
    torch.manual_seed(42)
    model = Quadformer(**CONFIGS[config])
    logging.info("Model:\n%s", model)

    x = torch.randn(1, 3, 128, 128)
    output, losses, state = model.forward(x)
    logging.info("Output: %s", tuple(output.shape))
    logging.info("Losses: %s", {k: v.item() for k, v in losses.items()})
    logging.info("State: %s", {k: _get_shape(v) for k, v in state.items()})

    flops = FlopCountAnalysis(model, x)
    logging.info("FLOPs: %.1fM", flops.total() / 1e6)
    logging.info("Params: %.1fM", sum(p.numel() for p in model.parameters()) / 1e6)


def _get_shape(v):
    if isinstance(v, torch.Tensor):
        return tuple(v.shape)
    return None
