import logging

import pytest
import torch
from fvcore.nn import FlopCountAnalysis

from topomoe.models.softmoe import SoftMoEMLP, SoftMoETransformer

CONFIGS = {
    "softmoe_2stage": {
        "depths": (3, 3),
        "slots_per_token": 2,
        "embed_dim": 384,
        "num_heads": 6,
    },
    "softmoe_3stage": {
        "depths": (2, 2, 2),
        "slots_per_token": 2,
        "embed_dim": 384,
        "num_heads": 6,
    },
}


@pytest.mark.parametrize(
    "config",
    [
        "softmoe_2stage",
        "softmoe_3stage",
    ],
)
def test_model(config: str):
    torch.manual_seed(42)
    model = SoftMoETransformer(**CONFIGS[config])
    logging.info("Model:\n%s", model)

    x = torch.randn(1, 3, 128, 128)
    output, losses, state = model.forward(x)
    logging.info("Output: %s", tuple(output.shape))
    logging.info("Losses: %s", {k: v.item() for k, v in losses.items()})
    logging.info("State: %s", {k: _get_shape(v) for k, v in state.items()})

    flops = FlopCountAnalysis(model, x)
    logging.info("FLOPs: %.1fM", flops.total() / 1e6)
    logging.info("Params: %.1fM", sum(p.numel() for p in model.parameters()) / 1e6)


def test_soft_moe_mlp():
    mlp = SoftMoEMLP(
        num_experts=4,
        in_features=384,
        hidden_features=768,
        out_features=384,
    )
    logging.info("%s", mlp)

    x = torch.randn(2, 128, 384)
    z = mlp.forward(x)
    assert tuple(z.shape) == (2, 128, 384)


def _get_shape(v):
    if isinstance(v, torch.Tensor):
        return tuple(v.shape)
    return None
