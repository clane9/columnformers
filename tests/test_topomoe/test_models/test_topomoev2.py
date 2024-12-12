import logging

import pytest
import torch
from fvcore.nn import FlopCountAnalysis

from topomoe.models.topomoev2 import (
    ExpertLinear,
    TopoMaps,
    TopoMoEMLP,
    TopoMoETransformerV2,
)

CONFIGS = {
    "topomoev2_2stage_equal": {
        "depths": (3, 3),
        "widths": 8,
        "embed_dim": 384,
        "num_experts": (1, 4),
        "num_heads": 6,
        "wiring_lambd": 0.01,
    },
}


@pytest.mark.parametrize(
    "config",
    [
        "topomoev2_2stage_equal",
    ],
)
def test_model(config: str):
    torch.manual_seed(42)
    model = TopoMoETransformerV2(**CONFIGS[config])
    logging.info("Model:\n%s", model)

    x = torch.randn(1, 3, 128, 128)
    output, losses, state = model.forward(x)
    logging.info("Output: %s", tuple(output.shape))
    logging.info("Losses: %s", {k: v.item() for k, v in losses.items()})
    logging.info("State: %s", {k: _get_shape(v) for k, v in state.items()})

    flops = FlopCountAnalysis(model, x)
    logging.info("FLOPs: %.1fM", flops.total() / 1e6)
    logging.info("Params: %.1fM", sum(p.numel() for p in model.parameters()) / 1e6)


def test_expert_linear():
    linear = ExpertLinear(4, 384, 768)
    logging.info("%s", linear)

    # N, E, S, C
    x = torch.randn(2, 4, 32, 384)
    z = linear.forward(x)
    assert tuple(z.shape) == (2, 4, 32, 768)

    z1 = x[:, 1] @ linear.weight[1].t() + linear.bias[1]
    assert torch.allclose(z[:, 1], z1)


def test_topo_moe_mlp():
    grid_embed = torch.nn.Parameter(0.02 * torch.randn(128, 384))
    maps = TopoMaps(4, grid_embed)
    mlp = TopoMoEMLP(
        maps=maps,
        in_features=384,
        hidden_features=768,
        out_features=384,
        expert_capacity=2,
    )
    logging.info("%s", mlp)

    x = torch.randn(2, 128, 384)
    z = mlp.forward(x)
    assert tuple(z.shape) == (2, 128, 384)


def _get_shape(v):
    if isinstance(v, torch.Tensor):
        return tuple(v.shape)
    return None
