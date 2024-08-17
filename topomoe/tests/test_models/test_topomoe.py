import logging

import pytest
import torch
from fvcore.nn import FlopCountAnalysis

from topomoe.models.topomoe import SoftPool, TopoMoETransformer, TopoLinear, TopoMaps

CONFIGS = {
    "topomoe_2stage_equal": {
        "depths": (3, 3),
        "widths": 8,
        "embed_dim": 384,
        "num_experts": (1, 4),
        "num_heads": 6,
        "wiring_lambd": 0.01,
    },
    "topomoe_3stage_pyramid": {
        "depths": (2, 2, 2),
        "widths": (12, 8, 4),
        "stage_pools": (True, True, True),
        "embed_dim": 384,
        "num_experts": (1, 4, 8),
        "num_heads": 6,
        "wiring_lambd": 0.01,
    },
    "topomoe_2stage_nopool": {
        "depths": (3, 3),
        "widths": (8, 8),
        "stage_pools": (False, False),
        "embed_dim": 384,
        "num_experts": (1, 4),
        "num_heads": 6,
        "wiring_lambd": 0.01,
    },
}


@pytest.mark.parametrize(
    "config",
    [
        "topomoe_2stage_equal",
        "topomoe_3stage_pyramid",
        "topomoe_2stage_nopool",
    ],
)
def test_model(config: str):
    torch.manual_seed(42)
    model = TopoMoETransformer(**CONFIGS[config])
    logging.info("Model:\n%s", model)

    x = torch.randn(1, 3, 128, 128)
    output, losses, state = model.forward(x)
    logging.info("Output: %s", tuple(output.shape))
    logging.info("Losses: %s", {k: v.item() for k, v in losses.items()})
    logging.info("State: %s", {k: _get_shape(v) for k, v in state.items()})

    flops = FlopCountAnalysis(model, x)
    logging.info("FLOPs: %.1fM", flops.total() / 1e6)
    logging.info("Params: %.1fM", sum(p.numel() for p in model.parameters()) / 1e6)


@pytest.fixture(scope="module")
def grid_embed() -> torch.nn.Parameter:
    return torch.nn.Parameter(0.02 * torch.randn(128, 384))


@pytest.mark.parametrize("static", [False, True])
def test_soft_pool(static: bool, grid_embed: torch.nn.Parameter):
    # pooling from 128 to 64 slots
    pool = SoftPool(64, 384, static=static, in_grid_embed=grid_embed)
    logging.info("%s", pool)

    x = torch.randn(2, 128, 384)
    z, state = pool.forward(x)
    p = state["pool"]

    assert tuple(z.shape) == (2, 64, 384)
    assert tuple(p.shape) == ((64, 128) if static else (2, 64, 128))
    assert torch.allclose(p.sum(dim=-1), torch.ones(p.shape[:-1]))


def test_topo_maps(grid_embed: torch.nn.Parameter):
    maps = TopoMaps(4, grid_embed)
    logging.info("%s", maps)

    m, _ = maps.forward()

    assert tuple(m.shape) == (grid_embed.size(-2), 4)
    assert torch.allclose(m.sum(dim=-1), torch.ones(m.size(0)))


def test_topo_linear(grid_embed: torch.nn.Parameter):
    maps = TopoMaps(4, grid_embed)
    linear = TopoLinear(384, 768, maps)
    logging.info("%s", linear)

    x = torch.randn(2, 128, 384)
    z = linear.forward(x)
    assert tuple(z.shape) == (2, 128, 768)


def _get_shape(v):
    if isinstance(v, torch.Tensor):
        return tuple(v.shape)
    return None
