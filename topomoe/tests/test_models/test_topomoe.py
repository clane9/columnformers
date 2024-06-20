import logging

import pytest
import torch
from fvcore.nn import FlopCountAnalysis

from topomoe.models.topomoe import TopoMoETransformer, TopoLinear, TopoMaps

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
        "embed_dim": 384,
        "num_experts": (1, 4, 8),
        "num_heads": 6,
        "wiring_lambd": 0.01,
    },
}


@pytest.mark.parametrize(
    "config",
    [
        "topomoe_2stage_equal",
        "topomoe_3stage_pyramid",
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
def pos_embed() -> torch.nn.Parameter:
    return torch.nn.Parameter(0.02 * torch.randn(128, 384))


@pytest.mark.parametrize("token_wise", [False, True])
def test_topo_maps(token_wise: bool, pos_embed: torch.nn.Parameter):
    maps = TopoMaps(64, pos_embed, token_wise=token_wise)
    logging.info("%s", maps)

    attn = maps.forward()
    logging.info("Shape: %s", tuple(attn.shape))

    assert torch.allclose(attn.sum(dim=-1), torch.ones(attn.size(0)))


def test_topo_linear():
    pos_embed = torch.nn.Parameter(0.02 * torch.randn(128, 384))
    maps = TopoMaps(4, pos_embed, token_wise=False)
    linear = TopoLinear(384, 768, maps)
    logging.info("%s", linear)

    x = torch.randn(2, 128, 384)
    z = linear.forward(x)
    logging.info("Shape: %s", tuple(z.shape))


def _get_shape(v):
    if isinstance(v, torch.Tensor):
        return tuple(v.shape)
    return None
