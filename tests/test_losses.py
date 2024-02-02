import logging

import pytest
import torch

from columnformers.models.columnformer import gaussian_local_attn_bias
from columnformers.models.geometry import multilayer_geometry
from columnformers.tasks.losses import WiringCost


def test_wiring_cost():
    torch.manual_seed(42)

    geometry = multilayer_geometry((8,))
    wiring_cost = WiringCost(geometry, lambd=0.01)

    # random attention
    attn = torch.randn(32, 6, 64, 64).softmax(dim=-1)
    cost = wiring_cost(attn)
    logging.info("Cost (random): %.3e", cost.item())

    # local attention
    attn = gaussian_local_attn_bias(geometry).softmax(dim=-1)
    attn = attn.expand((32, 6, -1, -1))
    cost = wiring_cost(attn)
    logging.info("Cost (local): %.3e", cost.item())


if __name__ == "__main__":
    pytest.main([__file__])
