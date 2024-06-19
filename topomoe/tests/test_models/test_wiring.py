import logging
from typing import Tuple, Type

import pytest
import torch

from topomoe.models.wiring import L1WiringCost, CrossEntropyWiringCost


@pytest.fixture(scope="module")
def geo_embeds() -> Tuple[torch.Tensor, torch.Tensor]:
    dim = 384
    scale = dim**-0.5
    return scale * torch.randn(64, dim), scale * torch.randn(128, dim)


@pytest.mark.parametrize("cls", [L1WiringCost, CrossEntropyWiringCost])
def test_wiring_cost(
    cls: Type[L1WiringCost], geo_embeds: Tuple[torch.Tensor, torch.Tensor]
):
    geo_embed, in_geo_embed = geo_embeds
    cost_fn = cls(geo_embed, in_geo_embed)
    logging.info("Wiring cost function: %s", cost_fn)

    attn = torch.softmax(geo_embed @ in_geo_embed.t(), dim=-1)
    loss = cost_fn.forward(attn)
    logging.info("Loss: %.3e", loss.item())
