import logging
from typing import Tuple

import pytest
import torch

from columnformers.models.columnformer import gaussian_local_attn_bias
from columnformers.models.geometry import multilayer_geometry
from columnformers.losses import CrossEntropyLoss, L1WiringCost, State, TVMixtureLoss


@pytest.fixture(scope="module")
def loss_data() -> Tuple[State, torch.Tensor, State]:
    torch.manual_seed(42)
    batch = {
        "image": torch.randn(8, 3, 128, 128),
        "label": torch.randint(0, 100, (8,)),
    }
    output = torch.randn((8, 100), requires_grad=True)
    state = {
        "features": torch.randn((8, 6, 64, 384), requires_grad=True),
        "attn": torch.randn((8, 6, 6, 64, 64)).softmax(dim=-1).requires_grad_(),
        "coef": 2 * [None]
        + [torch.randn(64, 2).softmax(dim=-1).requires_grad_() for _ in range(4)],
    }
    return batch, output, state


def test_wiring_cost(loss_data: Tuple[State, torch.Tensor, State]):
    batch, output, state = loss_data

    geometry = multilayer_geometry((8,))
    wiring_cost = L1WiringCost(geometry)

    # random attention
    cost = wiring_cost(batch, output, state)
    cost.backward()
    grad_norm = torch.linalg.norm(state["attn"].grad)
    logging.info(
        "Wiring cost (random): %.3e, grad: %.3e", cost.item(), grad_norm.item()
    )

    # local attention
    attn = gaussian_local_attn_bias(geometry).softmax(dim=-1)
    attn = attn.expand((8, 6, 6, -1, -1)).requires_grad_()
    cost = wiring_cost(batch, output, {"attn": attn})
    cost.backward()
    grad_norm = torch.linalg.norm(attn.grad)
    logging.info("Wiring cost (local): %.3e, grad: %.3e", cost.item(), grad_norm.item())


def test_tv_loss(loss_data: Tuple[State, torch.Tensor, State]):
    batch, output, state = loss_data
    tv_loss = TVMixtureLoss()
    loss = tv_loss(batch, output, state)
    loss.backward()
    grad_norm = torch.linalg.norm(state["coef"][-1].grad)
    logging.info("TV loss (random): %.3e, grad: %.3e", loss.item(), grad_norm.item())


def test_ce_loss(loss_data: Tuple[State, torch.Tensor, State]):
    batch, output, state = loss_data
    ce_loss = CrossEntropyLoss()
    loss = ce_loss(batch, output, state)
    loss.backward()
    grad_norm = torch.linalg.norm(output.grad)
    logging.info("CE loss (random): %.3e, grad: %.3e", loss.item(), grad_norm.item())


if __name__ == "__main__":
    pytest.main([__file__])
