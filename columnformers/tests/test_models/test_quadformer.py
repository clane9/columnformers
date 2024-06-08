import logging

import pytest
import torch
from fvcore.nn import FlopCountAnalysis

from columnformers.models.quadformer import quadformer_tiny_patch16_128


@pytest.mark.parametrize("mlp_conserve", [False, True])
def test_quadformer(mlp_conserve: bool):
    torch.manual_seed(42)
    model = quadformer_tiny_patch16_128(mlp_conserve=mlp_conserve)
    logging.info("Model:\n%s", model)

    x = torch.randn(2, 3, 128, 128)
    output, state = model.forward(x)
    logging.info("Output: %s", tuple(output.shape))
    logging.info("State: %s", {k: tuple(v.shape) for k, v in state.items()})

    flops = FlopCountAnalysis(model, x)
    logging.info("FLOPs: %.1fM", flops.total() / 1e6)
    logging.info("Params: %.1fM", sum(p.numel() for p in model.parameters()) / 1e6)
