import logging

import pytest
import torch
from fvcore.nn import FlopCountAnalysis

from columnformers.models import create_model
from columnformers.typing import Columnformer


@pytest.mark.parametrize("model_name", ["columnformer_v1"])
def test_model(model_name: str):
    torch.manual_seed(42)
    model: Columnformer = create_model(model_name)
    logging.info("Model:\n%s", model)

    x = torch.randn(1, model.seq_len, model.embed_dim)
    output, state = model.forward(x)
    attn = state["attn"]
    logging.info("Output: %s, Attention: %s", output.shape, attn.shape)

    cost = model.wiring_cost(attn)
    logging.info("Cost: %.3e", cost.item())

    flops = FlopCountAnalysis(model, x)
    logging.info("Params: %.1fM", sum(p.numel() for p in model.parameters()) / 1e6)
    logging.info("FLOPs: %.1fM", flops.total() / 1e6)


if __name__ == "__main__":
    pytest.main([__file__])
