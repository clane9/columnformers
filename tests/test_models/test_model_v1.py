import logging

import pytest
import torch

from columnformers.models.model_v1 import Columnformer, columnformer_v1_patch16_128


def test_model():
    torch.manual_seed(42)
    model: Columnformer = columnformer_v1_patch16_128()
    logging.info("Model:\n%s", model)
    logging.info("Params: %.1fM", sum(p.numel() for p in model.parameters()) / 1e6)

    x = torch.randn(2, model.seq_len, model.embed_dim)
    output, attn = model.forward(x)
    logging.info("Output: %s, Attention: %s", output.shape, attn.shape)

    cost = model.wiring_cost(attn)
    logging.info("Cost: %.3e", cost.item())


if __name__ == "__main__":
    pytest.main([__file__])
