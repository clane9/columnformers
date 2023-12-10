import logging

import pytest
import torch

from columnformers.model_v1 import Columnformer, WiringCost


def test_model():
    torch.manual_seed(42)
    model = Columnformer(seq_len=256, embed_dim=384, depth=4)
    logging.info("Model:\n%s", model)
    logging.info("Params: %.1fM", sum(p.numel() for p in model.parameters()) / 1e6)

    x = torch.randn(2, 256, 384)
    output, attn = model.forward(x)
    logging.info("Output: %s, Attention: %s", output.shape, attn.shape)


def test_wiring_cost():
    torch.manual_seed(42)
    # 3d grid geometry
    embedding = torch.cartesian_prod(
        2 * torch.arange(4), torch.arange(8), torch.arange(8)
    ).to(torch.float32)
    dist = torch.cdist(embedding, embedding)
    cost_fn = WiringCost(dist)
    logging.info("Cost: %s", cost_fn)

    edges = torch.softmax(torch.rand(256, 256), dim=1)
    cost = cost_fn(edges)
    logging.info("Cost: %.3e", cost.item())


if __name__ == "__main__":
    pytest.main([__file__])
