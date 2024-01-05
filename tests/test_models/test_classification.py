import logging

import pytest
import torch
from fvcore.nn import FlopCountAnalysis

from columnformers.models import create_model
from columnformers.models.classification import VisionColumnformer


@pytest.mark.parametrize("global_pool", ["avg", "spatial"])
def test_model(global_pool: str):
    torch.manual_seed(42)
    batch = {
        "image": torch.randn(1, 3, 128, 128),
        "label": torch.zeros(1, dtype=torch.int64),
    }

    model: VisionColumnformer = create_model(
        "vision_columnformer_multilayer_patch16_128",
        global_pool=global_pool,
    )
    logging.info("Model:\n%s", model)

    _, state = model(batch)
    metric_str = ",  ".join(
        f"{k}: {state[k].item():.3e}" for k in ["ce_loss", "wiring_cost", "loss"]
    )
    logging.info("Metrics: %s", metric_str)

    flops = FlopCountAnalysis(model, batch)
    logging.info("Params: %.1fM", sum(p.numel() for p in model.parameters()) / 1e6)
    logging.info("FLOPs: %.1fM", flops.total() / 1e6)


if __name__ == "__main__":
    pytest.main([__file__])
