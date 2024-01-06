import logging

import pytest
import torch
from fvcore.nn import FlopCountAnalysis

from columnformers.models import create_model
from columnformers.models.vision_columnformer import VisionColumnformer


@pytest.mark.parametrize("global_pool", ["avg", "spatial"])
def test_model(global_pool: str):
    torch.manual_seed(42)
    images = torch.randn(1, 3, 128, 128)

    model: VisionColumnformer = create_model(
        "vision_columnformer_multilayer_patch16_128",
        global_pool=global_pool,
    )
    logging.info("Model:\n%s", model)

    output, state = model.forward(images)
    logging.info("Output: %s", output.shape)
    logging.info("State: %s", {k: v.shape for k, v in state.items()})

    flops = FlopCountAnalysis(model, images)
    logging.info("Params: %.1fM", sum(p.numel() for p in model.parameters()) / 1e6)
    logging.info("FLOPs: %.1fM", flops.total() / 1e6)


if __name__ == "__main__":
    pytest.main([__file__])
