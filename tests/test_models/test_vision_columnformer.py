import logging

import pytest
import torch
from fvcore.nn import FlopCountAnalysis

from columnformers.models import create_model
from columnformers.models.vision_columnformer import VisionColumnformer


@pytest.mark.parametrize(
    "model_name",
    [
        "vision_transformer_tiny_patch16_128",
        "vision_moemixer_tiny_patch16_128",
        "vision_columnformer_ff_tiny_patch16_128",
        "vision_columnformer_r_tiny_patch16_128",
        "vision_tut_tiny_patch16_128",
        "vision_tut_res_tiny_patch16_128",
    ],
)
@pytest.mark.parametrize("global_pool", ["avg", "spatial"])
def test_model(model_name: str, global_pool: str):
    torch.manual_seed(42)
    images = torch.randn(1, 3, 128, 128)

    model: VisionColumnformer = create_model(model_name, global_pool=global_pool)
    logging.info("Model:\n%s", model)

    output, state = model.forward(images)
    logging.info("Output: %s", output.shape)
    logging.info("State: %s", {k: _get_shape(v) for k, v in state.items()})

    flops = FlopCountAnalysis(model, images)
    logging.info("FLOPs: %.1fM", flops.total() / 1e6)
    logging.info("Params: %.1fM", sum(p.numel() for p in model.parameters()) / 1e6)


def _get_shape(v):
    if isinstance(v, torch.Tensor):
        return v.shape
    elif isinstance(v, (list, tuple)):
        return [_get_shape(vv) for vv in v]
    else:
        return None


if __name__ == "__main__":
    pytest.main([__file__])
