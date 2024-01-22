import logging

import pytest
import torch
from fvcore.nn import FlopCountAnalysis

from columnformers.models.columnformer import Columnformer

CONFIGS = {
    "transformer": {
        "embed_dim": 384,
        "depth": 6,
        "recurrent": False,
        "num_heads": 6,
        "mlp_ratio": 4.0,
        "untied": False,
        "seq_len": None,
        "skip_attn": True,
        "attn_bias": False,
        "qk_head_dim": None,
        "no_vp": False,
    },
    "columnformer_ff": {
        "embed_dim": 384,
        "depth": 6,
        "recurrent": False,
        "num_heads": 1,
        "mlp_ratio": 1 / 6.0,
        "untied": True,
        "seq_len": 64,
        "skip_attn": False,
        "attn_bias": True,
        "qk_head_dim": 64,
        "no_vp": True,
    },
    "columnformer_r": {
        "embed_dim": 384,
        "depth": 6,
        "recurrent": True,
        "num_heads": 1,
        "mlp_ratio": 1 / 6.0,
        "untied": True,
        "seq_len": 384,
        "skip_attn": False,
        "attn_bias": True,
        "qk_head_dim": 64,
        "no_vp": True,
    },
}


@pytest.mark.parametrize(
    "config",
    ["transformer", "columnformer_ff", "columnformer_r"],
)
def test_model(config: str):
    torch.manual_seed(42)
    model = Columnformer(**CONFIGS[config])
    logging.info("Model:\n%s", model)

    x = torch.randn(1, 64, model.embed_dim)
    output, state = model.forward(x)
    attn = state["attns"]
    logging.info("Output: %s, Attention: %s", output.shape, attn.shape)

    flops = FlopCountAnalysis(model, x)
    logging.info("FLOPs: %.1fM", flops.total() / 1e6)
    logging.info("Params: %.1fM", sum(p.numel() for p in model.parameters()) / 1e6)


if __name__ == "__main__":
    pytest.main([__file__])
