import logging

import pytest
import torch
from fvcore.nn import FlopCountAnalysis

from columnformers.models.columnformer import Block, Columnformer

CONFIGS = {
    "transformer": {
        "embed_dim": 384,
        "depth": 6,
        "num_heads": 6,
        "mlp_ratio": 4.0,
    },
    "columnformer_ff": {
        "attn_mode": "untied",
        "mlp_mode": "untied",
        "norm_mode": "untied",
        "embed_dim": 384,
        "depth": 6,
        "num_heads": 1,
        "mlp_ratio": 1 / 6.0,
        "seq_len": 64,
        "skip_attn": False,
        "attn_bias": True,
        "qk_head_dim": 64,
        "no_vp": True,
    },
    "columnformer_r": {
        "attn_mode": "untied",
        "mlp_mode": "untied",
        "norm_mode": "untied",
        "embed_dim": 384,
        "depth": 6,
        "recurrent": True,
        "num_heads": 1,
        "mlp_ratio": 1 / 6.0,
        "seq_len": 384,
        "skip_attn": False,
        "attn_bias": True,
        "qk_head_dim": 64,
        "no_vp": True,
    },
    "columnformer_ff_sel": {
        "attn_mode": "selection",
        "mlp_mode": "untied",
        "norm_mode": "untied",
        "embed_dim": 384,
        "depth": 6,
        "num_heads": 6,
        "mlp_ratio": 1 / 6.0,
        "seq_len": 64,
        "skip_attn": True,
        "attn_bias": True,
        "attn_head_bias": True,
    },
    "columnformer_ff_mix": {
        "attn_mode": "mixing",
        "mlp_mode": "untied",
        "norm_mode": "untied",
        "embed_dim": 384,
        "depth": 6,
        "num_heads": 6,
        "mlp_ratio": 1 / 6.0,
        "seq_len": 64,
        "skip_attn": True,
        "attn_bias": True,
        "attn_head_bias": True,
    },
    "transformer_moe": {
        "mlp_mode": "moe",
        "embed_dim": 384,
        "depth": 6,
        "num_heads": 6,
        "mlp_ratio": 4.0,
        "seq_len": 64,
        "moe_experts": [1, 1, 2, 2, 4, 4],
        "mlp_conserve": True,
    },
}


@pytest.mark.parametrize(
    "config",
    [
        "transformer",
        "columnformer_ff",
        "columnformer_r",
        "columnformer_ff_sel",
        "columnformer_ff_mix",
        "transformer_moe",
    ],
)
def test_model(config: str):
    torch.manual_seed(42)
    model = Columnformer(**CONFIGS[config])
    logging.info("Model:\n%s", model)

    x = torch.randn(1, 64, model.embed_dim)
    output, state = model.forward(x)
    attn = state["attn"]
    logging.info("Output: %s, Attention: %s", output.shape, attn.shape)

    flops = FlopCountAnalysis(model, x)
    logging.info("FLOPs: %.1fM", flops.total() / 1e6)
    logging.info("Params: %.1fM", sum(p.numel() for p in model.parameters()) / 1e6)


def test_moe_block():
    block = Block(
        attn_mode="moe",
        mlp_mode="moe",
        num_heads=6,
        seq_len=64,
        moe_experts=8,
    )

    # check that all coefficients are tied
    assert block.coef is block.attn.q.coef
    assert block.coef is block.attn.k.coef
    assert block.coef is block.mlp.fc1.coef
    assert block.coef is block.mlp.fc2.coef

    x = torch.randn(2, 64, 384)
    x, _ = block(x)
    loss = x.square().mean()
    loss.backward()

    # check that all coefficient grads are tied
    assert torch.allclose(block.coef.weight.grad, block.attn.q.coef.weight.grad)
    assert torch.allclose(block.coef.weight.grad, block.attn.k.coef.weight.grad)
    assert torch.allclose(block.coef.weight.grad, block.mlp.fc1.coef.weight.grad)
    assert torch.allclose(block.coef.weight.grad, block.mlp.fc2.coef.weight.grad)


if __name__ == "__main__":
    pytest.main([__file__])
