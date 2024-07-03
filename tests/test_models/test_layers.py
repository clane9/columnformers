import pytest

import logging
import torch
from torch import nn

import columnformers.models.layers as L


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
def test_block_sparse_locally_connected():
    torch.manual_seed(42)
    device = torch.device("cuda")

    loc = L.BlockSparseLocallyConnected(
        in_channels=8,
        out_channels=16,
        kernel_size=3,
        height=16,
        depthwise=False,
    )
    logging.info("%s", loc)

    conv = nn.Conv2d(
        in_channels=8,
        out_channels=16,
        kernel_size=3,
        stride=1,
        padding="same",
    )

    nn.init.ones_(loc.bsl.weight)
    nn.init.zeros_(loc.bsl.bias)
    nn.init.ones_(conv.weight)
    nn.init.zeros_(conv.bias)

    loc = loc.to(device)
    conv = conv.to(device)

    input = torch.randn((2, 8, 16, 16), device=device)
    input_loc = input.clone().requires_grad_(True)
    input_conv = input.clone().requires_grad_(True)

    output_loc = loc(input_loc)
    output_conv = conv(input_conv)
    assert torch.allclose(output_loc, output_conv, rtol=1e-4)

    loss_loc = (output_loc**2).mean()
    loss_conv = (output_conv**2).mean()
    loss_loc.backward()
    loss_conv.backward()
    grad_loc = input_loc.grad.data
    grad_conv = input_conv.grad.data
    assert torch.allclose(grad_loc, grad_conv, rtol=1e-4)
