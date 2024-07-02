import logging
import torch
from torch import nn

import columnformers.models.layers as L


def test_block_sparse_locally_connected():
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

    nn.init.ones_(loc.bsl.weight.values())
    nn.init.ones_(conv.weight)
    nn.init.zeros_(loc.bsl.bias)
    nn.init.zeros_(conv.bias)

    # TODO: finish testing on cuda
    input = torch.randn(2, 8, 16, 16)
    output_loc = loc(input)
    output_conv = conv(input)
    assert torch.allclose(output_loc, output_conv)
