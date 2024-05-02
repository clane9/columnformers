from collections.abc import Sequence
from typing import Union

import torch


def multilayer_geometry(
    widths: Union[int, Sequence[int]], depth_offset: float = 2.0
) -> torch.Tensor:
    """
    Construct a 3D geometry for a stack of square layers.

    Args:
        widths: sequence of layer widths or single layer width
        depth_offset: distance between layers

    Returns:
        dist, shape (N, N)
    """
    if not isinstance(widths, Sequence):
        widths = [widths]

    layers = []
    for ii, width in enumerate(widths):
        points = torch.linspace(-width / 2, width / 2, width)
        coords = torch.cartesian_prod(torch.tensor([ii * depth_offset]), points, points)
        layers.append(coords)
    embedding = torch.cat(layers)
    dist = torch.cdist(embedding, embedding)

    # Assume this is the feedforward case. Add a depth offset.
    if len(widths) == 1:
        dist = torch.sqrt(dist**2 + depth_offset**2)
    return dist
