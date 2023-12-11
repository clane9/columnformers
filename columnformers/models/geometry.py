from typing import List

import torch


def multilayer_embedding(widths: List[int], offset: float = 2.0) -> torch.Tensor:
    """
    Construct a 3D embedding for a stack of square layers.

    Args:
        widths: list of layer widths
        offset: distance between layers

    Returns:
        embedding, shape (N, 3)
    """
    layers = []
    for ii, width in enumerate(widths):
        points = torch.linspace(-width / 2, width / 2, width)
        coords = torch.cartesian_prod(torch.tensor([ii * offset]), points, points)
        layers.append(coords)
    embedding = torch.cat(layers)
    return embedding
