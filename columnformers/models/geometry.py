from typing import Optional, Tuple

import torch


def multilayer_geometry(
    widths: Tuple[int, ...], depth_offset: float = 2.0
) -> torch.Tensor:
    """
    Construct a 3D geometry for a stack of square layers.

    Args:
        widths: tuple of layer widths
        depth_offset: distance between layers

    Returns:
        dist, shape (N, N)
    """
    layers = []
    for ii, width in enumerate(widths):
        points = torch.linspace(-width / 2, width / 2, width)
        coords = torch.cartesian_prod(torch.tensor([ii * depth_offset]), points, points)
        layers.append(coords)
    embedding = torch.cat(layers)
    dist = torch.cdist(embedding, embedding)
    return dist


def l1_wiring_cost(
    attn: torch.Tensor,
    weight: torch.Tensor,
    scale: bool = True,
) -> torch.Tensor:
    cost = (attn * weight).sum(dim=(-2, -1)).mean()
    if scale:
        cost = cost / weight.max()
    return cost


def gaussian_local_attn_bias(
    dist: torch.Tensor, sigma: float = 2.0, min: Optional[float] = -8.0
):
    attn_bias = -(dist**2) / (2 * sigma**2)
    if min is not None:
        attn_bias = torch.clamp(attn_bias, min=min)
    return attn_bias
