import warnings
from typing import Callable, List, Literal, Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from timm.layers import trunc_normal_
from torch import nn
from torch.types import _device, _dtype

Layer = Callable[..., nn.Module]


class UntiedLinear(nn.Module):
    """
    Linear layer with untied weights across the sequence.
    """

    def __init__(
        self,
        seq_len: int,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty((seq_len, out_features, in_features)))
        if bias:
            self.bias = nn.Parameter(torch.empty(seq_len, out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        trunc_normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = torch.einsum("bnc,ndc->bnd", input, self.weight)
        if self.bias is not None:
            output = output + self.bias
        return output

    def extra_repr(self) -> str:
        return (
            f"{self.seq_len}, {self.in_features}, {self.out_features}, "
            f"bias={self.bias is not None}"
        )


class MixtureLinear(nn.Module):
    """
    Mixture of linear layers. The linear weights for each token in the sequence are
    computed as a linear combination of the weights in the mixture.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        coef: "MixtureCoefficients",
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.coef = coef

        self.weight = nn.Parameter(torch.empty((out_features, in_features, coef.rank)))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, coef.rank))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        trunc_normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # input: (B, N, C)
        # coef: (N, R)
        coef = self.coef()
        # Nb, this implementation for some reason uses significantly fewer flops
        # compared to equivalent alternatives (e.g. einsum) for some reason.
        weight = (coef @ self.weight.transpose(1, 2)).transpose(0, 1)
        if self.bias is not None:
            bias = coef @ self.bias.t()
        output = torch.einsum("bnc,ndc->bnd", input, weight)
        if self.bias is not None:
            output = output + bias
        return output

    def extra_repr(self) -> str:
        return f"{self.in_features}, {self.out_features}, bias={self.bias is not None}"


class MixtureCoefficients(nn.Module):
    def __init__(
        self,
        seq_len: int,
        rank: int = 16,
        softmax: bool = True,
        temp_scale: bool = True,
    ):
        super().__init__()
        assert not temp_scale or softmax, "temp_scale requires softmax"

        self.seq_len = seq_len
        self.rank = rank
        self.softmax = softmax
        self.temp_scale = temp_scale

        self.weight = nn.Parameter(torch.empty((seq_len, rank)))
        # scale is a per-token scaling in log-space following CLIP
        if self.temp_scale:
            self.scale = nn.Parameter(torch.empty((seq_len)))
        else:
            self.register_parameter("scale", None)
        self.reset_parameters()

    def reset_parameters(self):
        trunc_normal_(self.weight, std=self.rank**-0.5)
        if self.temp_scale:
            nn.init.zeros_(self.scale)

    def forward(self) -> torch.Tensor:
        coef = self.weight.clone()
        if self.temp_scale:
            scale = torch.clamp(torch.exp(self.scale), max=100)
            coef = scale[:, None] * F.normalize(coef, dim=1)
        if self.softmax:
            coef = coef.softmax(dim=1)
        return coef

    def extra_repr(self) -> str:
        return (
            f"{self.seq_len}, {self.rank}, softmax={self.softmax}, "
            f"temp_scale={self.temp_scale}"
        )


class UntiedLayerNorm(nn.Module):
    """
    Layer norm with untied weights across the sequence.
    """

    def __init__(
        self,
        seq_len: int,
        dim: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.empty(seq_len, dim))
            self.bias = nn.Parameter(torch.empty(seq_len, dim))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = F.layer_norm(input, (self.dim,), eps=self.eps)
        if self.elementwise_affine:
            input = input * self.weight + self.bias
        return input

    def no_weight_decay(self) -> List[str]:
        # Nb, not excluded by default since 2d
        return ["weight", "bias"]

    def extra_repr(self) -> str:
        return (
            f"{self.seq_len}, {self.dim}, eps={self.eps}, "
            f"elementwise_affine={self.elementwise_affine}"
        )


class MixtureLayerNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        coef: "MixtureCoefficients",
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.coef = coef

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.empty(dim, coef.rank))
            self.bias = nn.Parameter(torch.empty(dim, coef.rank))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = F.layer_norm(input, (self.dim,), eps=self.eps)
        if self.elementwise_affine:
            coef = self.coef()
            weight = coef @ self.weight.t()
            bias = coef @ self.bias.t()
            input = input * weight + bias
        return input

    def no_weight_decay(self) -> List[str]:
        # Nb, not excluded by default since 2d
        return ["weight", "bias"]

    def extra_repr(self) -> str:
        return (
            f"{self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}"
        )


class SpatialPool(nn.Module):
    """
    Pool a sequence of features with a learned attention weight per class.

    Args:
        seq_len: Length of the sequence, N.
        num_classes: Number of classes, K.
        drop: Dropout probability.

    Shape:
        - Input: (B, N, C)
        - Output: (B, K, C)
    """

    def __init__(self, seq_len: int, num_classes: int, drop: float = 0.0):
        super().__init__()
        self.seq_len = seq_len
        self.num_classes = num_classes

        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()
        self.weight = nn.Parameter(torch.empty(num_classes, seq_len))
        self.reset_parameters()

    def reset_parameters(self):
        trunc_normal_(self.weight, std=0.2)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        attn = torch.softmax(self.weight, dim=1)
        attn = self.drop(attn)
        output = attn @ input
        return output

    def extra_repr(self) -> str:
        return f"{self.seq_len}, {self.num_classes}"


def init_weights(module: nn.Module):
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, "init_weights"):
        module.init_weights()


class BlockSparseLinear(nn.Module):
    """
    A linear layer with block sparse connectivity.

    Args:
        connectivity: a binary tensor of shape (out_features, in_features) representing
            the connectivity between input and output units.
        bias: use bias
        blocksize: sparse block size, e.g. 16, 32. Must divide each dimension of
            connectivity
    """

    crow_indices: torch.Tensor
    col_indices: torch.Tensor
    mask: torch.Tensor

    def __init__(
        self, connectivity: torch.Tensor, bias: bool = True, blocksize: int = 32
    ):
        super().__init__()
        device_capability = _cuda_get_device_capability()
        if device_capability is None or device_capability < (8, 0):
            warnings.warn(
                "BlockSparseLinear only supported for CUDA A100 or higher",
                UserWarning,
            )

        self.in_features = connectivity.shape[1]
        self.out_features = connectivity.shape[0]
        self.blocksize = blocksize

        # convert to torch blocksparse representation if not already
        connectivity = connectivity.to_sparse_bsr(blocksize).float()
        self.register_buffer("crow_indices", connectivity.crow_indices())
        self.register_buffer("col_indices", connectivity.col_indices())
        self.register_buffer("mask", connectivity.values())

        n_blocks = (self.out_features // blocksize) * (self.in_features // blocksize)
        nnz_blocks = connectivity.values().size(0)
        self.sparsity = 1 - (nnz_blocks / n_blocks)

        # The weight parameter is just the sparse bsr values. We construct the sparse
        # bsr tensor on the fly. Trying to use a sparse bsr layout parameter doesn't
        # work. Mapping the model to cuda fails. The tensor container is mapped but not
        # the underlying values.
        self.weight = nn.Parameter(torch.empty_like(connectivity.values()))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        trunc_normal_(self.weight, std=0.02)
        self.weight.data.mul_(self.mask)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # apply sparse connectivity mask
        weight = self.mask * self.weight
        weight = torch.sparse_bsr_tensor(
            crow_indices=self.crow_indices,
            col_indices=self.col_indices,
            values=weight,
            size=(self.out_features, self.in_features),
        )
        x = F.linear(x, weight, self.bias)
        return x

    def extra_repr(self) -> str:
        return (
            f"{self.in_features}, {self.out_features}, "
            f"bias={self.bias is not None}, blocksize={self.blocksize}, "
            f"sparsity={self.sparsity:.2f}"
        )


class BlockSparseLocallyConnected(nn.Module):
    """
    A locally connected layer implemented using block sparse linear.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        height: int,
        depthwise: bool = False,
        bias: bool = True,
        blocksize: int = 32,
        in_shape: Literal["nlc", "nchw", "nd"] = "nchw",
    ):
        super().__init__()
        assert isinstance(kernel_size, int), "only square kernels supported"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.height = height
        self.kernel_size = kernel_size
        self.depthwise = depthwise
        self.blocksize = blocksize
        self.in_shape = in_shape
        self.channels_last = not depthwise

        connectivity = _sparse_local_connectivity(
            in_channels,
            out_channels,
            kernel_size,
            height,
            depthwise=depthwise,
            channels_last=self.channels_last,
        )
        self.bsl = BlockSparseLinear(
            connectivity=connectivity, bias=bias, blocksize=blocksize
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        needs_reshape = self.in_shape != "nd"

        if needs_reshape:
            in_pattern = "n (h w) c" if self.in_shape == "nlc" else "n c h w"
            out_pattern = "n (h w c)" if self.channels_last else "n (c h w)"
            input = rearrange(
                input, f"{in_pattern} -> {out_pattern}", h=self.height, w=self.height
            )

        output = self.bsl(input)

        if needs_reshape:
            output = rearrange(
                output,
                f"{out_pattern} -> {in_pattern}",
                c=self.out_channels,
                h=self.height,
                w=self.height,
            )
        return output


def _sparse_local_connectivity(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    height: int,
    depthwise: bool = False,
    channels_last: bool = False,
    dtype: _dtype = None,
    device: _device = None,
) -> torch.Tensor:
    """
    Construct sparse local connectivity matrix, shape
    (out_channels * height * height, in_channels * height * height). The returned
    connectivity will have sparse COO layout.

    The connectivity pattern is equivalent to
    `nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding="same")`

    If channels_last is True, the shape of connectivity in einops notation is
    "(h w cout) (h w cin)". Otherwise, it is "(cout h w) (cin h w)". The latter should
    be more efficient when depthwise is True (connectivity will be more block sparse).
    """
    assert kernel_size % 2 == 1, "kernel_size must be odd"
    assert (
        not depthwise or out_channels == in_channels
    ), "in channels must match out channels for depthwise"
    N = height * height

    # ij indices of input grid
    # (h^2, 2)
    col_indices = torch.cartesian_prod(torch.arange(height), torch.arange(height))

    # conv kernel index offsets. note that the kernel width is required to be odd.
    # (k^2, 2)
    kernel_half_width = (kernel_size - 1) // 2
    kernel_indices = torch.cartesian_prod(
        torch.arange(-kernel_half_width, kernel_half_width + 1),
        torch.arange(-kernel_half_width, kernel_half_width + 1),
    )

    # input edge indices for each output unit. these will be the column indices for the
    # sparse COO connectivity.
    # (h^2, k^2, 2)
    col_indices = col_indices.unsqueeze(1) + kernel_indices.unsqueeze(0)

    # input edge row indices
    # (h^2, k^2)
    row_indices = torch.arange(N).unsqueeze(1).repeat(1, kernel_size**2)

    # exclude edges falling outside grid
    mask = ((col_indices >= 0) & (col_indices < height)).all(axis=-1)
    col_indices = col_indices[mask]
    row_indices = row_indices[mask]

    # rasterize column indices
    col_indices = height * col_indices[..., 0] + col_indices[..., 1]

    # add channel blocks with full or depthwise (diagonal) connectivity
    if depthwise:
        channel_indices = torch.arange(out_channels).unsqueeze(1).repeat(1, 2)
    else:
        channel_indices = torch.cartesian_prod(
            torch.arange(out_channels), torch.arange(in_channels)
        )

    # we can insert the channels axis either at the front or the back
    # front is better for depthwise=True, back is better for depthwise=False
    if channels_last:
        row_indices = out_channels * row_indices.unsqueeze(1) + channel_indices[:, 0]
        col_indices = in_channels * col_indices.unsqueeze(1) + channel_indices[:, 1]
    else:
        row_indices = N * channel_indices[:, 0].unsqueeze(1) + row_indices
        col_indices = N * channel_indices[:, 1].unsqueeze(1) + col_indices

    row_indices = row_indices.flatten()
    col_indices = col_indices.flatten()

    # construct sparse connectivity tensor
    connectivity = (
        torch.sparse_coo_tensor(
            torch.stack([row_indices, col_indices]),
            torch.ones(len(row_indices), dtype=dtype),
            size=(out_channels * N, in_channels * N),
        )
        .coalesce()
        .to(device)
    )
    return connectivity


def _cuda_get_device_capability() -> Optional[Tuple[int, int]]:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_capability()
