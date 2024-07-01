from typing import Callable, List

import torch
import torch.nn.functional as F
from timm.layers import trunc_normal_
from torch import nn

try:
    from triton.ops.blocksparse import matmul as blocksparse_matmul  # noqa

    triton_available = True
except ImportError:
    triton_available = False

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

    TODO:
        [ ] initialize weight and bias. weight should be masked by connectivity at init.
            think about what the appropriate init std should be.
        [ ] create a dsd sparse matmul kernel following xformers.BlockSparseAttention:
            https://github.com/facebookresearch/xformers/blob/fad50d49834ab18dd137acc727bd4d567ff17842/xformers/components/attention/blocksparse.py#L96
        [ ] implement forward that should mask weight by connectivity and then call the
            blocksparse matmul kernel
    """

    def __init__(
        self, connectivity: torch.Tensor, bias: bool = True, blocksize: int = 16
    ):
        assert triton_available, "blocksparse linear requires triton"
        super().__init__()
        self.in_features = connectivity.shape[1]
        self.out_features = connectivity.shape[0]
        self.blocksize = blocksize

        # convert to torch blocksparse representation if not already
        connectivity = connectivity.to_sparse_bsr(blocksize)

        # block sparse layout as expected by triton
        # shape (1, out_features // block, in_features // block)
        # must be dtype int64
        layout = torch.sparse_csr_tensor(
            connectivity.crow_indices(),
            connectivity.col_indices(),
            torch.ones_like(connectivity.col_indices()),
        )
        layout = layout.to_dense().unsqueeze(0)

        # only keep raw values, don't need indices since we have layout
        # shape (nnz_blocks, block, block)
        connectivity = (connectivity.values() > 0).float()

        self.register_buffer("connectivity", connectivity)
        self.register_buffer("layout", layout)

        # TODO: initialize weight and bias

    def reset_parameters(self):
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def extra_repr(self) -> str:
        return (
            f"{self.in_features}, {self.out_features}, "
            f"bias={self.bias is not None}, blocksize={self.blocksize}"
        )


class BlockSparseLocallyConnected(nn.Module):
    """
    A locally connected layer implemented using block sparse linear.

    TODO: main step is just computing the connectivity based on conv params. shape
    should be something like: (out_height * out_width * out_channels, in_height *
    in_width * in_channels). Then we just use BlockSparseLinear.
    """
