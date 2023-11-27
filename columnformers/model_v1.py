from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from timm.layers.helpers import to_2tuple
from torch import nn

Layer = Callable[..., nn.Module]


class ColumnAttention(nn.Module):
    """
    Communicate between columns using attention.

    Changes compared to standard multi-head attention:

        - Untied weights across sequence
        - only one head
        - low-dim key and query (to save params)
        - value = input

    TODO:
        - sparse connectivity
    """

    def __init__(
        self,
        seq_len: int,
        dim: int,
        qk_dim: int,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.scale = dim**-0.5

        self.q = ColumnLinear(seq_len, dim, qk_dim, bias=True)
        self.k = ColumnLinear(seq_len, dim, qk_dim, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(
        self, x: torch.Tensor, return_attn: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        q = self.scale * self.q(x)
        k = self.k(x)
        attn = q @ k.transpose(-2, -1)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ x
        if return_attn:
            return x, attn
        return x


class ColumnMlp(nn.Module):
    """
    An independent Mlp for each feature column in a sequence. Following timm Mlp.
    """

    def __init__(
        self,
        seq_len: int,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Optional[Layer] = nn.GELU,
        bias: Union[bool, Tuple[bool, bool]] = True,
        drop: Union[float, Tuple[float, float]] = 0.0,
    ):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        biases = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = ColumnLinear(seq_len, in_features, hidden_features, bias=biases[0])
        self.act = act_layer() if act_layer is not None else nn.Identity()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = ColumnLinear(seq_len, hidden_features, out_features, bias=biases[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class ColumnLinear(nn.Module):
    """
    An independent linear layer for each column in a sequence.
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
        # Adapted from nn.Linear
        bound = self.in_features**-0.5
        nn.init.uniform_(self.weight, -bound, bound)
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


class ColumnNorm(nn.Module):
    """
    Layer norm with untied weight and bias across columns.
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

    def extra_repr(self) -> str:
        return (
            f"{self.seq_len}, {self.dim}, eps={self.eps}, "
            f"elementwise_affine={self.elementwise_affine}"
        )


class WiringCost(nn.Module):
    """
    L1 penalty weighted by wiring distance.
    """

    geometry: torch.Tensor
    dist: torch.Tensor

    def __init__(self, geometry: torch.Tensor, lambd: float = 0.01):
        super().__init__()
        self.lambd = lambd

        dist = torch.cdist(geometry, geometry)
        self.register_buffer("geometry", geometry)
        self.register_buffer("dist", dist)

    def forward(self, edges: torch.Tensor):
        cost = self.lambd * (edges.abs() * self.dist).sum(dim=(-2, -1)).mean()
        return cost

    def extra_repr(self) -> str:
        return f"{self.geometry.shape[0]}, {self.geometry.shape[1]}, lambd={self.lambd}"


class ColumnBlock(nn.Module):
    def __init__(
        self,
        seq_len: int,
        dim: int,
        mlp_ratio: float = 1 / 8.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        act_layer: Layer = nn.GELU,
    ):
        super().__init__()
        inner_dim = int(mlp_ratio * dim)

        self.norm1 = ColumnNorm(seq_len, dim)
        self.attn = ColumnAttention(
            seq_len=seq_len, dim=dim, qk_dim=inner_dim, attn_drop=attn_drop
        )
        self.norm2 = ColumnNorm(seq_len, dim)
        self.mlp = ColumnMlp(
            seq_len=seq_len,
            in_features=dim,
            hidden_features=inner_dim,
            out_features=dim,
            act_layer=act_layer,
            drop=(0.0, proj_drop),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        input, attn = self.attn(self.norm1(x), return_attn=True)
        x = x + input
        x = x + self.mlp(self.norm2(x))
        return x, attn


class Columnformer(nn.Module):
    def __init__(
        self,
        seq_len: int,
        dim: int,
        depth: int = 6,
        mlp_ratio: float = 1 / 8.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        act_layer: Layer = nn.GELU,
    ):
        super().__init__()
        self.depth = depth

        self.block = ColumnBlock(
            seq_len=seq_len,
            dim=dim,
            mlp_ratio=mlp_ratio,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            act_layer=act_layer,
        )

    def forward(
        self, x: torch.Tensor, depth: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        depth = depth or self.depth
        attn = None
        for _ in range(depth):
            x, step_attn = self.block(x)
            attn = step_attn if attn is None else attn + step_attn
        attn = attn / depth
        return x, attn
