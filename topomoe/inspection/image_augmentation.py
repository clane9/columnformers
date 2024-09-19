import random

import torch
from torch import nn as nn


class GaussianNoise(nn.Module):
    def __init__(
        self,
        batch_size: int,
        device: str,
        shuffle_every: bool = False,
        std: float = 1.0,
        max_iter: int = 400,
    ):
        super().__init__()
        self.batch_size, self.std_p, self.max_iter = batch_size, std, max_iter
        self.std = None
        self.rem = max_iter - 1
        self.device = device
        self.shuffle()
        self.shuffle_every = shuffle_every

    def shuffle(self):
        self.std = (
            torch.randn(self.batch_size, 3, 1, 1).to(self.device)
            * self.rem
            * self.std_p
            / self.max_iter
        )
        self.rem = (self.rem - 1 + self.max_iter) % self.max_iter

    def forward(self, img: torch.tensor) -> torch.tensor:
        if self.shuffle_every:
            self.shuffle()
        return img + self.std


class Clip(nn.Module):
    @torch.no_grad()
    def forward(self, x: torch.tensor) -> torch.tensor:
        return x.clamp(min=0, max=1)


class ColorJitter(nn.Module):
    def __init__(
        self,
        batch_size: int,
        device: str,
        shuffle_every: bool = False,
        mean: float = 1.0,
        std: float = 1.0,
    ):
        super().__init__()
        self.batch_size, self.mean_p, self.std_p = batch_size, mean, std
        self.mean = self.std = None
        self.device = device
        self.shuffle()
        self.shuffle_every = shuffle_every

    def shuffle(self):
        self.mean = (
            (
                torch.rand(
                    (
                        self.batch_size,
                        3,
                        1,
                        1,
                    )
                ).to(self.device)
                - 0.5
            )
            * 2
            * self.mean_p
        )
        self.std = (
            (
                torch.rand(
                    (
                        self.batch_size,
                        3,
                        1,
                        1,
                    )
                ).to(self.device)
                - 0.5
            )
            * 2
            * self.std_p
        ).exp()

    def forward(self, img: torch.tensor) -> torch.tensor:
        if self.shuffle_every:
            self.shuffle()
        return (img - self.mean) / self.std


class RepeatBatch(nn.Module):
    def __init__(self, repeat: int = 32):
        super().__init__()
        self.size = repeat

    def forward(self, img: torch.tensor):
        return img.repeat(self.size, 1, 1, 1)


class Jitter(nn.Module):
    def __init__(self, lim: int = 32):
        super().__init__()
        self.lim = lim

    def forward(self, x: torch.tensor) -> torch.tensor:
        off1 = random.randint(-self.lim, self.lim)
        off2 = random.randint(-self.lim, self.lim)
        return torch.roll(x, shifts=(off1, off2), dims=(2, 3))


class Tile(nn.Module):
    def __init__(self, rep: int = 384 // 16):
        super().__init__()
        self.rep = rep

    def forward(self, x: torch.tensor) -> torch.tensor:
        dim = x.dim()
        if dim < 3:
            raise NotImplementedError
        elif dim == 3:
            x.unsqueeze(0)
        final_shape = x.shape[:2] + (x.shape[2] * self.rep, x.shape[3] * self.rep)
        return (
            x.unsqueeze(2)
            .unsqueeze(4)
            .repeat(1, 1, self.rep, 1, self.rep, 1)
            .view(final_shape)
        )
