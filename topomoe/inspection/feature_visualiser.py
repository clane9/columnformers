import collections

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from topomoe.utils.saving_funcs import AbstractSaver


def _round(num: float) -> str:
    if num > 100:
        return str(int(round(num, 0)))
    if num > 10:
        return str(round(num, 1))
    return str(round(num, 2))


class InvLoss:
    def __init__(self, coefficient: float = 1.0):
        self.c = coefficient
        self.name = self.__class__.__name__
        self.last_value = 0

    def __call__(self, x: torch.tensor) -> torch.tensor:
        tensor = self.loss(x)
        self.last_value = tensor.item()
        return self.c * tensor

    def loss(self, x: torch.tensor):
        raise NotImplementedError

    def __str__(self):
        return f"{_round(self.c * self.last_value)}({_round(self.last_value)})"

    def reset(self) -> torch.tensor:
        return 0


class BaseTotalVariation(nn.Module):
    def __init__(self, p: int = 2):
        super().__init__()
        self.p = p

    def forward(self, x: torch.tensor) -> torch.tensor:
        x_wise = x[:, :, :, 1:] - x[:, :, :, :-1]
        y_wise = x[:, :, 1:, :] - x[:, :, :-1, :]
        diag_1 = x[:, :, 1:, 1:] - x[:, :, :-1, :-1]
        diag_2 = x[:, :, 1:, :-1] - x[:, :, :-1, 1:]
        return (
            x_wise.norm(p=self.p, dim=(2, 3)).mean()
            + y_wise.norm(p=self.p, dim=(2, 3)).mean()
            + diag_1.norm(p=self.p, dim=(2, 3)).mean()
            + diag_2.norm(p=self.p, dim=(2, 3)).mean()
        )


class TotalVariation(InvLoss):
    def loss(self, x: torch.tensor):
        return self.tv(x) * np.prod(x.shape[-2:]) / self.size

    def __init__(self, p: int = 2, size: int = 224, coefficient: float = 1.0):
        super().__init__(coefficient)
        self.tv = BaseTotalVariation(p)
        self.size = size * size


class LossArray:
    def __init__(self):
        self.losses = []
        self.last_value = 0

    def __add__(self, other: InvLoss):
        self.losses.append(other)
        return self

    def __call__(self, x: torch.tensor):
        tensor = sum(l(x) for l in self.losses)
        self.last_value = tensor.item()
        return tensor

    def header(self) -> str:
        rest = "\t".join(l.name for l in self.losses)
        return f"Loss\t{rest}"

    def __str__(self):
        rest = "\t".join(str(l) for l in self.losses)
        return f"{_round(self.last_value)}\t{rest}"

    def reset(self):
        return sum(l.reset() for l in self.losses)


class ViTAbsHookHolder(nn.Module):
    pass


class BasicHook:
    def __init__(self, module: nn.Module):
        self.hook = module.register_forward_hook(self.base_hook_fn)
        self.activations = None

    def close(self):
        self.hook.remove()

    def base_hook_fn(
        self, model: nn.Module, input_t: torch.tensor, output_t: torch.tensor
    ):
        x = input_t
        x = x[0][0] if isinstance(x[0], tuple) else x[0]
        return self.hook_fn(model, x)

    def hook_fn(self, model: nn.Module, x: torch.tensor):
        raise NotImplementedError


class ViTHook(BasicHook):
    def __init__(self, module: nn.Module, return_output: bool, name: str):
        super().__init__(module)
        self.mode = return_output
        self.name = name

    def base_hook_fn(
        self, model: nn.Module, input_t: torch.tensor, output_t: torch.tensor
    ):
        x = input_t if not self.mode else output_t
        x = x[0] if isinstance(x, tuple) else x
        return self.hook_fn(model, x)

    def hook_fn(self, model: nn.Module, x: torch.tensor):
        self.activations = x


class ViTFeatHook(InvLoss):
    def __init__(self, hook: ViTAbsHookHolder, key: str, coefficient: float = 1.0):
        super().__init__(coefficient)
        self.hook = hook
        self.key = key

    def loss(self, x: torch.tensor):
        d, o = self.hook(x)
        all_feats = d[self.key][0].mean(dim=1)
        mn = min(all_feats.shape)
        return -all_feats[:mn, :mn].diag().mean()


class ViTEnsFeatHook(ViTFeatHook):
    def __init__(
        self, hook: ViTAbsHookHolder, key: str, feat: int = 0, coefficient: float = 1.0
    ):
        super().__init__(hook, key, coefficient)
        self.f = feat

    def loss(self, x: torch.tensor):
        d, o = self.hook(x)
        all_feats = d[self.key][0].mean(dim=1)
        mn = min(all_feats.shape)
        return -all_feats[:mn, self.f].diag().mean()


class ViTGeLUHook(ViTAbsHookHolder):
    def __init__(self, classifier: nn.Module, sl: slice = None):
        super().__init__()
        self.cl = classifier
        sl = slice(None, None) if sl is None else sl
        self.just_save = [m for m in classifier.modules() if isinstance(m, nn.GELU)]
        self.attentions = self.just_save[sl]
        self.high = [ViTHook(m, True, "high") for m in self.attentions]

    def forward(self, x: torch.tensor) -> tuple[dict, torch.tensor]:
        out = self.cl(x)
        options = [self.high]
        options = [
            [o.activations for o in l] if l is not None else None for l in options
        ]
        names = ["high"]
        return {n: o for n, o in zip(names, options) if o is not None}, out


class ImageNetVisualizer:
    def __init__(
        self,
        loss_array: LossArray,
        device: str,
        saver: AbstractSaver = None,
        pre_aug: nn.Module = None,
        post_aug: nn.Module = None,
        steps: int = 2000,
        lr: float = 0.1,
        save_every: int = 200,
        print_every: int = 5,
        **_,
    ):
        self.loss = loss_array
        self.saver = saver

        self.pre_aug = pre_aug
        self.post_aug = post_aug

        self.save_every = save_every
        self.print_every = print_every
        self.steps = steps
        self.lr = lr
        self.device = device

    def __call__(self, img: torch.tensor = None, optimizer: optim.Optimizer = None):
        img = img.detach().clone().to(self.device).requires_grad_()

        optimizer = (
            optimizer
            if optimizer is not None
            else optim.Adam([img], lr=self.lr, betas=(0.5, 0.99), eps=1e-8)
        )
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.steps, 0.0)

        print(f"#i\t{self.loss.header()}", flush=True)

        best_loss = 1e9
        best_loss_img = None
        best_iteration = None

        for i in range(self.steps + 1):
            optimizer.zero_grad()
            augmented = self.pre_aug(img) if self.pre_aug is not None else img
            loss = self.loss(augmented)

            if i % self.print_every == 0:
                print(f"{i}\t{self.loss}", flush=True)
            if i % self.save_every == 0 and self.saver is not None:
                self.saver.save(img, i)

            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            img.data = (self.post_aug(img) if self.post_aug is not None else img).data

            if loss.item() < best_loss:
                best_loss_img = img
                best_iteration = i

            self.loss.reset()
            torch.cuda.empty_cache()

        optimizer.state = collections.defaultdict(dict)
        return img, best_loss_img, best_iteration
