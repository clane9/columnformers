"""
Optimization utils.
"""

import fnmatch
import logging
import math
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer

LRSchedule = Callable[[int], float]


def backward_step(
    loss: torch.Tensor,
    optimizer: Optimizer,
    scaler: Optional[GradScaler] = None,
    need_update: bool = True,
    max_grad_norm: Optional[float] = 1.0,
) -> float:
    """
    Compute backward and optimization step with optional gradient clipping and loss
    scaling.
    """
    if scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()

    if need_update:
        total_norm = clip_grad_(optimizer, scaler, max_grad_norm)

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()
    else:
        lr = total_norm = None

    return {"lr": lr, "total_norm": total_norm}


def update_lr_(optimizer: Optimizer, lr: float):
    for group in optimizer.param_groups:
        group["lr"] = lr


def clip_grad_(
    optimizer: Optimizer,
    scaler: Optional[GradScaler] = None,
    max_grad_norm: Optional[float] = 1.0,
):
    max_grad_norm = max_grad_norm or float("inf")

    # unscale the gradients of optimizer's assigned params in-place
    if scaler is not None:
        scaler.unscale_(optimizer)

    params = [p for group in optimizer.param_groups for p in group["params"]]
    total_norm = clip_grad_norm_(params, max_norm=max_grad_norm).item()
    return total_norm


class OptimizationStep:
    def __init__(
        self,
        lr_schedule: Callable[[int], float],
        max_grad_norm: float = 1.0,
    ):
        self.lr_schedule = lr_schedule
        self.max_grad_norm = max_grad_norm

    def step(
        self,
        step: int,
        loss: torch.Tensor,
        optimizer: Optimizer,
        scaler: Optional[GradScaler] = None,
        need_update: bool = True,
    ) -> Dict[str, float]:
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if need_update:
            lr = self.lr_schedule(step)
            self._update_lr(optimizer, lr)
            total_norm = self._clip_grad(optimizer, scaler)

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        else:
            lr = total_norm = None

        return {"lr": lr, "total_norm": total_norm}

    def _update_lr(self, optimizer: Optimizer, lr: float):
        # update lr in place
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    def _clip_grad(self, optimizer: Optimizer, scaler: Optional[GradScaler] = None):
        # clip grads by norm in place
        if scaler is not None:
            # unscale the gradients of optimizer's assigned params in-place
            scaler.unscale_(optimizer)

        params = [p for group in optimizer.param_groups for p in group["params"]]
        total_norm = clip_grad_norm_(params, max_norm=self.max_grad_norm).item()
        return total_norm


def cosine_lr_schedule(
    step: int,
    *,
    base_lr: float,
    total_steps: int,
    do_warmup: bool = True,
    do_decay: bool = True,
    warmup_fraction: float = 0.1,
    min_lr_fraction: float = 0.05,
):
    """
    Get learning rate for current step according to a linear warmup + cosine decay
    schedule.

    Reference:
        https://github.com/karpathy/nanoGPT
    """
    warmup_steps = int(warmup_fraction * total_steps)
    min_lr = min_lr_fraction * base_lr

    # linear warmup
    if do_warmup and step < warmup_steps:
        lr = min_lr + (step / warmup_steps) * (base_lr - min_lr)
    # cosine decay
    elif do_decay:
        decay_ratio = min((step - warmup_steps) / (total_steps - warmup_steps), 1.0)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        lr = min_lr + coeff * (base_lr - min_lr)
    else:
        lr = base_lr
    return lr


def create_optimizer(
    model: nn.Module,
    no_decay_keys: Optional[List[str]] = None,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    betas: Tuple[float, float] = (0.9, 0.95),
):
    """
    Create an AdamW optimizer with weight decay and no decay param groups.
    """
    decay_params = []
    no_decay_params = []
    no_decay_keys = set(no_decay_keys)

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name in no_decay_keys:
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=betas)
    return optimizer


def get_no_decay_keys(model: nn.Module) -> List[str]:
    """
    Return a list of parameter names that should not be weight decayed.

    Don't decay biases, layernorms, embeddings, or special tokens. A combination of
    what's done in timm and nanoGPT.
    """
    keys = [
        name
        for name, p in model.named_parameters()
        if name.endswith(("bias", "embed", "token")) or ".norm" in name
    ]
    return keys


def set_requires_grad(
    model: nn.Module,
    patterns: List[str],
    requires_grad: bool = False,
) -> List[str]:
    """
    Set the requires_grad flag for all parameters matching a pattern. Returns a list of
    updated parameter names.
    """
    updated = []
    for name, p in model.named_parameters():
        for pattern in patterns:
            if fnmatch.fnmatch(name, pattern):
                p.requires_grad_(requires_grad)
                updated.append(name)
                break

    return updated


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    strict: bool = True,
    load_opt_state: bool = True,
):
    """
    Load a checkpoint in place. Returns a tuple of the start epoch, best metric, and a
    named tuple of (missing_keys, unexpected_keys).
    """
    state = torch.load(checkpoint_path, map_location=device)
    # Remove prefix from DDP if present
    model_state = {
        k.removeprefix("_orig_mod.module."): v for k, v in state["model"].items()
    }
    bad_keys = model.load_state_dict(model_state, strict=strict)
    if bad_keys.missing_keys:
        logging.warning("Missing checkpoint keys:\n%s", bad_keys.missing_keys)
    if bad_keys.unexpected_keys:
        logging.warning("Unexpected checkpoint keys:\n%s", bad_keys.unexpected_keys)

    if load_opt_state:
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"]
        best_metric = state["metric"]
    else:
        start_epoch = 0
        best_metric = float("inf")
    return start_epoch, best_metric, bad_keys


def save_checkpoint(
    epoch: int,
    metric: float,
    is_best: bool,
    model: Union[DDP, nn.Module],
    optimizer: torch.optim.Optimizer,
    out_dir: Union[str, Path],
    max_checkpoints: int = 5,
):
    """
    Save a checkpoint to a directory.
    """
    if isinstance(model, DDP):
        model = model.module

    state = {
        "epoch": epoch,
        "metric": metric,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    ckpt_dir = Path(out_dir) / "checkpoints"
    ckpt_path = ckpt_dir / f"ckpt-{epoch:04d}.pt"
    ckpt_path.parent.mkdir(exist_ok=True)
    torch.save(state, ckpt_path)

    all_ckpts = sorted(ckpt_dir.glob("ckpt-[0-9]*.pt"))
    for p in all_ckpts[:-max_checkpoints]:
        p.unlink()

    ckpt_last = ckpt_dir / "ckpt-last.pt"
    ckpt_last.unlink(missing_ok=True)
    ckpt_last.symlink_to(ckpt_path.name)

    if is_best:
        ckpt_best = ckpt_dir / "ckpt-best.pt"
        torch.save(state, ckpt_best)
