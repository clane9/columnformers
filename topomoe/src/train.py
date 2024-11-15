import json
import logging
import math
import shutil
import sys
import time
from argparse import Namespace
from collections import defaultdict
from contextlib import suppress
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import wandb
import yaml
from datasets import Dataset
from fvcore.nn import FlopCountAnalysis
from matplotlib import pyplot as plt
from timm.data import DEFAULT_CROP_PCT, ImageDataset, create_dataset, create_loader
from timm.data.readers.reader_hfds import ReaderHfds
from timm.utils import AverageMeter, random_seed, reduce_tensor
from torch.cuda.amp import GradScaler
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers.hf_argparser import HfArg, HfArgumentParser

from topomoe.src import utils as ut
from topomoe.src.inspection import Figure, Metric, create_figures, create_metrics
from topomoe.src.models import create_model, list_models

np.set_printoptions(precision=3)
plt.switch_backend("Agg")


@dataclass
class Args:
    name: Optional[str] = HfArg(default=None, help="experiment name")
    project: str = HfArg(default="columnformers", help="project name")
    desc: Optional[str] = HfArg(default=None, help="description to attach to run")
    out_dir: str = HfArg(default="results", help="path to root output directory")
    # Model
    model: str = HfArg(
        default="topomoe_tiny_2s_patch16_128",
        help=f"model ({', '.join(list_models())})",
    )
    num_classes: Optional[int] = HfArg(default=None, help="number of classes")
    num_heads: Optional[int] = HfArg(
        aliases=["--nh"], default=None, help="number of attention heads"
    )
    mlp_ratio: Optional[List[float]] = HfArg(
        aliases=["--mlpr"],
        default=None,
        help="mlp ratio. can be a single value or a list of values, "
        "e.g. --mlpr 4.0, --mlpr 4.0 4.0 4.0 2.0 2.0 2.0",
    )
    num_experts: Optional[List[int]] = HfArg(
        default=None,
        help="number of experts. can be a single value or a list of values",
    )
    mlp_conserve: Optional[bool] = HfArg(
        default=None,
        help="Divide params by num experts "
        "`expert_params = dim * mlp_ratio / num_experts`",
    )
    drop_rate: float = HfArg(aliases=["--dr"], default=0.0, help="head dropout rate")
    proj_drop_rate: float = HfArg(
        aliases=["--pdr"], default=0.0, help="projection dropout rate"
    )
    attn_drop_rate: float = HfArg(
        aliases=["--adr"], default=0.0, help="attention dropout rate"
    )
    static_pool: bool = HfArg(default=False, help="use static position based pooling")
    wiring_lambd: float = HfArg(
        aliases=["--wlambd"], default=0.0, help="wiring length penalty"
    )
    wiring_sigma: float = HfArg(
        aliases=["--wsigma"], default=2.0, help="wiring length radius stdev"
    )
    # Dataset
    dataset: str = HfArg(
        default="hfds/clane9/imagenet-100", help="timm-compatible dataset name"
    )
    data_dir: Optional[str] = HfArg(default=None, help="dataset directory")
    download: bool = HfArg(default=True, help="download dataset")
    train_split: str = HfArg(default="train", help="name of training split")
    val_split: str = HfArg(default="validation", help="name of val split")
    train_num_samples: Optional[int] = HfArg(
        default=None,
        help="Manually specify num samples in train split, for IterableDatasets",
    )
    val_num_samples: Optional[int] = HfArg(
        default=None,
        help="Manually specify num samples in validation split, for IterableDatasets",
    )
    epoch_repeats: int = HfArg(
        default=0, help="number of times to repeat dataset epoch per train epoch"
    )
    scale: List[float] = HfArg(
        default_factory=lambda: [0.5, 1.0], help="image random crop scale"
    )
    ratio: List[float] = HfArg(
        default_factory=lambda: [3 / 4, 4 / 3], help="image random crop ratio"
    )
    hflip: float = HfArg(default=0.0, help="hflip probability")
    color_jitter: Optional[float] = HfArg(
        aliases=["--jitter"], default=None, help="color jitter value"
    )
    crop_pct: float = HfArg(default=DEFAULT_CROP_PCT, help="eval crop pct")
    workers: int = HfArg(aliases=["-j"], default=4, help="data loading workers")
    prefetch: bool = HfArg(default=True, help="use cuda prefetching")
    in_memory: bool = HfArg(
        aliases=["--inmem"], default=False, help="keep dataset in memory"
    )
    # Optimization
    epochs: int = HfArg(default=2, help="number of epochs")
    batch_size: int = HfArg(
        aliases=["--bs"], default=256, help="batch size per replica"
    )
    lr: float = HfArg(default=6e-4, help="learning rate")
    decay_lr: bool = HfArg(default=True, help="decay learning rate")
    warmup_fraction: float = HfArg(default=0.1, help="fraction of warmup steps")
    min_lr_fraction: float = HfArg(
        default=0.05, help="minimum lr as a fraction of max lr"
    )
    weight_decay: float = HfArg(aliases=["--wd"], default=0.05, help="weight decay")
    beta1: float = HfArg(default=0.9, help="AdamW beta1")
    beta2: float = HfArg(default=0.95, help="AdamW beta2")
    grad_accum_steps: int = HfArg(
        aliases=["--accum"], default=1, help="number of gradient accumulation steps"
    )
    clip_grad: Optional[float] = HfArg(default=1.0, help="gradient norm clipping")
    # Figures and metrics
    figures_cfg: Optional[str] = HfArg(default=None, help="path to yaml figures config")
    metrics_cfg: Optional[str] = HfArg(default=None, help="path to yaml metrics config")
    save_figures: bool = HfArg(default=True, help="save figures")
    # Logistics
    checkpoint: Optional[str] = HfArg(
        aliases=["--ckpt"], default=None, help="checkpoint to load"
    )
    strict_load: bool = HfArg(aliases=["--strict"], default=True, help="strict loading")
    restart: bool = HfArg(
        default=False, help="Restart training rather than resume from checkpoint"
    )
    cuda: bool = HfArg(default=True, help="use cuda")
    amp: bool = HfArg(default=False, help="use AMP")
    amp_dtype: str = HfArg(default="float16", help="AMP dtype (float16, bfloat16)")
    compile: bool = HfArg(default=False, help="use torch compile")
    overwrite: bool = HfArg(default=False, help="overwrite pre-existing results")
    wandb: bool = HfArg(default=False, help="log to wandb")
    log_interval: int = HfArg(
        aliases=["--logint"], default=10, help="log every n steps"
    )
    figure_interval: int = HfArg(
        aliases=["--figint"], default=5, help="save figures every n epochs"
    )
    checkpoint_interval: int = HfArg(
        aliases=["--ckptint"], default=20, help="save checkpoint every n epochs"
    )
    max_checkpoints: int = HfArg(
        aliases=["--maxckpt"], default=2, help="number of recent checkpoints to keep"
    )
    debug: bool = HfArg(default=False, help="quick debug mode")
    seed: int = HfArg(default=42, help="random seed")


def main(args: Args):
    start_time = time.monotonic()
    random_seed(args.seed)

    # Device and distributed training setup
    clust = ut.ClusterEnv(args.cuda)
    if clust.ddp:
        init_process_group(backend="nccl")
        torch.cuda.set_device(clust.device)

    # Output naming
    commit_sha = ut.get_sha()
    if args.name is None:
        name_seed = ut.seed_hash(commit_sha, json.dumps(args.__dict__))
        name = ut.get_exp_name(seed=name_seed)
    else:
        name = args.name
    out_dir = Path(args.out_dir) / args.project
    out_dir = out_dir / name

    # Creating output dir
    overwritten = False
    if clust.master_process and out_dir.exists():
        if args.overwrite:
            overwritten = True
            shutil.rmtree(out_dir)
        else:
            raise FileExistsError(f"Output directory {out_dir} already exists")
    if clust.master_process:
        out_dir.mkdir(parents=True)
    else:
        while not out_dir.exists():
            time.sleep(0.1)

    log_path = out_dir / "logs" / f"log-{clust.rank:02d}.txt"
    log_path.parent.mkdir(exist_ok=True)
    ut.setup_logging(path=log_path, stdout=clust.master_process, rank=clust.rank)

    # Wandb setup
    if clust.master_process and args.wandb:
        wandb.init(project=args.project, name=name, config=args.__dict__)

    # Initial logging
    logging.info("Starting training: %s/%s", args.project, name)
    logging.info("Args:\n%s", yaml.safe_dump(args.__dict__, sort_keys=False))
    logging.info(commit_sha)

    logging.info("Writing to %s", out_dir)
    if overwritten:
        logging.warning("Overwriting previous results")
    if clust.master_process:
        with (out_dir / "args.yaml").open("w") as f:
            yaml.safe_dump(args.__dict__, f, sort_keys=False)

    logging.info("Running on: %s", clust)

    # AMP setup
    if args.amp:
        logging.info(
            f"Running in mixed precision ({args.amp_dtype}) with native PyTorch AMP"
        )

        amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
        autocast = partial(
            torch.autocast, device_type=clust.device.type, dtype=amp_dtype
        )
        # bfloat16 does not need loss scaler, following timm
        scaler = GradScaler() if args.amp_dtype == "float16" else None
    else:
        autocast = suppress
        scaler = None

    # Dataset
    logging.info("Loading dataset %s", args.dataset)
    dataset_train = create_dataset(
        args.dataset,
        root=args.data_dir,
        split=args.train_split,
        is_training=True,
        download=args.download,
        batch_size=args.batch_size,
        num_samples=args.train_num_samples,
        repeats=args.epoch_repeats,
    )
    dataset_eval = create_dataset(
        args.dataset,
        root=args.data_dir,
        split=args.val_split,
        is_training=False,
        download=args.download,
        batch_size=args.batch_size,
        num_samples=args.val_num_samples,
    )
    if args.in_memory:
        logging.info("Loading dataset into memory")
        load_dataset_in_memory(dataset_train)
        load_dataset_in_memory(dataset_eval)

    input_size = int(args.model.split("_")[-1])
    input_size = (3, input_size, input_size)
    loader_train = create_loader(
        dataset_train,
        input_size=input_size,
        batch_size=args.batch_size,
        is_training=True,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        color_jitter=args.color_jitter,
        interpolation="bicubic",
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        distributed=clust.ddp,
        device=clust.device,
        use_prefetcher=args.prefetch,
    )
    loader_eval = create_loader(
        dataset_eval,
        input_size=input_size,
        batch_size=args.batch_size,
        is_training=False,
        crop_pct=args.crop_pct,
        interpolation="bicubic",
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        distributed=clust.ddp,
        device=clust.device,
        use_prefetcher=args.prefetch,
    )

    # Model and task
    logging.info("Creating model: %s", args.model)
    num_classes = args.num_classes or get_num_classes(dataset_train)
    model = create_model(
        args.model,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        num_experts=args.num_experts,
        mlp_conserve=args.mlp_conserve,
        num_classes=num_classes,
        drop_rate=args.drop_rate,
        proj_drop_rate=args.proj_drop_rate,
        attn_drop_rate=args.attn_drop_rate,
        static_pool=args.static_pool,
        wiring_lambd=args.wiring_lambd,
        wiring_sigma=args.wiring_sigma,
    )
    model: torch.nn.Module = model.to(clust.device)
    logging.info("%s", model)

    param_count = sum(p.numel() for p in model.parameters())
    flop_count = get_flops(model, loader_eval, clust.device)
    counts = {"params (M)": param_count / 1e6, "flops (M)": flop_count / 1e6}
    if clust.master_process and args.wandb:
        wandb.log({f"counts.{k}": v for k, v in counts.items()}, step=0)
    logging.info("Params: %.0fM, FLOPs: %.0fM", param_count / 1e6, flop_count / 1e6)
    logging.info("Counts:\n%s", json.dumps(counts))

    # Optimizer
    logging.info("Creating optimizer")
    no_decay_keys = ut.collect_no_weight_decay(model)
    optimizer = ut.create_optimizer(
        model,
        no_decay_keys=no_decay_keys,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
    )
    epoch_steps = math.ceil(len(loader_train) / args.grad_accum_steps)
    lr_schedule = ut.CosineDecaySchedule(
        base_lr=args.lr,
        total_steps=args.epochs * epoch_steps,
        do_decay=args.decay_lr,
        warmup_fraction=args.warmup_fraction,
        min_lr_fraction=args.min_lr_fraction,
    )
    logging.info("%s", optimizer)
    logging.info("No decay keys:\n%s", no_decay_keys)
    logging.info("Steps per epoch: %d", epoch_steps)

    # Figures and metrics
    if args.figures_cfg is not None:
        with open(args.figures_cfg) as f:
            figures_cfg = yaml.safe_load(f)
    else:
        figures_cfg = None
    figure_builders = create_figures(figures_cfg)
    logging.info("Figures: %s", figure_builders)

    if args.metrics_cfg is not None:
        with open(args.metrics_cfg) as f:
            metrics_cfg = yaml.safe_load(f)
    else:
        metrics_cfg = None
    metric_builders = create_metrics(metrics_cfg)
    logging.info("Metrics: %s", metric_builders)

    # Load checkpoint
    if args.checkpoint:
        logging.info("Loading checkpoint: %s", args.checkpoint)
        start_epoch, best_loss = ut.load_checkpoint(
            args.checkpoint,
            model,
            optimizer,
            device=clust.device,
            strict=args.strict_load,
            load_opt_state=not args.restart,
        )
    else:
        start_epoch = 0
        best_loss = float("inf")
    best_epoch = start_epoch
    best_metrics = {}

    if clust.ddp:
        model = DDP(model, device_ids=[clust.local_rank])

    # Compile after ddp for more optimizations
    if args.compile:
        assert hasattr(torch, "compile"), "PyTorch >= 2.0 required for torch.compile"
        logging.info("Compiling the model")
        model = torch.compile(model)

    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(start_epoch, args.epochs):
        logging.info("Starting epoch %d", epoch)

        if hasattr(dataset_train, "set_epoch"):
            dataset_train.set_epoch(epoch)
        elif clust.ddp and hasattr(loader_train.sampler, "set_epoch"):
            loader_train.sampler.set_epoch(epoch)

        train_one_epoch(
            args=args,
            epoch=epoch,
            model=model,
            loss_fn=loss_fn,
            train_loader=loader_train,
            optimizer=optimizer,
            lr_schedule=lr_schedule,
            clust=clust,
            autocast=autocast,
            scaler=scaler,
            figure_builders=figure_builders,
            metric_builders=metric_builders,
            out_dir=out_dir,
        )

        loss, metrics = validate(
            args=args,
            epoch=epoch,
            step=(epoch + 1) * epoch_steps - 1,
            model=model,
            loss_fn=loss_fn,
            val_loader=loader_eval,
            clust=clust,
            figure_builders=figure_builders,
            metric_builders=metric_builders,
            out_dir=out_dir,
        )
        is_best = loss < best_loss

        if clust.master_process and (
            epoch % args.checkpoint_interval == 0 or epoch + 1 == args.epochs
        ):
            ut.save_checkpoint(
                epoch=epoch,
                loss=loss,
                is_best=is_best,
                model=model,
                optimizer=optimizer,
                out_dir=out_dir,
                max_checkpoints=args.max_checkpoints,
            )

        if is_best:
            best_loss = loss
            best_epoch = epoch
            best_metrics = metrics

        if args.debug:
            break

    if clust.master_process and args.wandb:
        last_step = args.epochs * epoch_steps
        wandb.log({f"last.{k}": v for k, v in metrics.items()}, step=last_step)
        wandb.log({f"best.{k}": v for k, v in best_metrics.items()}, step=last_step)
    logging.info("Last metrics:\n%s", json.dumps(metrics))
    logging.info("Best metrics:\n%s", json.dumps(best_metrics))

    logging.info("Done! Run time: %.0fs", time.monotonic() - start_time)
    logging.info("*** Best loss: %#.3g (epoch %d)", best_loss, best_epoch)

    if clust.ddp:
        destroy_process_group()


def train_one_epoch(
    *,
    args: Args,
    epoch: int,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_schedule: ut.LRSchedule,
    clust: ut.ClusterEnv,
    autocast: Callable,
    scaler: Optional[GradScaler],
    figure_builders: Dict[str, Figure],
    metric_builders: Dict[str, Metric],
    out_dir: Path,
):
    model.train()
    if clust.use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad()

    loss_m = AverageMeter()
    data_time_m = AverageMeter()
    step_time_m = AverageMeter()

    save_figures = args.save_figures and (
        epoch % args.figure_interval == 0 or epoch + 1 == args.epochs or args.debug
    )

    epoch_batches = len(train_loader)
    accum_steps = args.grad_accum_steps
    epoch_steps = math.ceil(epoch_batches / accum_steps)
    first_step = epoch * epoch_steps
    last_accum_steps = epoch_batches % accum_steps
    last_batch_idx_to_accum = epoch_batches - last_accum_steps

    end = time.monotonic()
    for batch_idx, (input, target) in enumerate(train_loader):
        step = first_step + batch_idx // accum_steps
        is_last_batch = batch_idx + 1 == epoch_batches
        need_update = is_last_batch or (batch_idx + 1) % accum_steps == 0
        if batch_idx >= last_batch_idx_to_accum:
            accum_steps = last_accum_steps

        if not args.prefetch:
            input, target = input.to(clust.device), target.to(clust.device)
        batch_size = input.size(0)
        data_time = time.monotonic() - end

        # forward pass
        with autocast():
            output, losses, state = model(input)
            losses["class_loss"] = loss_fn(output, target)
        loss = sum(losses.values())

        loss_item = to_item(loss, clust=clust)
        state = {"image": input, "target": target, "output": output, **state}

        if accum_steps > 1:
            loss = loss / accum_steps

        if math.isnan(loss_item) or math.isinf(loss_item):
            raise RuntimeError("NaN/Inf loss encountered on step %d; exiting", step)

        # update lr
        lr = lr_schedule(step)
        ut.update_lr_(optimizer, lr)

        # backward and optimization step
        total_norm = ut.backward_step(
            loss,
            optimizer,
            scaler=scaler,
            need_update=need_update,
            max_grad_norm=args.clip_grad,
        )

        # end of iteration timing
        if clust.use_cuda:
            torch.cuda.synchronize()
        step_time = time.monotonic() - end

        loss_m.update(loss_item, batch_size)
        data_time_m.update(data_time, batch_size)
        step_time_m.update(step_time, batch_size)

        if (
            (step % args.log_interval == 0 and need_update)
            or is_last_batch
            or args.debug
        ):
            losses_items = {k: to_item(v, clust=clust) for k, v in losses.items()}
            metrics = {
                k: v.item()
                for func in metric_builders.values()
                for k, v in func(state).items()
            }
            metrics = {**losses_items, **metrics}

            tput = (clust.world_size * args.batch_size) / step_time_m.avg
            if clust.use_cuda:
                alloc_mem_gb = torch.cuda.max_memory_allocated() / 1e9
                res_mem_gb = torch.cuda.max_memory_reserved() / 1e9
            else:
                alloc_mem_gb = res_mem_gb = 0.0

            logging.info(
                f"Train: {epoch:>3d} [{batch_idx:>3d}/{epoch_batches}][{step:>6d}]"
                f"  Loss: {loss_m.val:#.3g} ({loss_m.avg:#.3g})"
                f"  LR: {lr:.3e}"
                f"  Grad: {total_norm:.3e}"
                f"  Time: {data_time_m.avg:.3f},{step_time_m.avg:.3f} {tput:.0f}/s"
                f"  Mem: {alloc_mem_gb:.2f},{res_mem_gb:.2f} GB"
            )

            if clust.master_process:
                record = {
                    "step": step,
                    "epoch": epoch,
                    "loss": loss_m.val,
                    "lr": lr,
                    "grad": total_norm,
                    "data_time": data_time_m.avg,
                    "step_time": step_time_m.avg,
                    "tput": tput,
                    **metrics,
                }

                with (out_dir / "train_log.json").open("a") as f:
                    print(json.dumps(record), file=f)

                if args.wandb:
                    wandb.log({f"train.{k}": v for k, v in record.items()}, step=step)

        # Restart timer for next iteration
        end = time.monotonic()

        if args.debug:
            break

    if clust.master_process and save_figures:
        paths = {}
        for name, func in figure_builders.items():
            figs = func(state)
            for key, fig in figs.items():
                path = out_dir / "figures" / name / f"{key}-{epoch:04d}-train.png"
                paths[key] = path

                path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(path, bbox_inches="tight")
                plt.close(fig)

        if args.wandb:
            images = {f"figs.train.{k}": wandb.Image(str(p)) for k, p in paths.items()}
            wandb.log(images, step=step)


@torch.no_grad()
def validate(
    *,
    args: Args,
    epoch: int,
    step: int,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    val_loader: DataLoader,
    clust: ut.ClusterEnv,
    figure_builders: Dict[str, Figure],
    metric_builders: Dict[str, Metric],
    out_dir: Path,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    if clust.use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    loss_m = AverageMeter()
    data_time_m = AverageMeter()
    step_time_m = AverageMeter()

    metric_ms = defaultdict(AverageMeter)

    save_figures = args.save_figures and (
        epoch % args.figure_interval == 0 or epoch + 1 == args.epochs or args.debug
    )

    epoch_batches = len(val_loader)
    end = time.monotonic()
    for batch_idx, (input, target) in enumerate(val_loader):
        if not args.prefetch:
            input, target = input.to(clust.device), target.to(clust.device)
        batch_size = input.size(0)
        data_time = time.monotonic() - end

        # forward pass
        output, losses, state = model(input)
        losses["class_loss"] = loss_fn(output, target)
        loss = sum(losses.values())

        loss_item = to_item(loss, clust=clust)
        losses_items = {k: to_item(v, clust=clust) for k, v in losses.items()}
        state = {"image": input, "target": target, "output": output, **state}
        metrics = {
            k: v.item()
            for func in metric_builders.values()
            for k, v in func(state).items()
        }
        metrics = {**losses_items, **metrics}

        # end of iteration timing
        if clust.use_cuda:
            torch.cuda.synchronize()
        step_time = time.monotonic() - end

        loss_m.update(loss_item, batch_size)
        data_time_m.update(data_time, batch_size)
        step_time_m.update(step_time, batch_size)

        for name, val in metrics.items():
            metric_ms[name].update(val, batch_size)

        if (
            batch_idx % args.log_interval == 0
            or batch_idx + 1 == epoch_batches
            or args.debug
        ):
            tput = (clust.world_size * args.batch_size) / step_time_m.avg
            if clust.use_cuda:
                alloc_mem_gb = torch.cuda.max_memory_allocated() / 1e9
                res_mem_gb = torch.cuda.max_memory_reserved() / 1e9
            else:
                alloc_mem_gb = res_mem_gb = 0.0

            logging.info(
                f"Val: {epoch:>3d} [{batch_idx:>3d}/{epoch_batches}]"
                f"  Loss: {loss_m.val:#.3g} ({loss_m.avg:#.3g})"
                f"  Time: {data_time_m.avg:.3f},{step_time_m.avg:.3f} {tput:.0f}/s"
                f"  Mem: {alloc_mem_gb:.2f},{res_mem_gb:.2f} GB"
            )

        if args.debug:
            break

        # Reset timer
        end = time.monotonic()

    if clust.master_process:
        record = {
            "step": step,
            "epoch": epoch,
            "loss": loss_m.avg,
            "data_time": data_time_m.avg,
            "step_time": step_time_m.avg,
            "tput": tput,
            **{name: meter.avg for name, meter in metric_ms.items()},
        }

        with (out_dir / "val_log.json").open("a") as f:
            print(json.dumps(record), file=f)

        if args.wandb:
            wandb.log({f"val.{k}": v for k, v in record.items()}, step=step)

    if clust.master_process and save_figures:
        paths = {}
        for name, func in figure_builders.items():
            figs = func(state)
            for key, fig in figs.items():
                path = out_dir / "figures" / name / f"{key}-{epoch:04d}-val.png"
                paths[key] = path

                fig.savefig(path, bbox_inches="tight")
                plt.close(fig)

        if args.wandb:
            images = {f"figs.val.{k}": wandb.Image(str(p)) for k, p in paths.items()}
            wandb.log(images, step=step)

    metrics = {k: v.avg for k, v in metric_ms.items()}
    metrics = {"loss": loss_m.avg, **metrics}
    return loss_m.avg, metrics


def load_dataset_in_memory(dataset: ImageDataset):
    assert isinstance(dataset.reader, ReaderHfds)
    dataset.reader.dataset = Dataset.from_dict(
        dataset.reader.dataset.to_dict(),
        features=dataset.reader.dataset.features,
    )


def get_num_classes(dataset: ImageDataset):
    assert isinstance(dataset.reader, ReaderHfds)
    return dataset.reader.dataset.features["label"].num_classes


@torch.no_grad()
def get_flops(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    x, _ = next(iter(loader))
    x = x[:1].to(device)
    flops = FlopCountAnalysis(model, x)
    return flops.total()


def to_item(x: torch.Tensor, clust: ut.ClusterEnv) -> float:
    if clust.ddp:
        x = reduce_tensor(x.detach(), clust.world_size).item()
    else:
        x = x.detach().item()
    return x


if __name__ == "__main__":
    args: Args
    parser = HfArgumentParser(Args)
    if sys.argv[1].endswith(".yaml"):
        # If the first argument is a yaml file, parse it first to get default arguments.
        (args,) = parser.parse_yaml_file(yaml_file=sys.argv[1])

        # Treat any remaining args as overrides
        parsed = parser.parse_args(
            args=sys.argv[2:], namespace=Namespace(**asdict(args))
        )
        (args,) = parser.parse_dict(parsed.__dict__)
    else:
        (args,) = parser.parse_args_into_dataclasses()

    try:
        main(args)
    except Exception as exc:
        logging.error("Exited with exception", exc_info=exc)
        sys.exit(1)
