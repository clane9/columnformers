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
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import wandb
import yaml
from fvcore.nn import FlopCountAnalysis
from matplotlib import pyplot as plt
from timm.utils import AverageMeter, random_seed, reduce_tensor
from torch.cuda.amp import GradScaler
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers.hf_argparser import HfArg, HfArgumentParser

from columnformers import utils as ut
from columnformers.data import create_dataset, create_loader, list_datasets
from columnformers.inspection import (
    Figure,
    Metric,
    create_figure,
    create_metric,
    list_figures,
    list_metrics,
)
from columnformers.models import create_model, list_models
from columnformers.tasks import ImageClassification, Task

np.set_printoptions(precision=3)


@dataclass
class Args:
    project: str = HfArg(default="columnformers", help="project name")
    # Model
    model: str = HfArg(
        default="vision_columnformer_r_tiny_patch16_128",
        help=f"model ({', '.join(list_models())})",
    )
    num_heads: Optional[int] = HfArg(
        aliases=["--nh"], default=None, help="number of attention heads"
    )
    mlp_ratio: Optional[float] = HfArg(
        aliases=["--mlpr"], default=None, help="mlp ratio"
    )
    untied: Optional[str] = HfArg(
        default=None,
        help="untied mode coded as str of 1 or 3 comma separated values for norm, attn, "
        "mlp. e.g. '1' or '1,0,0'",
    )
    skip_attn: Optional[bool] = HfArg(
        aliases=["--skip"], default=None, help="include attention skip connection"
    )
    attn_bias: Optional[bool] = HfArg(
        aliases=["--attnb"], default=None, help="use learned attention bias"
    )
    qk_head_dim: Optional[int] = HfArg(
        aliases=["--qkd"], default=None, help="query and key head dimension"
    )
    no_vp: Optional[bool] = HfArg(
        aliases=["--novp"], default=None, help="don't use value and projection"
    )
    init_local_attn: Optional[bool] = HfArg(
        aliases=["--initloc"], default=None, help="initialize with local attention bias"
    )
    global_pool: str = HfArg(
        aliases=["--pool"], default="avg", help="global pooling mode (avg, spatial)"
    )
    pos_embed: Optional[bool] = HfArg(
        aliases=["--pos"], default=None, help="use position embedding"
    )
    drop_rate: float = HfArg(aliases=["--dr"], default=0.0, help="head dropout rate")
    proj_drop_rate: float = HfArg(
        aliases=["--pdr"], default=0.0, help="projection dropout rate"
    )
    attn_drop_rate: float = HfArg(
        aliases=["--adr"], default=0.0, help="attention dropout rate"
    )
    wiring_lambd: float = HfArg(
        aliases=["--wlambd"], default=0.0, help="wiring length penalty"
    )
    # Dataset
    dataset: str = HfArg(
        aliases=["--ds"],
        default="imagenet-100",
        help=f"dataset ({', '.join(list_datasets())})",
    )
    crop_min_scale: float = HfArg(
        aliases=["--scale"], default=1.0, help="image random crop scale"
    )
    hflip: float = HfArg(default=0.0, help="hflip probability")
    color_jitter: Optional[float] = HfArg(
        aliases=["--jitter"], default=None, help="color jitter value"
    )
    workers: int = HfArg(aliases=["-j"], default=4, help="data loading workers")
    prefetch: bool = HfArg(default=True, help="use cuda prefetching")
    # Optimization
    epochs: int = HfArg(default=100, help="number of epochs")
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
    # Paths
    out_dir: str = HfArg(default="results", help="path to root output directory")
    prefix: Optional[str] = HfArg(default=None, help="experiment name prefix")
    name: Optional[str] = HfArg(default=None, help="full experiment name")
    desc: Optional[str] = HfArg(default=None, help="description to attach to run")
    # Figures and metrics
    figures: List[str] = HfArg(
        default_factory=lambda: ["image_grid", "attn_feat_corr"],
        help=f"figures to generate ({', '.join(list_figures())})",
    )
    metrics: List[str] = HfArg(
        default_factory=lambda: ["accuracy", "attn_entropy"],
        help=f"metrics to generate ({', '.join(list_metrics())})",
    )
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
        name = ut.get_exp_name(args.prefix, seed=name_seed)
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
    input_size = int(args.model.split("_")[-1])
    dataset = create_dataset(
        args.dataset,
        input_size=input_size,
        min_scale=args.crop_min_scale,
        hflip=args.hflip,
        color_jitter=args.color_jitter,
        keep_in_memory=clust.device.type == "cuda" and not args.debug,
    )
    num_classes = dataset["train"].features["label"].num_classes
    logging.info("Dataset: %s", dataset)
    logging.info("Input size: %dx%d", input_size)

    loaders = {}
    for split, ds in dataset.items():
        loaders[split] = create_loader(
            ds,
            shuffle=True,
            batch_size=args.batch_size,
            drop_last=True,
            num_workers=args.workers,
            distributed=clust.ddp,
            pin_memory=not args.prefetch,
            use_prefetcher=args.prefetch,
            device=clust.device,
        )

    # Model and task
    logging.info("Creating model: %s", args.model)
    model = create_model(
        args.model,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        untied=parse_untied(args.untied),
        skip_attn=args.skip_attn,
        attn_bias=args.attn_bias,
        qk_head_dim=args.qk_head_dim,
        no_vp=args.no_vp,
        init_local_attn=args.init_local_attn,
        num_classes=num_classes,
        global_pool=args.global_pool,
        pos_embed=args.pos_embed,
        drop_rate=args.drop_rate,
        proj_drop_rate=args.proj_drop_rate,
        attn_drop_rate=args.attn_drop_rate,
    )
    model: torch.nn.Module = model.to(clust.device)
    logging.info("%s", model)

    task = ImageClassification(wiring_lambd=args.wiring_lambd)
    logging.info("Task: %s", task)

    param_count = sum(p.numel() for p in model.parameters())
    flop_count = get_flops(model, loaders["validation"], clust.device)
    logging.info("Params: %.0fM, FLOPs: %.0fM", param_count / 1e6, flop_count / 1e6)
    if clust.master_process and args.wandb:
        wandb.log({"param_count": param_count, "flop_count": flop_count}, step=0)

    # Optimizer
    logging.info("Creating optimizer")
    no_decay_keys = ut.get_no_decay_keys(model)
    optimizer = ut.create_optimizer(
        model,
        no_decay_keys=no_decay_keys,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
    )
    epoch_steps = math.ceil(len(loaders["train"]) / args.grad_accum_steps)
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
    figure_builders = {name: create_figure(name) for name in args.figures}
    metric_builders = {name: create_metric(name) for name in args.metrics}
    logging.info("Figures: %s", figure_builders)
    logging.info("Metrics: %s", metric_builders)

    # Load checkpoint
    if args.checkpoint:
        logging.info("Loading checkpoint: %s", args.checkpoint)
        start_epoch, best_metric = ut.load_checkpoint(
            args.checkpoint,
            model,
            optimizer,
            device=clust.device,
            strict=args.strict_load,
            load_opt_state=not args.restart,
        )
    else:
        start_epoch = 0
        best_metric = float("inf")
    best_epoch = start_epoch

    if clust.ddp:
        model = DDP(model, device_ids=[clust.local_rank])

    # Compile after ddp for more optimizations
    if args.compile:
        assert hasattr(torch, "compile"), "PyTorch >= 2.0 required for torch.compile"
        logging.info("Compiling the model")
        model = torch.compile(model)

    for epoch in range(start_epoch, args.epochs):
        logging.info("Starting epoch %d", epoch)
        if clust.ddp:
            loaders["train"].sampler.set_epoch(epoch)

        train_one_epoch(
            args=args,
            epoch=epoch,
            model=model,
            task=task,
            train_loader=loaders["train"],
            optimizer=optimizer,
            lr_schedule=lr_schedule,
            clust=clust,
            autocast=autocast,
            scaler=scaler,
            figure_builders=figure_builders,
            metric_builders=metric_builders,
            out_dir=out_dir,
        )

        metric = validate(
            args=args,
            epoch=epoch,
            step=(epoch + 1) * epoch_steps,
            model=model,
            task=task,
            val_loader=loaders["validation"],
            clust=clust,
            figure_builders=figure_builders,
            metric_builders=metric_builders,
            out_dir=out_dir,
        )

        if clust.master_process and (
            epoch % args.checkpoint_interval == 0 or epoch + 1 == args.epochs
        ):
            ut.save_checkpoint(
                epoch=epoch,
                metric=metric,
                is_best=metric < best_metric,
                model=model,
                optimizer=optimizer,
                out_dir=out_dir,
                max_checkpoints=args.max_checkpoints,
            )

        if metric < best_metric:
            best_metric = metric
            best_epoch = epoch

        if args.debug:
            break

    if clust.master_process and args.wandb:
        wandb.log(
            {"score_last": metric, "score_best": best_metric},
            step=args.epochs * epoch_steps,
        )

    logging.info("Done! Run time: %.0fs", time.monotonic() - start_time)
    logging.info("*** Best metric: %.3f (epoch %d)", best_metric, best_epoch)

    if clust.ddp:
        destroy_process_group()


def train_one_epoch(
    *,
    args: Args,
    epoch: int,
    model: torch.nn.Module,
    task: Task,
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
    task.train()
    if clust.use_cuda:
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
    for batch_idx, batch in enumerate(train_loader):
        step = first_step + batch_idx // accum_steps
        is_last_batch = batch_idx + 1 == epoch_batches
        need_update = is_last_batch or (batch_idx + 1) % accum_steps == 0
        if batch_idx >= last_batch_idx_to_accum:
            accum_steps = last_accum_steps

        batch = to_device(batch, clust.device)
        batch_size = get_batch_size(batch)
        data_time = time.monotonic() - end

        # forward pass
        with autocast():
            loss, state = task.forward(model, batch)
        if clust.ddp:
            loss_item = reduce_tensor(loss.detach(), clust.world_size).item()
        else:
            loss_item = loss.item()
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
            metrics = {name: func(state) for name, func in metric_builders.items()}
            update_metrics(metrics, state)

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
                    wandb.log({"train": record}, step=step)

        # Restart timer for next iteration
        end = time.monotonic()

        if args.debug:
            break

    if clust.master_process and save_figures:
        paths = {}
        for name, func in figure_builders.items():
            fig = func(state)
            if fig is not None:
                path = out_dir / "figures" / name / f"{name}-{epoch:04d}-train.png"
                paths[name] = path

                path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(path, bbox_inches="tight")
                plt.close(fig)

        if args.wandb:
            images = {name: wandb.Image(str(path)) for name, path in paths.items()}
            wandb.log({"train": {"figures": images}}, step=step)


@torch.no_grad()
def validate(
    *,
    args: Args,
    epoch: int,
    step: int,
    model: torch.nn.Module,
    task: Task,
    val_loader: DataLoader,
    clust: ut.ClusterEnv,
    figure_builders: Dict[str, Figure],
    metric_builders: Dict[str, Metric],
    out_dir: Path,
) -> float:
    model.eval()
    task.eval()
    if clust.use_cuda:
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
    for batch_idx, batch in enumerate(val_loader):
        batch = to_device(batch, clust.device)
        batch_size = get_batch_size(batch)
        data_time = time.monotonic() - end

        loss, state = task.forward(model, batch)
        if clust.ddp:
            loss_item = reduce_tensor(loss.detach(), clust.world_size).item()
        else:
            loss_item = loss.item()

        metrics = {name: func(state) for name, func in metric_builders.items()}
        update_metrics(metrics, state)

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
            wandb.log({"val": record}, step=step)

    if clust.master_process and save_figures:
        paths = {}
        for name, func in figure_builders.items():
            fig = func(state)
            if fig is not None:
                path = out_dir / "figures" / name / f"{name}-{epoch:04d}-val.png"
                paths[name] = path

                fig.savefig(path, bbox_inches="tight")
                plt.close(fig)

        if args.wandb:
            images = {name: wandb.Image(str(path)) for name, path in paths.items()}
            wandb.log({"val": {"figures": images}}, step=step)

    return loss_m.avg


def parse_untied(untied: Optional[str]):
    if untied is not None:
        untied = tuple([bool(int(val) for val in untied.strip().split(","))])
        if len(untied) == 1:
            untied = untied[0]
    return untied


@torch.no_grad()
def get_flops(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    batch = next(iter(loader))
    x = batch["image"][:1].to(device)
    flops = FlopCountAnalysis(model, x)
    return flops.total()


def to_device(batch: Dict[str, torch.Tensor], device: torch.device):
    batch = batch.copy()
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    return batch


def get_batch_size(batch: Dict[str, torch.Tensor]):
    key = next(iter(batch))
    return len(batch[key])


def update_metrics(metrics: Dict[str, Any], state: Dict[str, torch.Tensor]):
    for k, v in state.items():
        if is_scalar(v):
            metrics[k] = v.detach().item()


def is_scalar(value: torch.Tensor):
    return len(value.shape) == 0


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
