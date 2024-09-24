"""
A script for extracting features from pretrained timm models
"""

import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
import yaml
from timm.utils import AverageMeter, random_seed
from torch import Tensor
from torch.utils.data import DataLoader
from transformers.hf_argparser import HfArg, HfArgumentParser

from columnformers import utils as ut
from columnformers.data import create_dataset, create_loader, list_datasets
from columnformers.inspection.features import (
    FeatureExtractor,
    H5Writer,
    process_features,
)
from columnformers.models import create_model, list_models
from columnformers.models.columnformer import AttnMode, MlpMode, NormMode
from columnformers.train import _get_enum_values, parse_csv, to_device

logging.basicConfig(
    format="[%(levelname)s %(asctime)s]: %(message)s", level=logging.INFO
)
torch.backends.cudnn.benchmark = True

SEED = 42


@dataclass
class Args:
    layers: List[str] = HfArg(help="list of layer names to extract")
    split: str = HfArg(default="validation", help="which data split to extract")
    pool_size: Optional[int] = HfArg(default=None, help="adaptive average pool size")
    overwrite: bool = HfArg(default=False, help="overwrite pre-existing results")

    # Model
    model: str = HfArg(
        default="vision_transformer_tiny_patch16_128",
        help=f"model ({', '.join(list_models())})",
    )
    attn_mode: Optional[str] = HfArg(
        default=None,
        help=f"attention mode ({', '.join(_get_enum_values(AttnMode))})",
    )
    mlp_mode: Optional[str] = HfArg(
        default=None,
        help=f"mlp mode ({', '.join(_get_enum_values(MlpMode))})",
    )
    norm_mode: Optional[str] = HfArg(
        default=None,
        help=f"norm mode ({', '.join(_get_enum_values(NormMode))})",
    )
    num_heads: Optional[int] = HfArg(
        aliases=["--nh"], default=None, help="number of attention heads"
    )
    mlp_ratio: Optional[str] = HfArg(
        aliases=["--mlpr"],
        default=None,
        help="mlp ratio. can be a single value or a list of values, "
        "e.g. '4', '4,4,4,2,2,2'",
    )
    skip_attn: Optional[bool] = HfArg(
        aliases=["--skip"], default=None, help="include attention skip connection"
    )
    attn_bias: Optional[bool] = HfArg(
        aliases=["--attnb"], default=None, help="use learned attention bias"
    )
    attn_head_bias: Optional[bool] = HfArg(
        aliases=["--attnhb"], default=None, help="use learned attention bias per head"
    )
    qk_head_dim: Optional[int] = HfArg(
        aliases=["--qkd"], default=None, help="query and key head dimension"
    )
    no_vp: Optional[bool] = HfArg(
        aliases=["--novp"], default=None, help="don't use value and projection"
    )
    num_experts: Optional[str] = HfArg(
        default=None,
        help="number of experts for MoE MLP. can be a single value or a list of values, "
        "e.g. '2', '1,1,2,2,4,4'",
    )
    pool_stages: Optional[str] = HfArg(
        default=None,
        help="stages to apply pooling, branching, and double blocks in quadformer, "
        "e.g. '2,4'",
    )
    mlp_conserve: Optional[bool] = HfArg(
        default=None,
        help="Divide params by num experts "
        "`expert_params = dim * mlp_ratio / num_experts`",
    )
    init_local_attn: Optional[bool] = HfArg(
        aliases=["--initloc"], default=None, help="initialize with local attention bias"
    )
    depth_offset: int = HfArg(
        aliases=["--offset"], default=2.0, help="distance offset between layers"
    )
    global_pool: str = HfArg(
        aliases=["--pool"], default="avg", help="global pooling mode (avg, spatial)"
    )
    pos_embed: Optional[bool] = HfArg(
        aliases=["--pos"], default=None, help="use position embedding"
    )
    time_embed: Optional[bool] = HfArg(
        aliases=["--time_embed"], default=None, help="use time embedding"
    )
    drop_rate: float = HfArg(aliases=["--dr"], default=0.0, help="head dropout rate")
    proj_drop_rate: float = HfArg(
        aliases=["--pdr"], default=0.0, help="projection dropout rate"
    )
    attn_drop_rate: float = HfArg(
        aliases=["--adr"], default=0.0, help="attention dropout rate"
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
    workers: int = HfArg(aliases=["-j"], default=0, help="data loading workers")
    prefetch: bool = HfArg(default=True, help="use cuda prefetching")
    in_memory: bool = HfArg(
        aliases=["--inmem"], default=True, help="keep dataset in memory"
    )
    # Optimization
    epochs: int = HfArg(default=100, help="number of epochs")
    batch_size: int = HfArg(
        aliases=["--bs"], default=256, help="batch size per replica"
    )
    # Paths
    # Features will be written to:
    #   out_dir / model
    out_dir: Path = HfArg(
        default=Path("features"), help="path to root output directory"
    )
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
    debug: bool = HfArg(default=False, help="quick debug mode")


def main(args: Args):
    start_time = time.monotonic()
    args_dict = ut.args_to_dict(args)
    random_seed(SEED)

    clust = ut.ClusterEnv(args.cuda)

    out_dir = args.out_dir / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.split}_features.h5"
    if out_path.exists():
        if not args.overwrite:
            logging.info(f"Output path {out_path} already exists; exiting")
            return
        out_path.unlink()

    logging.info("Starting feature extraction")
    logging.info("Args:\n%s", yaml.safe_dump(args_dict, sort_keys=False))
    logging.info(ut.get_sha())

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    logging.info(f"Extracting with a single process on {device}")

    logging.info(f"Loading split {args.split}")

    input_size = int(args.model.split("_")[-1])
    dataset = create_dataset(
        args.dataset,
        input_size=input_size,
        min_scale=args.crop_min_scale,
        hflip=args.hflip,
        color_jitter=args.color_jitter,
        keep_in_memory=args.in_memory,
    )
    num_classes = dataset["train"].features["label"].num_classes

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
    logging.info(f"\n\tnum samples: {len(ds)}\n")

    logging.info("Creating model %s", args.model)

    model = create_model(
        args.model,
        attn_mode=args.attn_mode,
        mlp_mode=args.mlp_mode,
        norm_mode=args.norm_mode,
        num_heads=args.num_heads,
        mlp_ratio=parse_csv(args.mlp_ratio, float),
        skip_attn=args.skip_attn,
        attn_bias=args.attn_bias,
        attn_head_bias=args.attn_head_bias,
        qk_head_dim=args.qk_head_dim,
        no_vp=args.no_vp,
        num_experts=parse_csv(args.num_experts),
        pool_stages=parse_csv(args.pool_stages, squeeze=False),
        mlp_conserve=args.mlp_conserve,
        init_local_attn=args.init_local_attn,
        depth_offset=args.depth_offset,
        num_classes=num_classes,
        global_pool=args.global_pool,
        pos_embed=args.pos_embed,
        time_embed=args.time_embed,
        drop_rate=args.drop_rate,
        proj_drop_rate=args.proj_drop_rate,
        attn_drop_rate=args.attn_drop_rate,
    )
    model: torch.nn.Module = model.to(device)
    logging.info(
        f"param count: {sum([m.numel() for m in model.parameters()])/1e6:.0f}M"
    )

    if args.checkpoint:
        optimizer = ut.create_optimizer(model)
        logging.info("Loading checkpoint: %s", args.checkpoint)
        ut.load_checkpoint(
            args.checkpoint,
            model,
            optimizer,
            device=clust.device,
            strict=args.strict_load,
            load_opt_state=not args.restart,
        )

    # Construct feature extractor
    # NOTE: there is also torchvision create_feature_extractor() we may want to
    # use instead.
    # https://pytorch.org/vision/stable/feature_extraction.html
    extractor = FeatureExtractor(model, args.layers)
    logging.info(
        "Extracting features from %d layers:\n%s",
        len(extractor.layers),
        extractor.layers,
    )

    logging.info(f"Extracting to {out_path}")
    extract_features(args, extractor, loaders[args.split], out_path, device)

    runtime = time.monotonic() - start_time
    logging.info(f"Done! runtime: {runtime:.2f}s")


@torch.no_grad()
def extract_features(
    args: Args,
    extractor: "FeatureExtractor",
    loader: DataLoader,
    out_path: Path,
    device: torch.device,
) -> Tensor:
    extractor.model.eval()

    num_samples = len(loader.dataset)
    writer = H5Writer(path=out_path, maxsize=8)
    batch_time_m = AverageMeter()

    end = time.time()
    last_idx = len(loader) - 1
    with writer as writer:
        for epoch in range(args.epochs):
            for batch_idx, batch in enumerate(loader):
                last_batch = batch_idx == last_idx
                batch = to_device(batch, device)
                images = batch["image"]

                # Each feature shape: (batch_size, ...)
                _, batch_features = extractor(images)

                for name, values in batch_features.items():
                    # Reshape features to (N, T, D) and optionally perform adaptive
                    # average pooling to the target size
                    if args.pool_size is not None:
                        values = process_features(
                            values, reduction="pool", pool_size=args.pool_size
                        )

                    values = values.cpu().numpy()
                    if epoch == 0 and batch_idx == 0:
                        shape = (args.epochs * num_samples,) + values.shape[1:]
                        writer.create_dataset(name, shape=shape, dtype=values.dtype)
                    writer.put(name, values)

                if args.cuda:
                    torch.cuda.synchronize()

                batch_time_m.update(time.time() - end)
                end = time.time()
                if last_batch or batch_idx % args.log_interval == 0:
                    logging.info(
                        f"Extract: [{epoch:>2d}/{args.epochs-1}][{batch_idx:>3d}/{last_idx}]  "
                        f"Time: {batch_time_m.val:.3f} ({batch_time_m.avg:.3f})"
                    )

                if args.debug:
                    break


if __name__ == "__main__":
    args: Args
    parser = HfArgumentParser(Args)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        # If we pass only one argument to the script and it's the path to a yaml file,
        # let's parse it to get our arguments.
        args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[1]))
    else:
        (args,) = parser.parse_args_into_dataclasses()
    main(args)
