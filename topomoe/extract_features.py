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
from timm.data import DEFAULT_CROP_PCT, create_dataset, create_loader
from timm.utils import AverageMeter, random_seed
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.hf_argparser import HfArg, HfArgumentParser

from topomoe import utils as ut
from topomoe.inspection.features import FeatureExtractor, H5Writer, process_features
from topomoe.models import create_model, list_models
from topomoe.train import Args as TrainArgs
from topomoe.train import get_num_classes, load_dataset_in_memory

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
    add_pos: bool = HfArg(
        default=False, help="add stage position embedding to pooled input"
    )
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
    workers: int = HfArg(aliases=["-j"], default=0, help="data loading workers")
    prefetch: bool = HfArg(default=True, help="use cuda prefetching")
    in_memory: bool = HfArg(
        aliases=["--inmem"], default=False, help="keep dataset in memory"
    )
    # Optimization
    epochs: int = HfArg(default=100, help="number of epochs")
    batch_size: int = HfArg(
        aliases=["--bs"], default=256, help="batch size per replica"
    )
    # Paths
    # Features will be written to out_dir / model
    out_dir: Path = HfArg(
        default=Path("topomoe_features"), help="path to root output directory"
    )
    checkpoint_path: Optional[Path] = HfArg(
        aliases=["--ckpt"], default=None, help="checkpoint to load"
    )
    model_args_path: Optional[Path] = HfArg(
        aliases=["--args"], default=None, help="args to load for trained model"
    )
    strict_load: bool = HfArg(aliases=["--strict"], default=True, help="strict loading")
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

    dataset = create_dataset(
        args.dataset,
        root=args.data_dir,
        split=args.split,
        is_training=True,
        download=args.download,
        batch_size=args.batch_size,
    )
    if args.in_memory:
        logging.info("Loading dataset into memory")
        load_dataset_in_memory(dataset)

    input_size = int(args.model.split("_")[-1])
    input_size = (3, input_size, input_size)

    loader = create_loader(
        dataset,
        input_size=input_size,
        batch_size=args.batch_size,
        is_training=True if args.split == "train" else False,
        scale=args.scale if args.split == "train" else None,
        ratio=args.ratio if args.split == "train" else None,
        hflip=args.hflip if args.split == "train" else 0.5,
        crop_pct=args.crop_pct if args.split == "validation" else None,
        color_jitter=args.color_jitter if args.split == "train" else 0.4,
        interpolation="bicubic",
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        distributed=clust.ddp,
        device=clust.device,
        use_prefetcher=args.prefetch,
    )

    logging.info(f"\n\tnum samples: {len(dataset)}\n")

    logging.info("Creating model %s", args.model)

    num_classes = get_num_classes(dataset)

    if args.checkpoint_path and args.model_args_path:
        model_arg_parser = HfArgumentParser(TrainArgs)
        (model_args,) = model_arg_parser.parse_yaml_file(yaml_file=args.model_args_path)
    else:
        model_args = args

    model = create_model(
        model_args.model,
        num_heads=model_args.num_heads,
        mlp_ratio=model_args.mlp_ratio,
        num_experts=model_args.num_experts,
        mlp_conserve=model_args.mlp_conserve,
        num_classes=num_classes,
        drop_rate=model_args.drop_rate,
        proj_drop_rate=model_args.proj_drop_rate,
        attn_drop_rate=model_args.attn_drop_rate,
        # add_pos=model_args.add_pos,
        wiring_lambd=model_args.wiring_lambd,
        wiring_sigma=model_args.wiring_sigma,
    )
    model: torch.nn.Module = model.to(device)
    logging.info(
        f"param count: {sum([m.numel() for m in model.parameters()])/1e6:.0f}M"
    )

    if args.checkpoint_path:
        logging.info("Loading checkpoint: %s", args.checkpoint_path)
        ckpt = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model"])

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
    extract_features(args, extractor, loader, out_path, device)

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
            for batch_idx, (input, target) in tqdm(
                enumerate(loader), total=len(loader)
            ):
                last_batch = batch_idx == last_idx

                if not args.prefetch:
                    input, target = input.to(device), target.to(device)

                # Each feature shape: (batch_size, ...)
                _, batch_features = extractor(input)

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
