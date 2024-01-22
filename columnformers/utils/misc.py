"""
Misc utils. Copied/hacked together from various sources.
"""

import hashlib
import inspect
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch

from .slug import random_slug


class ClusterEnv:
    """
    Holds information about the current cluster environment.
    """

    def __init__(self, use_cuda: bool = True):
        use_cuda = use_cuda and torch.cuda.is_available()
        ddp = int(os.environ.get("RANK", -1)) != -1

        if ddp:
            assert use_cuda, "Distributed training requires CUDA"
            rank = int(os.environ["RANK"])
            local_rank = int(os.environ["LOCAL_RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            device = torch.device(f"cuda:{local_rank}")
        else:
            rank = local_rank = 0
            world_size = 1
            device = torch.device("cuda" if use_cuda else "cpu")

        self.use_cuda = use_cuda
        self.ddp = ddp
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = device

    @property
    def master_process(self) -> bool:
        return self.rank == 0

    def __repr__(self):
        return (
            f"Cluster(use_cuda={self.use_cuda}, ddp={self.ddp}, "
            f"rank={self.rank}, local_rank={self.local_rank}, "
            f"world_size={self.world_size}, device={self.device})"
        )


def seed_hash(*args):
    """
    Derive an integer hash from all args, for use as a random seed.

    Copied from:
        https://github.com/facebookresearch/DomainBed/blob/main/domainbed/lib/misc.py
    """
    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2**31)


def setup_logging(
    level: str = "INFO",
    path: Optional[Path] = None,
    stdout: bool = True,
    rank: Optional[int] = None,
):
    """
    Setup root logger.
    """
    if rank is not None:
        fmt = f"[%(levelname)s %(asctime)s {rank:>3d}]: %(message)s"
    else:
        fmt = "[%(levelname)s %(asctime)s]: %(message)s"
    formatter = logging.Formatter(fmt, datefmt="%y-%m-%d %H:%M:%S")

    logger = logging.getLogger()
    logger.setLevel(level)
    # clean up any pre-existing handlers
    for h in logger.handlers:
        logger.removeHandler(h)
    logger.root.handlers = []

    if stdout:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(level)
        logger.addHandler(stream_handler)

    if path:
        file_handler = logging.FileHandler(path, mode="a")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)


def get_sha():
    """
    Get the current commit hash

    Copied from:
        https://github.com/facebookresearch/dino/blob/main/utils.py
    """
    cwd = Path(__file__).parent.parent.absolute()

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode("ascii").strip()

    sha = "N/A"
    diff = "clean"
    branch = "N/A"
    try:
        sha = _run(["git", "rev-parse", "HEAD"])
        subprocess.check_output(["git", "diff"], cwd=cwd)
        diff = _run(["git", "diff-index", "HEAD"])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


def get_exp_name(seed: int, prefix: Optional[str] = None):
    """
    Generate a unique experiment name based on a prefix and a random seed.

    Example::
        >>> get_exp_name("my-experiment", 123)
        >>> "202309011000-my-experiment-clumsy-cricket"
    """
    name = datetime.now().strftime("%y%m%d%H%M%S")
    if prefix:
        name = name + "-" + prefix
    name = name + "-" + random_slug(seed=seed)
    return name


def filter_kwargs(func: Callable, kwargs: Dict[str, Any]):
    """
    Filter unused extra kwargs. Returns filtered kwargs and a list of extra args.
    """
    allowed_args = set(inspect.getfullargspec(func).args)
    extra_args = []
    kwargs = kwargs.copy()
    for k in list(kwargs):
        if k not in allowed_args:
            kwargs.pop(k)
            extra_args.append(k)
    return kwargs, extra_args
