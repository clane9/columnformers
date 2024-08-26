"""
Misc utils.
"""

import hashlib
import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from torch.utils.data import Dataset


class ZipDataset(Dataset):
    """
    Torch dataset that zips several datasets.

    Note:
        Unlike builtin `zip`, this also flattens the items together into a
        single tuple.
    """

    def __init__(self, *datasets: Dataset):
        assert all(
            len(datasets[0]) == len(ds) for ds in datasets
        ), "Size mismatch between datasets"

        self.datasets = datasets

    def __getitem__(self, idx: int):
        return tuple(item for ds in self.datasets for item in self._as_tuple(ds[idx]))

    @staticmethod
    def _as_tuple(val):
        if not isinstance(val, tuple):
            val = (val,)
        return val

    def __len__(self):
        return len(self.datasets[0])


def args_to_dict(args: object):
    """
    Cast args fields to primitive types for serialization.
    """
    state = {}
    for k, v in args.__dict__.items():
        if isinstance(v, Path):
            v = str(v)
        state[k] = v
    return state


def seed_hash(*args):
    """
    Derive an integer hash from all args, for use as a random seed.

    Copied from:
    https://github.com/facebookresearch/DomainBed/blob/main/domainbed/lib/misc.py
    """
    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2**31)


def generate_splits(
    num_samples: int, split_sizes: List[float], seed: int
) -> List[np.ndarray]:
    """
    Generate reproducible data splits.

    Args:
        num_samples: number of samples
        split_sizes: fractional split sizes summing to one
        seed: random seed

    Returns:
        A list of split indices arrays
    """
    assert sum(split_sizes) == 1.0, "split_sizes must sum to 1"

    split_lengths = np.asarray(split_sizes) * num_samples
    split_ends = np.round(np.cumsum(split_lengths)).astype(int)
    split_starts = np.concatenate([[0], split_ends[:-1]])

    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_samples)

    splits = [
        np.sort(indices[start:end]) for start, end in zip(split_starts, split_ends)
    ]
    return splits


def label_to_one_hot(label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a categorical label to one-hot representation. Return the one-hot
    representation and the unique label values.
    """
    shape = label.shape
    label = label.flatten()
    uniq, label = np.unique(label, return_inverse=True)
    one_hot = np.zeros((len(label), len(uniq)))
    one_hot[np.arange(len(label)), label] = 1.0
    if len(shape) > 1:
        one_hot = one_hot.reshape(shape + (len(uniq),))
    return one_hot, uniq


def one_hot_to_label(
    one_hot: np.ndarray, uniq: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Convert a one-hot to categorical representation.
    """
    label = np.argmax(one_hot, axis=-1)
    if uniq is not None:
        label = uniq[label]
    return label


def setup_logging(out_dir: Optional[Path] = None, level: str = "INFO"):
    """
    Setup root logger.
    """
    fmt = "[%(levelname)s %(asctime)s %(lineno)4d]: %(message)s"
    formatter = logging.Formatter(fmt, datefmt="%y-%m-%d %H:%M:%S")

    logger = logging.getLogger()
    logger.setLevel(level)
    # clean up any pre-existing handlers
    for h in logger.handlers:
        logger.removeHandler(h)
    logger.root.handlers = []

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)
    logger.addHandler(stream_handler)

    if out_dir:
        log_path = out_dir / "log.txt"
        file_handler = logging.FileHandler(log_path, mode="a")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

    # Redefining the root logger is not strictly best practice.
    # https://stackoverflow.com/a/7430495
    # But I want the convenience to just call e.g. `logging.info()`.
    logging.root = logger  # type: ignore


def get_sha():
    """
    Get the current commit hash
    """
    # Copied from: https://github.com/facebookresearch/dino/blob/main/utils.py
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
