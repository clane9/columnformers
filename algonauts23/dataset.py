"""
Utilities for loading the Algonauts fMRI data, images, and ROIs.
"""

import json
import re
from collections import defaultdict
from copy import deepcopy
from glob import glob
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from algonauts23 import ALGONAUTS_DIR, SUBS
from algonauts23.utils import generate_splits, seed_hash


class AlgonautsSample(NamedTuple):
    image: Optional[Union[Image.Image, torch.Tensor]] = None
    activity: Optional[torch.Tensor] = None
    annotations: Optional[List[Dict[str, Any]]] = None


class AlgonautsDataset(Dataset[AlgonautsSample]):
    """
    Algonauts 2023 dataset.

    Args:
        sub: subject (subj01, ..., subj08)
        split: data split ("train", "val", "testval", "training-official",
            "test-official"). Note that "train", "val", and "testval" are fixed
            derived splits based on "training-official".
        root: path to root Algonauts2023 directory
        transform: torchvision transform to apply to scene images
        with_activity: load fMRI activity (if available)
        with_annotation: load COCO annotations (if available)
        mmap: memorymap the fMRI data

    Example::

        dataset = AlgonautsDataset(sub="subj01", split="train")
        image, activity, annotations = dataset[0]
    """

    OFFICIAL_SPLITS = ("training-official", "test-official")
    DERIVED_SPLITS = {
        "train": ["training-official", 0.85],
        "val": ["training-official", 0.1],
        "testval": ["training-official", 0.05],
    }
    DERIVED_SPLIT_SEED = 2023

    def __init__(
        self,
        sub: str,
        split: str,
        root: Union[str, Path] = ALGONAUTS_DIR,
        transform: Optional[Callable] = None,
        with_image: bool = True,
        with_activity: bool = True,
        with_annotation: bool = True,
        mmap: bool = True,
        device: Optional[torch.device] = None,
    ):
        assert (
            split in self.OFFICIAL_SPLITS or split in self.DERIVED_SPLITS
        ), f"Invalid split {split}"
        assert sub in SUBS, f"Invalid sub {sub}"
        assert with_image or with_activity, "Loading images or activity is required"

        self.root = Path(root)
        self.sub = sub
        self.split = split
        self.transform = transform
        self.with_image = with_image
        self.with_activity = with_activity
        self.with_annotation = with_annotation
        self.mmap = mmap
        self.device = device

        # get official dataset split (without the -official)
        if split in self.DERIVED_SPLITS:
            self.parent_split, _ = self.DERIVED_SPLITS[split]
            orig_split = self.parent_split
        else:
            self.parent_split = None
            orig_split = split
        orig_split = orig_split.split("-")[0]
        split_dir = self.root / sub / f"{orig_split}_split"

        self.image_list = sorted(
            glob(str(split_dir / f"{orig_split}_images" / "*.png"))
        )

        # fMRI activity
        if with_activity and split != "test-official":
            self.fmri_lh = np.load(
                split_dir / f"{orig_split}_fmri" / f"lh_{orig_split}_fmri.npy",
                mmap_mode=("r" if mmap else None),
            )
            self.fmri_rh = np.load(
                split_dir / f"{orig_split}_fmri" / f"rh_{orig_split}_fmri.npy",
                mmap_mode=("r" if mmap else None),
            )

            if not mmap and device is not None:
                self.fmri_lh = torch.as_tensor(
                    self.fmri_lh, dtype=torch.float32, device=device
                )
                self.fmri_rh = torch.as_tensor(
                    self.fmri_rh, dtype=torch.float32, device=device
                )
        else:
            self.fmri_lh = self.fmri_rh = None

        # COCO annotations (resampled)
        instances_path = self.root / f"instances_{orig_split}.json"
        if with_annotation and instances_path.exists():
            annotations = defaultdict(list)
            with instances_path.open() as f:
                instances = json.load(f)
            for annot in instances["annotations"]:
                annotations[annot["nsd_image_id"]].append(annot)
            self.annotations = annotations
            self.category_map = {
                cat["id"]: cat["name"] for cat in instances["categories"]
            }
        else:
            self.annotations = None
            self.category_map = None

        # Load indices for derived splits
        total_num_samples = len(self.image_list)
        self.indices = self._load_indices(total_num_samples)

        if self.indices is None:
            self.num_samples = total_num_samples
        else:
            self.num_samples = len(self.indices)

    def _load_indices(self, num_samples: int) -> Optional[np.ndarray]:
        if self.split not in self.DERIVED_SPLITS:
            return None

        indices_path = (
            self.root / self.sub / "derived_splits" / f"{self.split}_indices.npy"
        )
        if indices_path.exists():
            return np.load(indices_path)

        names_sizes = [
            (name, size)
            for name, (parent, size) in self.DERIVED_SPLITS.items()
            if parent == self.parent_split
        ]
        names, sizes = zip(*names_sizes)
        idx = names.index(self.split)

        # Generate a unique seed depending on the parent split, subject ID, and
        # shared seed. This way, randomness is uncorrelated across subjects.
        seed = seed_hash(
            "generate_splits",
            self.parent_split,
            self.sub,
            self.DERIVED_SPLIT_SEED,
        )
        indices = generate_splits(num_samples, sizes, seed)[idx]

        indices_path.parent.mkdir(exist_ok=True)
        np.save(indices_path, indices)
        return indices

    def __getitem__(self, idx: int) -> AlgonautsSample:
        if self.indices is not None:
            idx = self.indices[idx]

        image_path = Path(self.image_list[idx])

        if self.with_image:
            image = pil_loader(image_path)
            if self.transform is not None:
                image = self.transform(image)
        else:
            image = torch.zeros((0,), dtype=torch.float32)

        if self.with_activity and self.fmri_lh is not None:
            if torch.is_tensor(self.fmri_lh):
                act_lh = self.fmri_lh[idx]
                act_rh = self.fmri_rh[idx]
                activity = torch.cat([act_lh, act_rh])
            else:
                act_lh = np.asarray(self.fmri_lh[idx])
                act_rh = np.asarray(self.fmri_rh[idx])
                activity = np.concatenate([act_lh, act_rh])
                activity = torch.as_tensor(activity, dtype=torch.float32)
        else:
            activity = torch.zeros((0,), dtype=torch.float32)

        if self.with_annotation:
            annotations = self.get_annotations(image_path)
        else:
            annotations = []

        return AlgonautsSample(image, activity, annotations)

    def get_annotations(self, image_path: Union[str, Path]) -> List[Dict[str, Any]]:
        if not self.annotations:
            return []
        nsd_id = self.get_nsd_id(image_path)
        annotations = deepcopy(self.annotations.get(nsd_id, []))
        return annotations

    def get_image_path(self, idx: int) -> Path:
        if self.indices is not None:
            idx = self.indices[idx]
        return Path(self.image_list[idx])

    @staticmethod
    def get_nsd_id(image_path: Union[str, Path]) -> int:
        image_path = Path(image_path)
        return int(re.search("nsd-([0-9]+)", image_path.name).group(1))

    def get_all_activity(self) -> np.ndarray:
        if self.fmri_lh is None:
            raise RuntimeError("No fMRI activity available/loaded")
        if torch.is_tensor(self.fmri_lh):
            fmri_lh = self.fmri_lh.cpu().numpy()
            fmri_rh = self.fmri_rh.cpu().numpy()
        else:
            fmri_lh = np.asarray(self.fmri_lh)
            fmri_rh = np.asarray(self.fmri_rh)
        activity = np.concatenate([fmri_lh, fmri_rh], axis=1)
        if self.indices is not None:
            activity = activity[self.indices]
        return activity

    def get_activity_dim(self, hemi: Optional[str] = None) -> Optional[int]:
        if self.fmri_lh is None:
            return None

        if hemi == "lh":
            dim = self.fmri_lh.shape[1]
        elif hemi == "rh":
            dim = self.fmri_rh.shape[1]
        else:
            dim = self.fmri_lh.shape[1] + self.fmri_rh.shape[1]
        return dim

    def __len__(self):
        return self.num_samples


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")
