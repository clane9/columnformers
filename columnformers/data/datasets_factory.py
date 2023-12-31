from typing import Any, Callable, Dict, List, Union

import torch
from datasets import DatasetDict, load_dataset
from PIL import Image
from timm.data.transforms_factory import create_transform

IMAGENET100_REPO_ID = "clane9/imagenet-100"


def imagenet100(keep_in_memory: bool = False) -> DatasetDict:
    """
    Load the ImageNet-100 dataset from the HuggingFace Hub.
    """
    dataset = load_dataset(IMAGENET100_REPO_ID, keep_in_memory=keep_in_memory)
    return dataset


def micro_imagenet100(
    train_size: int = 6000,
    seed: int = 42,
    keep_in_memory: bool = False,
) -> DatasetDict:
    """
    Load the Micro ImageNet-100 dataset that includes a small train set.
    """
    dataset = imagenet100(keep_in_memory=keep_in_memory)
    dataset["train"] = dataset["train"].train_test_split(
        train_size=train_size,
        stratify_by_column="label",
        seed=seed,
    )["train"]
    return dataset


DATASETS_REGISTRY = {
    "imagenet-100": imagenet100,
    "micro-imagenet-100": micro_imagenet100,
}


def create_dataset(
    dataset_name: str,
    input_size: int = 128,
    min_scale: float = 0.4,
    hflip: float = 0.5,
    color_jitter: float = 0.4,
    interpolation: str = "bicubic",
    keep_in_memory: bool = False,
) -> DatasetDict:
    dsets: DatasetDict = DATASETS_REGISTRY[dataset_name](keep_in_memory=keep_in_memory)

    for split, ds in dsets.items():
        image_transform = create_transform(
            input_size=input_size,
            is_training=split == "train",
            scale=(min_scale, 1.0),
            hflip=hflip,
            color_jitter=color_jitter,
            interpolation=interpolation,
        )
        transform = _get_batch_transform(image_transform)
        ds.set_transform(transform)
    return dsets


def _get_batch_transform(image_transform: Callable[[Image.Image], torch.Tensor]):
    def transform(
        batch: Dict[str, List[Union[Image.Image, Any]]]
    ) -> Dict[str, torch.Tensor]:
        batch["image"] = [image_transform(img.convert("RGB")) for img in batch["image"]]
        return batch

    return transform
