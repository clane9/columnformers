from functools import partial
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from datasets import (
    ClassLabel,
    Dataset,
    DatasetDict,
    Features,
    Image,
    interleave_datasets,
    load_dataset,
)
from PIL import Image as I
from timm.data.transforms_factory import create_transform

IMAGENET100_REPO_ID = "clane9/imagenet-100"


def imagenet100(keep_in_memory: bool = False) -> DatasetDict:
    """
    Load the ImageNet-100 dataset from the HuggingFace Hub.
    """
    dataset = load_dataset(IMAGENET100_REPO_ID, keep_in_memory=keep_in_memory)
    return dataset


def micro_imagenet100(
    train_samples: int = 6000,
    train_repeats: int = 20,
    seed: int = 42,
    keep_in_memory: bool = False,
) -> DatasetDict:
    """
    Load the Micro ImageNet-100 dataset that includes a small train set.
    """
    dataset = imagenet100()
    dataset["train"] = dataset["train"].train_test_split(
        train_size=train_samples,
        stratify_by_column="label",
        seed=seed,
    )["train"]

    if keep_in_memory:
        for split, ds in dataset.items():
            dataset[split] = Dataset.from_dict(ds.to_dict(), features=ds.features)

    if train_repeats > 1:
        dataset["train"] = interleave_datasets(
            [dataset["train"] for _ in range(train_repeats)]
        )
    return dataset


def debug100(
    train_samples: int = 256,
    val_samples: int = 256,
    img_size: int = 128,
    seed: int = 42,
    keep_in_memory: bool = False,
):
    def generator(num_samples: int, seed: int):
        rng = np.random.default_rng(seed)

        for _ in range(num_samples):
            img = rng.integers(256, size=(img_size, img_size, 3), dtype=np.uint8)
            img = I.fromarray(img)
            label = rng.integers(100)
            yield {"image": img, "label": label}

    features = Features({"image": Image(), "label": ClassLabel(num_classes=100)})
    dataset = {
        "train": Dataset.from_generator(
            partial(generator, num_samples=train_samples, seed=seed),
            features=features,
            keep_in_memory=keep_in_memory,
        ),
        "validation": Dataset.from_generator(
            partial(generator, num_samples=val_samples, seed=seed + 1),
            features=features,
            keep_in_memory=keep_in_memory,
        ),
    }
    return DatasetDict(dataset)


DATASETS_REGISTRY = {
    "imagenet-100": imagenet100,
    "micro-imagenet-100": micro_imagenet100,
    "debug-100": debug100,
}


def list_datasets() -> List[str]:
    return list(DATASETS_REGISTRY.keys())


def create_dataset(
    dataset_name: str,
    input_size: int = 128,
    min_scale: float = 0.4,
    hflip: float = 0.5,
    color_jitter: Optional[float] = 0.4,
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
            color_jitter=None if color_jitter == 0 else color_jitter,
            interpolation=interpolation,
        )
        transform = _get_batch_transform(image_transform)
        ds.set_transform(transform)
    return dsets


def _get_batch_transform(image_transform: Callable[[I.Image], torch.Tensor]):
    def transform(
        batch: Dict[str, List[Union[I.Image, Any]]],
    ) -> Dict[str, torch.Tensor]:
        batch["image"] = [image_transform(img.convert("RGB")) for img in batch["image"]]
        return batch

    return transform
