from contextlib import suppress
from functools import partial
from typing import Dict, Generator, Optional

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Prefetcher:
    """
    A wrapper around a DataLoader that loads the next batch to the device
    asynchronously. The loader should yield dicts mapping strings to tensors. Copied
    from timm with some simplification.

    References:
        https://github.com/huggingface/pytorch-image-models/blob/v0.9.12/timm/data/loader.py#L76
    """

    def __init__(self, loader: DataLoader, device: Optional[torch.device] = None):
        self.loader = loader
        self.device = DEFAULT_DEVICE if device is None else device
        self.is_cuda = self.device.type == "cuda"

    def __iter__(self) -> Generator[Dict[str, torch.Tensor], None, None]:
        batch: Optional[Dict[str, torch.Tensor]] = None
        next_batch: Dict[str, torch.Tensor]
        first = True

        if self.is_cuda:
            stream = torch.cuda.Stream()
            stream_context = partial(torch.cuda.stream, stream=stream)
        else:
            stream = None
            stream_context = suppress

        for next_batch in self.loader:
            with stream_context():
                for k, v in next_batch.items():
                    if isinstance(v, torch.Tensor):
                        next_batch[k] = v.to(device=self.device, non_blocking=True)

            if not first:
                yield batch
            else:
                first = False

            if stream is not None:
                torch.cuda.current_stream().wait_stream(stream)

            batch = next_batch

        yield batch

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset


def create_loader(
    dataset: Dataset,
    shuffle: bool = True,
    batch_size: int = 512,
    num_workers: int = 0,
    distributed: bool = False,
    pin_memory: bool = False,
    drop_last: bool = True,
    use_prefetcher: bool = True,
    device: Optional[torch.device] = None,
) -> Dict[str, DataLoader]:
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
    else:
        sampler = RandomSampler(dataset) if shuffle else None

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    if use_prefetcher:
        loader = Prefetcher(loader, device)
    return loader
