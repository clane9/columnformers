import fnmatch
import math
import tempfile
from io import FileIO
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle

Shapes = List[Tuple[int, ...]]


class FeatureExtractor:
    """
    Extract intermediate activations from torch model.

    Example::

        extractor = FeatureExtractor(model, layers=["blocks.11"])
        # dictionary layers -> tensors
        output, features = extractor(data)
    """

    def __init__(self, model: nn.Module, layers: List[str], detach: bool = True):
        self.model = model
        self.layers = self.expand_layers(model, layers)
        self.detach = detach

        self._features: Dict[str, Tensor] = {}
        self._handles: Dict[str, RemovableHandle] = {}

        for layer in self.layers:
            self._handles[layer] = self._record(
                model, layer, self._features, detach=detach
            )

    def __call__(self, *args: Any, **kwargs: Any) -> Tuple[Any, Dict[str, Tensor]]:
        """
        Call model and return model output plus intermediate features.
        """
        # Get last recorded features if called with no args
        if len(args) == len(kwargs) == 0:
            return None, self._features.copy()

        output, *rest = self.model(*args, **kwargs)
        return output, self._features.copy()

    @staticmethod
    def _record(
        model: nn.Module,
        layer: str,
        features: Dict[str, Tensor],
        detach: bool = True,
    ):
        def hook(mod: nn.Module, input: Tuple[Tensor, ...], output: Tensor):
            if detach:
                output = output.detach()
            features[layer] = output

        mod = model.get_submodule(layer)
        handle = mod.register_forward_hook(hook)
        return handle

    def __del__(self):
        # TODO: _handles may not be defined if theres an error in __init__
        for handle in self._handles.values():
            handle.remove()

    @staticmethod
    def expand_layers(model: nn.Module, layers: List[str]) -> List[str]:
        """
        Get all layers in `model` matching the list of layer names and/or glob
        patterns in `layers`.
        """
        all_layers = [name for name, _ in model.named_modules() if len(name) > 0]
        all_layers_set = set(all_layers)
        special_chars = set("[]*?")

        expanded = []
        for layer in layers:
            if special_chars.isdisjoint(layer):
                if layer not in all_layers_set:
                    raise ValueError(f"Layer {layer} not in model")
                expanded.append(layer)
            else:
                matched = fnmatch.filter(all_layers, layer)
                if len(matched) == 0:
                    raise ValueError(f"Pattern {layer} didn't match any layers")
                expanded.extend(matched)
        return expanded


class H5Writer:
    """
    Asynchronous h5 data writer.

    Example::

        writer = H5Writer("test.h5")
        with writer as writer:
            writer.create_dataset("test/A", shape=(10000, 100), dtype="float32")

            for batch in batches:
                # returns immediately and keeps track of offset
                writer.put("test/A", batch)

    """

    def __init__(
        self,
        path: Union[str, Path],
        overwrite: bool = False,
        maxsize: int = 8,
    ):
        self.path = Path(path)
        self.overwrite = overwrite
        self.maxsize = maxsize

        self._is_open = False
        self._tmp: Optional[FileIO] = None
        self._f: Optional[h5py.File] = None
        self._q: Optional[Queue] = None
        self._offsets: Optional[Dict[str, int]] = None
        self._shapes: Optional[Dict[str, Tuple[int, ...]]] = None

    def create_dataset(
        self, name: str, shape: Tuple[int, ...], dtype: Optional[Any] = None
    ):
        """
        Create an h5 "dataset" inside the file.
        """
        assert self._is_open, "Writer not open"
        self._f.create_dataset(name, shape=shape, dtype=dtype)
        self._offsets[name] = 0
        self._shapes[name] = shape

    def put(
        self,
        name: str,
        values: np.ndarray,
        offset: Optional[int] = None,
    ):
        """
        Put new data into a dataset (asynchronously).
        """
        assert self._is_open, "Writer not open"
        if name not in self._offsets:
            raise ValueError(f"Dataset {name} not yet created")

        if offset is None:
            offset = self._offsets[name]
        shape = self._shapes[name]
        if offset + len(values) > shape[0]:
            raise RuntimeError(f"Values too big for dataset {name}")
        if values.shape[1:] != shape[1:]:
            raise RuntimeError(f"Values don't match shape of dataset {name}")

        self._q.put((name, values, offset))
        self._offsets[name] += len(values)

    def _open(self):
        if self.path.exists():
            if self.overwrite:
                self.path.unlink()
            else:
                raise FileExistsError(f"Feature path {self.path} already exists")

        self._tmp = tempfile.NamedTemporaryFile(
            mode="wb",
            dir=self.path.parent,
            prefix=".tmp-",
            suffix=self.path.suffix,
            delete=False,
        )
        self._f = h5py.File(self._tmp, mode="w")
        self._q = Queue(maxsize=self.maxsize)
        self._offsets = {}
        self._shapes = {}
        self._t = Thread(
            target=self._worker, kwargs={"f": self._f, "q": self._q}, daemon=True
        )
        self._t.start()
        self._is_open = True

    def _close(self, join: bool = True):
        if not self._is_open:
            return
        if join:
            self._q.put(None)
            self._q.join()
            self._t.join()
        self._f.close()
        self._tmp.close()

    @staticmethod
    def _worker(f: h5py.File, q: Queue):
        while True:
            item = q.get()
            if item is None:
                q.task_done()
                break

            name, values, offset = item
            end = offset + len(values)
            dataset = f[name]
            dataset[offset:end] = values
            q.task_done()

    def __enter__(self) -> "H5Writer":
        self._open()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._close(join=(exc_type is None))
        if exc_type is None:
            Path(self._tmp.name).rename(self.path)
        else:
            Path(self._tmp.name).unlink(missing_ok=True)


@torch.no_grad()
def process_features(
    feat: torch.Tensor,
    reduction: Optional[str] = "pool",
    pool_size: Optional[int] = 7,
    flatten: bool = False,
) -> torch.Tensor:
    """
    Reshape and optionally reduce and/or flatten features from linear, conv, or
    transformer layers.

    Args:
        feat: linear (N, D), conv (N, C, H, W) or transformer (N, T, D) features
        reduction: one of

            - mean: average over all tokens/patches/pixels.
            - pool: adaptive average pool height and width to a target size.
              Note that for transformers with an initial class token, the class
              token is excluded from pooling and prepended to the reduced
              feature output.
            - cls: pluck just the class token (from transformers that have one).
              (Note: alias of "mean" for conv features.)
            - none: keep all features

        pool_size: adaptive average pooling size when reduction is "pool".
            (Note: if `reduction` is `"pool"` and `pool_size` is `None`, the
            result is the same as `reduction="none"`.)
        flatten: flatten feature dimensions at the end

    Returns:
        A feature tensor of shape (N, S, M) or (N, S * M) if `flatten` is `True`.
    """

    if feat.ndim not in {2, 3, 4}:
        raise ValueError(f"Unsupported feature shape {feat.shape}")
    if reduction not in {"mean", "pool", "cls", "none", None}:
        raise ValueError(f"Invalid reduction {reduction}")

    # linear ouput (n, d)
    if feat.ndim == 2:
        # (n, 1, d)
        feat = feat.unsqueeze(1)

    # conv layer output (n, c, h, w)
    elif feat.ndim == 4:
        if reduction == "pool":
            output_size = pool_size
        elif reduction in {"none", None}:
            output_size = None
        else:
            # NOTE: letting cls be an alias for mean with conv features. Might
            # be a bit surprising.
            output_size = 1

        if output_size > 0:
            feat = F.adaptive_avg_pool2d(feat, output_size=output_size)
        # (n, c, h * w)
        feat = feat.flatten(start_dim=2)
        # (n, h * w, c)
        feat = feat.transpose(1, 2)

    # transformer layer output (n, t, d)
    elif feat.ndim == 3:
        n, t, d = feat.shape
        # assume we have an initial cls token if number of tokens is a square + 1
        has_cls = is_square(t - 1)

        if reduction == "mean":
            # (n, 1, d)
            feat = feat.mean(dim=1, keepdim=True)

        elif reduction == "pool" and pool_size > 0:
            if not (has_cls or is_square(t)):
                raise ValueError("Expected a square or square + 1 number of tokens")
            if has_cls:
                cls_feat = feat[:, :1, :]
                feat = feat[:, 1:, :]

            p = math.isqrt(feat.size(1))
            # (n, p, p, d)
            feat = feat.reshape(n, p, p, d)
            # (n, d, p, p)
            feat = torch.permute(feat, (0, 3, 1, 2))
            # (n, d, os, os)
            feat = F.adaptive_avg_pool2d(feat, output_size=pool_size)
            # (n, d, os * os)
            feat = feat.flatten(start_dim=2)
            # (n, os * os, c)
            feat = feat.transpose(1, 2)

            if has_cls:
                feat = torch.cat([cls_feat, feat], dim=1)

        elif reduction == "cls":
            if not has_cls:
                raise ValueError("cls reduction expects a square + 1 number of tokens")
            # (n, 1, d)
            feat = feat[:, :1, :]

    if flatten:
        feat = feat.flatten(start_dim=1)
    return feat


def is_square(x: int) -> bool:
    """
    Check if an integer is square.
    """
    return x == math.isqrt(x) ** 2
