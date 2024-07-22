import fnmatch
from typing import Any, Dict, List, Tuple

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

        output = self.model(*args, **kwargs)
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
