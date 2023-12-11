from typing import Callable, Dict, List, Optional, Union

import torch

_MODELS: Dict[str, Callable[..., torch.nn.Module]] = {}


def register_model(name_or_func: Union[Optional[str], Callable] = None):
    def _decorator(func: Callable):
        name = name_or_func if isinstance(name_or_func, str) else func.__name__
        if name in _MODELS:
            raise ValueError(f"Model {name} already registered")
        _MODELS[name] = func
        return func

    if isinstance(name_or_func, Callable):
        return _decorator(name_or_func)
    return _decorator


def create_model(name: str, **kwargs) -> torch.nn.Module:
    if name not in _MODELS:
        raise ValueError(f"Model {name} not registered")
    model = _MODELS[name](**kwargs)
    return model


def list_models() -> List[str]:
    return list(_MODELS)
