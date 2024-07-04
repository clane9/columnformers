from typing import Any, Callable, Dict, List, Optional, Union

import torch
from matplotlib import figure

State = Dict[str, torch.Tensor]
FigureBuilder = Callable[[State], Dict[str, figure.Figure]]

_FIGURES: Dict[str, Callable[..., FigureBuilder]] = {}


def register_figure(name_or_func: Union[Optional[str], Callable] = None):
    def _decorator(func: Callable):
        name = name_or_func if isinstance(name_or_func, str) else func.__name__
        if name in _FIGURES:
            raise ValueError(f"Figure {name} already registered")
        _FIGURES[name] = func
        return func

    if isinstance(name_or_func, Callable):
        return _decorator(name_or_func)
    return _decorator


def create_figure(name: str, **kwargs) -> FigureBuilder:
    if name not in _FIGURES:
        raise ValueError(f"Figure {name} not registered")
    fig = _FIGURES[name](**kwargs)
    return fig


def create_figures(
    cfg: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, FigureBuilder]:
    cfg = cfg or {name: {} for name in list_figures()}
    figs = {name: create_figure(name, **kwargs) for name, kwargs in cfg.items()}
    return figs


def list_figures() -> List[str]:
    return list(_FIGURES)
