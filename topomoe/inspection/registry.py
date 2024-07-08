import warnings
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from matplotlib import figure

State = Dict[str, torch.Tensor]
Metric = Callable[[State], Dict[str, torch.Tensor]]
Figure = Callable[[State], Dict[str, figure.Figure]]

_METRICS: Dict[str, Callable[..., Metric]] = {}
_FIGURES: Dict[str, Callable[..., Figure]] = {}


def register_metric(name_or_func: Union[Optional[str], Callable] = None):
    def _decorator(func: Callable):
        if isinstance(name_or_func, str):
            name = name_or_func
        elif hasattr(name_or_func, "name"):
            name = name_or_func.name
        else:
            name = func.__name__
        if name in _METRICS:
            warnings.warn(f"Metric {name} already registered; overwriting", UserWarning)
        _METRICS[name] = func
        return func

    if isinstance(name_or_func, Callable):
        return _decorator(name_or_func)
    return _decorator


def create_metric(name: str, **kwargs) -> Metric:
    if name not in _METRICS:
        raise ValueError(f"Metric {name} not registered")
    metric = _METRICS[name](**kwargs)
    return metric


def create_metrics(
    cfg: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Metric]:
    cfg = cfg or {name: {} for name in list_metrics()}
    metrics = {name: create_metric(name, **kwargs) for name, kwargs in cfg.items()}
    return metrics


def list_metrics() -> List[str]:
    return list(_METRICS)


def register_figure(name_or_func: Union[Optional[str], Callable] = None):
    def _decorator(func: Callable):
        if isinstance(name_or_func, str):
            name = name_or_func
        elif hasattr(name_or_func, "name"):
            name = name_or_func.name
        else:
            name = func.__name__
        if name in _FIGURES:
            warnings.warn(f"Figure {name} already registered; overwriting", UserWarning)
        _FIGURES[name] = func
        return func

    if isinstance(name_or_func, Callable):
        return _decorator(name_or_func)
    return _decorator


def create_figure(name: str, **kwargs) -> Figure:
    if name not in _FIGURES:
        raise ValueError(f"Figure {name} not registered")
    fig = _FIGURES[name](**kwargs)
    return fig


def create_figures(
    cfg: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Figure]:
    cfg = cfg or {name: {} for name in list_figures()}
    figs = {name: create_figure(name, **kwargs) for name, kwargs in cfg.items()}
    return figs


def list_figures() -> List[str]:
    return list(_FIGURES)
