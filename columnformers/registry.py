from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

DEFAULT_HF_HUB_ORG = "clane9"
_MODELS: Dict[str, Callable[..., torch.nn.Module]] = {}
_CONFIGS: Dict[str, "Config"] = {}


@dataclass
class Config:
    name: str
    kwargs: Dict[str, Any] = field(default_factory=dict)
    defaults: Dict[str, Any] = field(default_factory=dict)
    has_weights: bool = False
    hf_hub_id: Optional[str] = f"{DEFAULT_HF_HUB_ORG}/"
    hf_hub_file: Optional[str] = "model.safetensors"

    @property
    def model_name(self):
        return self.name.split(".")[0]


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


def register_configs(cfgs: Dict[str, Dict[str, Any]]):
    new_configs = {}
    for name, cfg in cfgs.items():
        if name in _CONFIGS:
            raise ValueError(f"Config {name} already registered")
        new_configs[name] = Config(name=name, **cfg)
    _CONFIGS.update(new_configs)


def create_model(name: str, pretrained: bool = False, **kwargs) -> torch.nn.Module:
    cfg = _CONFIGS.get(name) or Config(name)
    if cfg.model_name not in _MODELS:
        raise KeyError(f"Model {cfg.model_name} not registered")
    if pretrained and not cfg.has_weights:
        raise ValueError(f"No weights available for model {name}")
    try:
        # Update user kwargs with (overridable) model defaults
        # Note that model kwargs cannot be overwritten
        kwargs = {**cfg.defaults, **kwargs}
        model = _MODELS[cfg.model_name](**cfg.kwargs, **kwargs)
    except TypeError as err:
        raise ValueError("Attempting to override model config kwargs") from err

    if pretrained:
        state_dict = load_weights(cfg)
        model.load_state_dict(state_dict)
    return model


def load_weights(cfg: Config) -> Dict[str, torch.Tensor]:
    if not cfg.has_weights:
        raise ValueError(f"No weights available for model {cfg.name}")

    hf_hub_id = cfg.hf_hub_id
    if hf_hub_id.endswith("/"):
        hf_hub_id = hf_hub_id + cfg.name

    local_path = hf_hub_download(repo_id=hf_hub_id, filename=cfg.hf_hub_file)

    suffix = Path(cfg.hf_hub_file).suffix
    if suffix == ".safetensors":
        state_dict = load_file(local_path)
    else:
        state_dict = torch.load(local_path, map_location="cpu")
    return state_dict


def list_models() -> List[str]:
    return list({**_MODELS, **_CONFIGS})


def list_pretrained() -> List[str]:
    return [name for name, cfg in _CONFIGS.items() if cfg.has_weights]
