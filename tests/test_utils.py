import logging

import pytest
import torch
from torch import nn

from columnformers.utils import collect_no_weight_decay


class Linear(nn.Linear):
    def no_weight_decay(self):
        return ["weight"]


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(5, 5),
            nn.Linear(5, 5),
            Linear(5, 5),
        )
        self.norm = nn.LayerNorm(5)
        self.proj = nn.Linear(5, 5)
        self.token = nn.Parameter(torch.zeros(5))

    def no_weight_decay(self):
        return ["token"]


def test_collect_no_weight_decay():
    model = Model()
    no_decay_list = collect_no_weight_decay(model)
    logging.info("No decay list: %s", no_decay_list)
    assert no_decay_list == [
        "token",
        "main.2.weight",
        "main.0.bias",
        "main.1.bias",
        "main.2.bias",
        "norm.weight",
        "norm.bias",
        "proj.bias",
    ]


if __name__ == "__main__":
    pytest.main([__file__])
