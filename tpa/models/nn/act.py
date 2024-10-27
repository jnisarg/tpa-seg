from functools import partial
from typing import Optional

import torch.nn as nn


__all__ = ["build_act"]


REGISTERED_ACT_DICT = {
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "hswish": nn.Hardswish,
    "swish": nn.SiLU,
    "gelu": partial(nn.GELU, approximate="tanh"),
    "none": nn.Identity,
}


def build_act(name: str, **kwargs) -> Optional[nn.Module]:
    if name in REGISTERED_ACT_DICT:
        return REGISTERED_ACT_DICT[name](**kwargs)
    else:
        return None
