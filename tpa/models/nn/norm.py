from typing import Optional

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm


__all__ = ["LayerNorm2d", "build_norm", "set_norm_eps"]


class LayerNorm2d(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x - torch.mean(x, dim=1, keepdim=True)
        out = out / torch.sqrt(torch.mean(out * out, dim=1, keepdim=True) + self.eps)
        if self.elementwise_affine:
            out = out * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return out


REGISTERED_NORM_DICT = {
    "bn2d": nn.BatchNorm2d,
    "ln": nn.LayerNorm,
    "ln2d": LayerNorm2d,
    "none": nn.Identity,
}


def build_norm(
    name: str = "bn2d", num_features: int = None, **kwargs
) -> Optional[nn.Module]:
    if name in ["ln", "ln2d"]:
        kwargs["normalized_shape"] = num_features
    else:
        kwargs["num_features"] = num_features

    if name in REGISTERED_NORM_DICT:
        return REGISTERED_NORM_DICT[name](**kwargs)
    else:
        return None


def set_norm_eps(model: nn.Module, eps: Optional[float] = None) -> None:
    for m in model.modules():
        if isinstance(m, (nn.LayerNorm, _BatchNorm)):
            if eps is not None:
                m.eps = eps
            else:
                m.eps = 1e-5
