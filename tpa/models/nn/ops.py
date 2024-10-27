from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from tpa.models.nn.act import build_act
from tpa.models.nn.norm import build_norm
from tpa.models.utils import get_same_padding, list_sum, resize, val2list, val2tuple


__all__ = [
    "ConvLayer",
    "UpsampleLayer",
    "InterpolateConvUpsampleLayer",
    "LinearLayer",
    "IdentityLayer",
    "DSConv",
    "MBConv",
    "FusedMBConv",
    "ResBlock",
    "ResidualBlock",
    "DAGBlock",
    "OpSequential",
]


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        use_bias: bool = False,
        dropout: float = 0.0,
        act: Optional[str] = None,
        norm: Optional[str] = None,
    ):
        super(ConvLayer, self).__init__()

        padding = get_same_padding(kernel_size)
        padding *= dilation

        self.dropout = nn.Dropout(dropout, inplace=True) if dropout > 0.0 else None
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=use_bias,
        )
        self.norm = build_norm(norm, out_channels)
        self.act = build_act(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class UpsampleLayer(nn.Module):
    def __init__(
        self,
        mode: Literal["bilinear", "nearest", "bicubic"] = "bilinear",
        size: Optional[int | tuple[int, int] | list[int]] = None,
        factor=2,
        align_corners=False,
    ):
        super(UpsampleLayer, self).__init__()

        self.mode = mode
        self.size = val2list(size, 2) if size is not None else None
        self.factor = None if self.size is not None else factor
        self.align_corners = align_corners

    @torch.autocast(device_type="cuda", enabled=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            self.size is not None and tuple(x.shape[-2:]) == self.size
        ) or self.factor == 1:
            return x
        if x.dtype in [torch.float16, torch.bfloat16]:
            x = x.float()
        return resize(x, self.size, self.factor, self.mode, self.align_corners)


class InterpolateConvUpsampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        factor: int,
        mode: Literal["bilinear", "nearest", "bicubic"] = "nearest",
    ):
        super(InterpolateConvUpsampleLayer, self).__init__()

        self.factor = factor
        self.mode = mode
        self.conv = ConvLayer(
            in_channels,
            out_channels,
            kernel_size,
            use_bias=True,
            norm=None,
            act=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=self.factor, mode=self.mode)
        return self.conv(x)


class LinearLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.0,
        use_bias: bool = True,
        norm: Optional[str] = None,
        act: Optional[str] = None,
    ):
        super(LinearLayer, self).__init__()

        self.dropout = nn.Dropout(dropout, inplace=True) if dropout > 0.0 else None
        self.linear = nn.Linear(in_features, out_features, bias=use_bias)
        self.norm = build_norm(norm, out_features)
        self.act = build_act(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.linear(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class IdentityLayer(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class DSConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        use_bias: bool = False,
        norm=("bn2d", "bn2d"),
        act=("relu6", None),
    ):
        super(DSConv, self).__init__()

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act = val2tuple(act, 2)

        self.depth_conv = ConvLayer(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            groups=in_channels,
            use_bias=use_bias[0],
            norm=norm[0],
            act=act[0],
        )
        self.point_conv = ConvLayer(
            in_channels,
            out_channels,
            kernel_size=1,
            use_bias=use_bias[1],
            norm=norm[1],
            act=act[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class MBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        mid_channels: Optional[int] = None,
        expand_ratio: int = 6,
        use_bias: bool = False,
        norm=("bn2d", "bn2d", "bn2d"),
        act=("relu6", "relu6", None),
    ):
        super(MBConv, self).__init__()

        use_bias = val2tuple(use_bias, 3)
        norm = val2tuple(norm, 3)
        act = val2tuple(act, 3)

        mid_channels = (
            round(in_channels * expand_ratio) if mid_channels is None else mid_channels
        )

        self.inverted_conv = ConvLayer(
            in_channels,
            mid_channels,
            kernel_size=1,
            stride=1,
            use_bias=use_bias[0],
            norm=norm[0],
            act=act[0],
        )
        self.depth_conv = ConvLayer(
            mid_channels,
            mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=mid_channels,
            use_bias=use_bias[1],
            norm=norm[1],
            act=act[1],
        )
        self.point_conv = ConvLayer(
            mid_channels,
            out_channels,
            kernel_size=1,
            use_bias=use_bias[2],
            norm=norm[2],
            act=act[2],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inverted_conv(x)
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class FusedMBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        mid_channels: Optional[int] = None,
        expand_ratio: int = 6,
        groups: int = 1,
        use_bias: bool = False,
        norm=("bn2d", "bn2d"),
        act=("relu6", None),
    ):
        super(FusedMBConv, self).__init__()

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act = val2tuple(act, 2)

        mid_channels = (
            round(in_channels * expand_ratio) if mid_channels is None else mid_channels
        )

        self.spatial_conv = ConvLayer(
            in_channels,
            mid_channels,
            kernel_size,
            stride,
            groups=groups,
            use_bias=use_bias[0],
            norm=norm[0],
            act=act[0],
        )
        self.point_conv = ConvLayer(
            mid_channels,
            out_channels,
            kernel_size=1,
            use_bias=use_bias[1],
            norm=norm[1],
            act=act[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spatial_conv(x)
        x = self.point_conv(x)
        return x


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        mid_channels: Optional[int] = None,
        expand_ratio: int = 1,
        use_bias: bool = False,
        norm=("bn2d", "bn2d"),
        act=("relu6", None),
    ):
        super(ResBlock, self).__init__()

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act = val2tuple(act, 2)

        mid_channels = (
            round(in_channels * expand_ratio) if mid_channels is None else mid_channels
        )

        self.conv1 = ConvLayer(
            in_channels,
            mid_channels,
            kernel_size,
            stride,
            use_bias=use_bias[0],
            norm=norm[0],
            act=act[0],
        )
        self.conv2 = ConvLayer(
            mid_channels,
            out_channels,
            kernel_size,
            stride=1,
            use_bias=use_bias[1],
            norm=norm[1],
            act=act[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(
        self,
        main: Optional[nn.Module] = None,
        shortcut: Optional[nn.Module] = None,
        post_act: Optional[nn.Module] = None,
        pre_norm: Optional[nn.Module] = None,
    ):
        super(ResidualBlock, self).__init__()

        self.main = main
        self.shortcut = shortcut
        self.post_act = build_act(post_act)
        self.pre_norm = pre_norm

    def forward_main(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm is None:
            return self.main(x)
        return self.main(self.pre_norm(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.main is None:
            res = x
        elif self.shortcut is None:
            res = self.forward_main(x)
        else:
            res = self.forward_main(x) + self.shortcut(x)
            if self.post_act is not None:
                res = self.post_act(res)
        return res


class DAGBlock(nn.Module):
    def __init__(
        self,
        inputs: dict[str, nn.Module],
        merge: str,
        post_input: Optional[nn.Module],
        middle: Optional[nn.Module],
        outputs: dict[str, nn.Module],
    ):
        super(DAGBlock, self).__init__()

        self.input_keys = list(inputs.keys())
        self.input_ops = nn.ModuleList(list(inputs.values()))
        self.merge = merge
        self.post_input = post_input

        self.middle = middle

        self.output_keys = list(outputs.keys())
        self.output_ops = nn.ModuleList(list(outputs.values()))

    def forward(self, feature_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        feat = [op(feature_dict[k]) for k, op in zip(self.input_keys, self.input_ops)]
        if self.merge == "add":
            feat = list_sum(feat)
        elif self.merge == "cat":
            feat = torch.concat(feat, dim=1)
        else:
            raise NotImplementedError(f"merge={self.merge} not implemented.")

        if self.post_input is not None:
            feat = self.post_input(feat)

        if self.middle is not None:
            feat = self.middle(feat)

        output_dict = dict()
        for k, op in zip(self.output_keys, self.output_ops):
            output_dict[k] = op(feat)
        return output_dict


class OpSequential(nn.Module):
    def __init__(self, ops: list[Optional[nn.Module]]):
        super(OpSequential, self).__init__()
        valid_ops = []
        for op in ops:
            if op is not None:
                valid_ops.append(op)
        self.ops = nn.ModuleList(valid_ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for op in self.ops:
            x = op(x)
        return x
