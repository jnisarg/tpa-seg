import time
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from tpa.models.nn import (
    ConvLayer,
    DAGBlock,
    DSConv,
    IdentityLayer,
    MBConv,
    OpSequential,
    ResidualBlock,
    UpsampleLayer,
    InterpolateConvUpsampleLayer,
)


@dataclass
class EncoderConfig:
    """Configuration for TPAEncoder"""

    width_list: List[int]
    depth_list: List[int]
    expand_ratio: int = 4
    norm: str = "bn2d"
    act: str = "relu6"


@dataclass
class DecoderConfig:
    """Configuration for TPADecoder"""

    fid_list: List[str]
    in_channel_list: List[int]
    stride_list: List[int]
    head_stride: int
    head_width: int
    head_depth: int
    expand_ratio: float
    middle_op: str
    final_expand: Optional[float]
    n_classes: int
    dropout: float = 0.0
    norm: str = "bn2d"
    act: str = "relu6"


class TPAEncoder(nn.Module):
    """Trailer Parking Assist Architecture Encoder"""

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.width_list = []
        self.input_stem = self._build_input_stem(config)
        self.stages = self._build_stages(config)

    def _build_input_stem(self, config: EncoderConfig) -> OpSequential:
        """Build the input stem of the encoder"""
        stem_layers = [
            ConvLayer(
                in_channels=3,
                out_channels=config.width_list[0],
                kernel_size=3,
                stride=2,
                norm=config.norm,
                act=config.act,
            )
        ]

        for _ in range(config.depth_list[0]):
            block = self._build_local_block(
                in_channels=config.width_list[0],
                out_channels=config.width_list[0],
                stride=1,
                expand_ratio=1,
                norm=config.norm,
                act=config.act,
            )
            stem_layers.append(ResidualBlock(block, IdentityLayer()))

        self.width_list.append(config.width_list[0])
        return OpSequential(stem_layers)

    def _build_stages(self, config: EncoderConfig) -> List[OpSequential]:
        """Build the stages of the encoder"""
        stages = []
        in_channels = config.width_list[0]

        for width, depth in zip(config.width_list[1:], config.depth_list[1:]):
            stage_layers = []
            for i in range(depth):
                stride = 2 if i == 0 else 1
                block = self._build_local_block(
                    in_channels=in_channels,
                    out_channels=width,
                    stride=stride,
                    expand_ratio=config.expand_ratio,
                    norm=config.norm,
                    act=config.act,
                )
                block = ResidualBlock(block, IdentityLayer() if stride == 1 else None)
                stage_layers.append(block)
                in_channels = width

            stages.append(OpSequential(stage_layers))
            self.width_list.append(in_channels)

        return nn.ModuleList(stages)

    @staticmethod
    @lru_cache(maxsize=32)
    def _build_local_block(
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: int,
        norm: str = "bn2d",
        act: str = "relu6",
    ) -> nn.Module:
        """Build a local block with caching for repeated configurations"""
        if expand_ratio == 1:
            return DSConv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                use_bias=False,
                norm=norm,
                act=(act, None),
            )

        return MBConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            expand_ratio=expand_ratio,
            use_bias=False,
            norm=norm,
            act=(act, act, None),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = {"input": x}
        outputs["stage0"] = x = self.input_stem(x)

        for stage_id, stage in enumerate(self.stages, 1):
            outputs[f"stage{stage_id}"] = x = stage(x)

        outputs["stage_final"] = x
        return outputs


class TPADecoder(DAGBlock):
    """Trailer Parking Assist Architecture Decoder"""

    def __init__(self, config: DecoderConfig):
        inputs = self._build_inputs(config)
        middle = self._build_middle_layers(config)
        outputs = self._build_outputs(config)

        super().__init__(inputs, "add", None, middle, outputs)

    @staticmethod
    def _build_inputs(config: DecoderConfig) -> Dict[str, nn.Module]:
        """Build input layers for the decoder"""
        inputs = {}
        for fid, in_channel, stride in zip(
            config.fid_list, config.in_channel_list, config.stride_list
        ):
            factor = stride // config.head_stride
            if factor == 1:
                inputs[fid] = ConvLayer(
                    in_channel, config.head_width, 1, norm=config.norm, act=config.act
                )
            else:
                inputs[fid] = OpSequential(
                    [
                        ConvLayer(
                            in_channel,
                            config.head_width,
                            1,
                            norm=config.norm,
                            act=config.act,
                        ),
                        # UpsampleLayer(factor=factor),
                        InterpolateConvUpsampleLayer(
                            in_channels=config.head_width,
                            out_channels=config.head_width,
                            kernel_size=3,
                            factor=factor,
                        ),
                    ]
                )
        return inputs

    @staticmethod
    def _build_middle_layers(config: DecoderConfig) -> OpSequential:
        """Build middle layers for the decoder"""
        middle_layers = []
        for _ in range(config.head_depth):
            if config.middle_op == "mbconv":
                block = MBConv(
                    config.head_width,
                    config.head_width,
                    expand_ratio=config.expand_ratio,
                    norm=config.norm,
                    act=(config.act, config.act, None),
                )
            elif config.middle_op == "fmbconv":
                block = MBConv(
                    config.head_width,
                    config.head_width,
                    expand_ratio=config.expand_ratio,
                    norm=config.norm,
                    act=(config.act, None),
                )
            else:
                raise ValueError(f"Unsupported middle_op: {config.middle_op}")

            middle_layers.append(ResidualBlock(block, IdentityLayer()))
        return OpSequential(middle_layers)

    @staticmethod
    def _build_outputs(config: DecoderConfig) -> Dict[str, nn.Module]:
        """Build output layers for the decoder"""
        final_width = config.head_width * (
            1 if config.final_expand is None else config.final_expand
        )
        expand_layer = (
            None
            if config.final_expand is None
            else ConvLayer(
                config.head_width,
                int(config.head_width * config.final_expand),
                1,
                norm=config.norm,
                act=config.act,
            )
        )

        return {
            "segout": OpSequential(
                [
                    expand_layer,
                    ConvLayer(
                        int(final_width),
                        config.n_classes,
                        1,
                        use_bias=True,
                        dropout=config.dropout,
                        norm=config.norm,
                        act=None,
                    ),
                    UpsampleLayer(factor=config.head_stride),
                ]
            )
        }


class TPANetwork(nn.Module):
    """Complete Trailer Parking Assist Architecture Network"""

    def __init__(self, encoder: TPAEncoder, decoder: TPADecoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded["segout"]

    @property
    def num_parameters(self) -> int:
        """Return the number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def load_pretrained(self, weights_path: Union[str, Path]) -> None:
        """Load pretrained weights from a file"""
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        try:
            state_dict = torch.load(weights_path, map_location="cpu")
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]

            self.load_state_dict(state_dict)
            self._is_initialized = True
            print(f"Successfully loaded pretrained weights from {weights_path}")

        except Exception as e:
            print(f"Error loading pretrained weights: {str(e)}")
            raise

    def save_pretrained(self, save_path: Union[str, Path]) -> None:
        """Save model weights to a file"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            torch.save(
                {
                    "model_state_dict": self.state_dict(),
                    "config": {
                        "encoder": self.encoder.config.__dict__,
                        "decoder": self.decoder.config.__dict__,
                    },
                },
                save_path,
            )
            print(f"Successfully saved model weights to {save_path}")

        except Exception as e:
            print(f"Error saving model weights: {str(e)}")
            raise

    @torch.no_grad()
    def measure_latency(
        self,
        input_shape: Tuple[int, ...] = (1, 3, 320, 512),
        num_iterations: int = 100,
        warmup_iterations: int = 10,
        device: str = "cuda",
    ) -> Dict[str, float]:
        """
        Measure model latency with detailed statistics

        Args:
            input_shape: Input tensor shape (batch_size, channels, height, width)
            num_iterations: Number of iterations for measurement
            warmup_iterations: Number of warmup iterations
            device: Device to run measurements on ('cuda' or 'cpu')

        Returns:
            Dictionary containing latency statistics (mean, std, min, max)
        """
        if not torch.cuda.is_available() and device == "cuda":
            print("CUDA not available, falling back to CPU")
            device = "cpu"

        self.eval()
        self.to(device)

        # Warmup
        x = torch.randn(*input_shape, device=device)
        for _ in range(warmup_iterations):
            _ = self(x)

        if device == "cuda":
            torch.cuda.synchronize()

        # Measure latency
        latencies = []
        for _ in range(num_iterations):
            start_time = time.perf_counter()
            _ = self(x)

            if device == "cuda":
                torch.cuda.synchronize()

            latencies.append(time.perf_counter() - start_time)

        # Calculate statistics
        latencies = torch.tensor(latencies) * 1000  # Convert to milliseconds
        stats = {
            "mean_ms": float(latencies.mean()),
            "std_ms": float(latencies.std()),
            "min_ms": float(latencies.min()),
            "max_ms": float(latencies.max()),
            "median_ms": float(latencies.median()),
            "device": device,
            "input_shape": input_shape,
            "iterations": num_iterations,
        }

        return stats


@contextmanager
def timer(name: str):
    """Context manager for timing code blocks"""
    start_time = time.perf_counter()
    yield
    elapsed_time = time.perf_counter() - start_time
    print(f"{name} took {elapsed_time*1000:.2f}ms")


def build_network(
    n_classes: int,
    width_list: List[int],
    depth_list: List[int],
    head_width: int,
    head_depth: int = 1,
    final_expand: Optional[float] = 4.0,
    norm: str = "bn2d",
    act: str = "relu6",
) -> TPANetwork:
    """Factory function to build a TPANetwork with the given configuration"""
    encoder_config = EncoderConfig(
        width_list=width_list,
        depth_list=depth_list,
        expand_ratio=4,
        norm=norm,
        act=act,
    )

    decoder_config = DecoderConfig(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[width_list[-1], width_list[-2], width_list[-3]],
        stride_list=[32, 16, 8],
        head_stride=8,
        head_width=head_width,
        head_depth=head_depth,
        expand_ratio=4,
        middle_op="mbconv",
        final_expand=final_expand,
        n_classes=n_classes,
        norm=norm,
        act=act,
    )

    return TPANetwork(
        encoder=TPAEncoder(encoder_config),
        decoder=TPADecoder(decoder_config),
    )


if __name__ == "__main__":
    # Example usage
    network = build_network(
        n_classes=4,
        width_list=[8, 16, 32, 64, 128],
        depth_list=[1, 2, 2, 2, 2],
        head_width=32,
        head_depth=1,
        final_expand=4.0,
        norm="bn2d",
        act="relu6",
    )

    sample_input = torch.randn(1, 3, 320, 512)
    output = network(sample_input)

    print(f"Output shape: {output.shape}")
    print(f"Trainable parameters: {network.num_parameters/1e6:.3f}M")

    # Measure latency
    stats = network.measure_latency(
        input_shape=(1, 3, 320, 512), num_iterations=100, warmup_iterations=10
    )
    print(f"Latency statistics: {stats}")
