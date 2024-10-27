import argparse
import os
import warnings
from typing import Optional, Tuple

import onnx
import onnxsim
import torch
import torch.nn as nn
from rich.console import Console

from tpa.models import build_network

warnings.filterwarnings(action="ignore")


class ModelWithArgmax(nn.Module):
    """Wrapper class that adds argmax to model output"""

    def __init__(self, model: nn.Module, dim: int = 1):
        super().__init__()
        self.model = model
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return torch.argmax(x, dim=self.dim, keepdim=True)


class ONNXExportConfig:
    """Configuration class for ONNX export parameters"""

    def __init__(self, args: argparse.Namespace):
        self.device = args.device
        self.checkpoint_path = args.checkpoint_path
        self.output_path = args.output_path
        self.input_shape = args.input_shape
        self.dynamic_axes = args.dynamic_axes
        self.simplify = args.simplify
        self.add_argmax = args.add_argmax


class ONNXExporter:
    """Main ONNX export class"""

    def __init__(self, config: ONNXExportConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.console = Console()

    def load_checkpoint(
        self, model: torch.nn.Module, checkpoint_path: str
    ) -> Optional[int]:
        """Loads model weights from checkpoint"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint.get("epoch", None)

    def apply_model_surgery(self, model: nn.Module) -> nn.Module:
        """Applies model surgery to add argmax operation"""
        if self.config.add_argmax:
            self.console.print(
                "[cyan]Adding argmax operation to model output...[/cyan]"
            )
            return ModelWithArgmax(model, dim=1)
        return model

    @torch.no_grad()
    def export_onnx(
        self,
        model: torch.nn.Module,
        input_shape: Tuple[int, ...],
        output_path: str,
        dynamic_axes: Optional[dict] = None,
        simplify: bool = True,
    ) -> None:
        """Exports model to ONNX format and optionally simplifies it"""
        model.eval()

        # Apply model surgery if requested
        model = self.apply_model_surgery(model)

        # Create dummy input
        dummy_input = torch.randn(input_shape, device=self.device)

        # Export to ONNX
        self.console.print("[cyan]Exporting model to ONNX format...[/cyan]")

        # Adjust output shape for verification based on whether argmax is added
        if self.config.add_argmax:
            # With argmax, output will be [B, 1, H, W]
            output_sample = model(dummy_input)
        else:
            # Without argmax, output will be [B, C, H, W]
            output_sample = model(dummy_input)

        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            opset_version=11,
            do_constant_folding=True,
            verbose=False,
        )

        if simplify:
            self.console.print("[cyan]Simplifying ONNX model...[/cyan]")
            # Load the ONNX model
            onnx_model = onnx.load(output_path)

            # Simplify
            model_simplified, check = onnxsim.simplify(onnx_model)

            if check:
                self.console.print("[green]✓ Model simplified successfully[/green]")
                # Save the simplified model
                onnx.save(model_simplified, output_path)
            else:
                self.console.print("[red]✗ Model simplification failed[/red]")

        # Verify the exported model
        self.verify_onnx_export(output_path, dummy_input, output_sample)

        self.console.print(f"[green]✓ Model exported to: {output_path}[/green]")

        # Print output shape information
        output_shape_str = "x".join(str(dim) for dim in output_sample.shape)
        self.console.print(f"[blue]Output shape: {output_shape_str}[/blue]")

    def verify_onnx_export(
        self, onnx_path: str, dummy_input: torch.Tensor, expected_output: torch.Tensor
    ) -> None:
        """Verifies the exported ONNX model matches PyTorch output"""
        import onnxruntime as ort

        self.console.print("[cyan]Verifying ONNX export...[/cyan]")

        # Create ONNX Runtime session
        ort_session = ort.InferenceSession(
            onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )

        # Run inference with ONNX Runtime
        ort_inputs = {"input": dummy_input.cpu().numpy()}
        ort_output = ort_session.run(None, ort_inputs)[0]

        # Convert outputs to numpy for comparison
        pytorch_output = expected_output.cpu().numpy()

        # Check shapes match
        shape_match = pytorch_output.shape == ort_output.shape
        if shape_match:
            self.console.print("[green]✓ Output shapes match[/green]")
        else:
            self.console.print(
                f"[red]✗ Shape mismatch - PyTorch: {pytorch_output.shape}, "
                f"ONNX: {ort_output.shape}[/red]"
            )


def parse_export_args() -> argparse.Namespace:
    """Parse command line arguments for ONNX export"""
    parser = argparse.ArgumentParser(
        description="TPA Segmentation ONNX Export",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Export settings
    exp_group = parser.add_argument_group("Export Configuration")
    exp_group.add_argument(
        "--checkpoint-path", type=str, required=True, help="Path to model checkpoint"
    )
    exp_group.add_argument(
        "--output-path", type=str, required=True, help="Path to save ONNX model"
    )
    exp_group.add_argument(
        "--input-shape",
        type=int,
        nargs=4,
        default=[1, 3, 320, 512],
        help="Input shape (batch_size, channels, height, width)",
    )
    exp_group.add_argument(
        "--dynamic-axes", action="store_true", help="Enable dynamic axes for batch size"
    )
    exp_group.add_argument(
        "--simplify", action="store_true", help="Simplify ONNX model after export"
    )
    exp_group.add_argument(
        "--add-argmax", action="store_true", help="Add argmax operation to model output"
    )

    # Model settings (same as evaluation)
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--n-classes", type=int, default=4, help="Number of segmentation classes"
    )
    model_group.add_argument(
        "--width-list",
        type=int,
        nargs="+",
        default=[8, 16, 32, 64, 128],
        help="Channel widths at each network level",
    )
    model_group.add_argument(
        "--depth-list",
        type=int,
        nargs="+",
        default=[1, 2, 2, 2, 2],
        help="Number of blocks at each network level",
    )
    model_group.add_argument(
        "--head-width", type=int, default=32, help="Number of channels in the head"
    )
    model_group.add_argument(
        "--head-depth",
        type=int,
        default=1,
        help="Number of convolution layers in the head",
    )

    # Model components (same as evaluation)
    components_group = parser.add_argument_group("Model Components")
    components_group.add_argument(
        "--norm",
        type=str,
        default="bn2d",
        choices=["bn2d", "ln", "ln2d", "none"],
        help="Normalization layer type",
    )
    components_group.add_argument(
        "--act",
        type=str,
        default="relu6",
        choices=["relu6", "silu", "relu", "none"],
        help="Activation function type",
    )

    # Device settings
    misc_group = parser.add_argument_group("Miscellaneous")
    misc_group.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for export",
    )

    args = parser.parse_args()

    # Validate device selection
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA is not available, falling back to CPU")
        args.device = "cpu"

    return args


def main():
    """Main entry point for ONNX export"""
    args = parse_export_args()
    config = ONNXExportConfig(args)

    # Build model
    model = build_network(
        n_classes=args.n_classes,
        width_list=args.width_list,
        depth_list=args.depth_list,
        head_width=args.head_width,
        head_depth=args.head_depth,
        norm=args.norm,
        act=args.act,
    )

    # Initialize exporter and load checkpoint
    exporter = ONNXExporter(config)
    _ = exporter.load_checkpoint(model, config.checkpoint_path)
    model = model.to(config.device)

    # Prepare dynamic axes if enabled
    dynamic_axes = (
        {"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        if args.dynamic_axes
        else None
    )

    output_dir = os.path.dirname(config.output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Export model to ONNX
    exporter.export_onnx(
        model=model,
        input_shape=tuple(args.input_shape),
        output_path=args.output_path,
        dynamic_axes=dynamic_axes,
        simplify=args.simplify,
    )


if __name__ == "__main__":
    main()
