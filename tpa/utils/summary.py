import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch
from tabulate import tabulate
from torch import nn

__all__ = ["log_model_stats"]


def count_parameters_and_size(model: nn.Module) -> Dict[str, Any]:
    """
    Count the total parameters and size of a PyTorch model.

    Args:
        model: PyTorch model

    Returns:
        Dictionary containing parameter statistics
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Calculate size in different units
    total_size_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    size_mb = total_size_bytes / (1024 * 1024)

    # Get per-layer breakdown
    layer_stats = defaultdict(dict)
    for name, param in model.named_parameters():
        layer_stats[name] = {
            "shape": tuple(param.shape),
            "parameters": param.numel(),
            "size_kb": param.nelement() * param.element_size() / 1024,
        }

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params,
        "model_size_mb": size_mb,
        "layer_statistics": dict(layer_stats),
    }


def estimate_flops(model: nn.Module, input_shape: Tuple[int, ...]) -> Dict[str, Any]:
    """
    Estimate FLOPs for common layer types in a PyTorch model.

    Args:
        model: PyTorch model
        input_shape: Input tensor shape (batch_size, channels, height, width) for vision models
                    or (batch_size, sequence_length, hidden_dim) for transformers

    Returns:
        Dictionary containing FLOP estimates
    """
    flops_dict = defaultdict(int)

    def hook_fn(module, input, output):
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            # Get input dimensions
            batch_size = input[0].size(0)
            input_channels = input[0].size(1)
            output_channels = output.size(1)
            output_height = output.size(2)
            output_width = output.size(3)
            kernel_size = module.kernel_size[0] * module.kernel_size[1]

            # Calculate FLOPs for convolution
            flops = (
                batch_size
                * output_channels
                * output_height
                * output_width
                * input_channels
                * kernel_size
            )
            flops_dict["conv"] += flops

        elif isinstance(module, nn.Linear):
            batch_size = input[0].size(0)
            input_features = module.in_features
            output_features = module.out_features

            # Calculate FLOPs for linear layer
            flops = batch_size * input_features * output_features
            flops_dict["linear"] += flops

        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            batch_size = input[0].size(0)
            num_features = module.num_features

            # Calculate FLOPs for batch normalization
            flops = batch_size * num_features * 2  # multiply and add
            flops_dict["batch_norm"] += flops

    # Register hooks
    hooks = []
    for module in model.modules():
        hooks.append(module.register_forward_hook(hook_fn))

    # Run a forward pass with dummy input
    device = next(model.parameters()).device
    dummy_input = torch.randn(input_shape).to(device)
    with torch.no_grad():
        model(dummy_input)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    total_flops = sum(flops_dict.values())

    return {
        "total_flops": total_flops,
        "flops_breakdown": dict(flops_dict),
        "flops_gflops": total_flops / (10**9),
    }


def format_number(num: float) -> str:
    """Format large numbers with appropriate units"""
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return f"{num:.2f}"


def create_summary_table(stats: Dict[str, Any]) -> pd.DataFrame:
    """Previous implementation remains the same"""
    summary_data = {
        "Metric": [
            "Total Parameters",
            "Trainable Parameters",
            "Non-trainable Parameters",
            "Model Size",
            "Total GFLOPs",
        ],
        "Value": [
            format_number(stats["parameters"]["total_parameters"]),
            format_number(stats["parameters"]["trainable_parameters"]),
            format_number(stats["parameters"]["non_trainable_parameters"]),
            f"{stats['parameters']['model_size_mb']:.2f} MB",
            f"{stats['computation']['flops_gflops']:.2f}",
        ],
    }
    return pd.DataFrame(summary_data)


def create_layer_table(stats: Dict[str, Any]) -> pd.DataFrame:
    """Create a detailed table of layer statistics with totals"""
    layer_stats = stats["parameters"]["layer_statistics"]
    layer_data = []

    total_params = 0
    total_size_kb = 0

    # Add individual layer rows
    for name, layer_info in layer_stats.items():
        params = layer_info["parameters"]
        size_kb = layer_info["size_kb"]
        total_params += params
        total_size_kb += size_kb

        layer_data.append(
            {
                "Layer": name,
                "Shape": str(layer_info["shape"]),
                "Parameters": format_number(params),
                "Size": f"{size_kb/1024:.2f} MB",
                "Parameters (%)": f"{(params/stats['parameters']['total_parameters'])*100:.1f}%",
                "Size (%)": f"{(size_kb/1024/stats['parameters']['model_size_mb'])*100:.1f}%",
            }
        )

    # Add total row
    layer_data.append(
        {
            "Layer": "TOTAL",
            "Shape": "-",
            "Parameters": format_number(total_params),
            "Size": f"{total_size_kb/1024:.2f} MB",
            "Parameters (%)": "100.0%",
            "Size (%)": "100.0%",
        }
    )

    df = pd.DataFrame(layer_data)

    # Style the DataFrame for better visualization
    def highlight_total(row):
        if row["Layer"] == "TOTAL":
            return ["font-weight: bold"] * len(row)
        return [""] * len(row)

    return df.style.apply(highlight_total, axis=1)


def create_flops_table(stats: Dict[str, Any]) -> pd.DataFrame:
    """Create a table of FLOPS breakdown with total"""
    flops_data = []
    total_flops = stats["computation"]["total_flops"]

    # Add individual operation rows
    for op_type, flops in stats["computation"]["flops_breakdown"].items():
        flops_data.append(
            {
                "Operation": op_type,
                "GFLOPs": f"{flops/1e9:.2f}",
                "Percentage": f"{(flops/total_flops)*100:.1f}%",
            }
        )

    # Add total row
    flops_data.append(
        {
            "Operation": "TOTAL",
            "GFLOPs": f"{total_flops/1e9:.2f}",
            "Percentage": "100.0%",
        }
    )

    df = pd.DataFrame(flops_data)

    # Style the DataFrame for better visualization
    def highlight_total(row):
        if row["Operation"] == "TOTAL":
            return ["font-weight: bold"] * len(row)
        return [""] * len(row)

    return df.style.apply(highlight_total, axis=1)


def log_model_stats(
    model: nn.Module,
    logger: logging.Logger,
    input_shape: Tuple[int, ...],
    output_dir: str = None,
    model_name: str = "model",
) -> Dict[str, Any]:
    """
    Log comprehensive model statistics in various formats

    Args:
        model: PyTorch model
        input_shape: Input tensor shape
        output_dir: Directory to save the logs (optional)
        model_name: Name of the model for logging
    """
    # Calculate statistics
    param_stats = count_parameters_and_size(model)
    flop_stats = estimate_flops(model, input_shape)

    stats = {
        "parameters": param_stats,
        "computation": flop_stats,
        "summary": {
            "total_params_millions": param_stats["total_parameters"] / 10**6,
            "model_size_mb": param_stats["model_size_mb"],
            "gflops": flop_stats["flops_gflops"],
        },
    }

    # Create tables
    summary_df = create_summary_table(stats)
    layer_df = create_layer_table(stats)
    flops_df = create_flops_table(stats)

    # Print formatted tables
    logger.info("\n=== Model Summary ===")
    logger.info(
        tabulate(summary_df, headers="keys", tablefmt="pretty", showindex=False)
    )

    logger.info("\n=== Layer Details ===")
    logger.info(
        tabulate(layer_df.data, headers="keys", tablefmt="pretty", showindex=False)
    )

    logger.info("\n=== FLOPS Breakdown ===")
    logger.info(
        tabulate(flops_df.data, headers="keys", tablefmt="pretty", showindex=False)
    )

    # Save to files if output_dir is provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save as CSV
        summary_df.to_csv(output_dir / f"{model_name}_summary.csv", index=False)
        layer_df.data.to_csv(output_dir / f"{model_name}_layers.csv", index=False)
        flops_df.data.to_csv(output_dir / f"{model_name}_flops.csv", index=False)

        # Save as JSON
        with open(output_dir / f"{model_name}_stats.json", "w") as f:
            json.dump(stats, f, indent=2, default=str)

        # Create visualizations
        plt.figure(figsize=(10, 6))
        plt.pie(
            [
                float(x.strip("%")) / 100 for x in flops_df.data["Percentage"][:-1]
            ],  # Exclude total row
            labels=flops_df.data["Operation"][:-1],
            autopct="%1.1f%%",
        )
        plt.title("FLOPS Distribution by Operation Type")
        plt.savefig(output_dir / f"{model_name}_flops_distribution.png")
        plt.close()

        logger.info(f"\nLogs saved to {output_dir}")

    return stats
