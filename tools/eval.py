import argparse
import os
import warnings
from typing import Dict

import cv2
import numpy as np
import torch
import torch.cuda.amp as amp
from rich.console import Console
from tqdm import tqdm

from tpa.datasets import DataloaderConfig, build_dataloaders
from tpa.models import build_network
from tpa.utils.metrics import MetricsCalculator
from tpa.utils.tpa_criterion import TPACriterion

warnings.filterwarnings(action="ignore")


class EvaluationConfig:
    """Configuration class for evaluation parameters"""

    def __init__(self, args: argparse.Namespace):
        self.device = args.device
        self.checkpoint_path = args.checkpoint_path
        self.save_predictions = args.save_predictions
        self.output_dir = args.output_dir
        self.class_names = args.class_names


class Evaluator:
    """Main evaluation class"""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.console = Console()

        if config.save_predictions and config.output_dir:
            os.makedirs(config.output_dir, exist_ok=True)
            self.colors = [
                (0.1, 0.1, 0.1),  # Dark gray for background
                (0.2, 0.7, 0.2),  # Green for trailer
                (0.9, 0.3, 0.1),  # Red-orange for trailer bar
                (0.1, 0.3, 0.9),  # Blue for trailer ball
            ]

    def load_checkpoint(self, model: torch.nn.Module, checkpoint_path: str):
        """Loads model weights from checkpoint"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint.get("epoch", None)

    @torch.no_grad()
    def evaluate(
        self,
        model: torch.nn.Module,
        val_loader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
    ) -> Dict[str, float]:
        """Evaluates the model on the validation set"""
        model.eval()
        metrics_calculator = MetricsCalculator(
            n_classes=len(self.config.class_names),
            class_names=self.config.class_names,
            device=self.device,
        )

        progress_bar = tqdm(
            val_loader, desc="Evaluating", leave=False, ncols=100, colour="cyan"
        )

        with amp.autocast():
            for images, masks, image_ids in progress_bar:
                images = images.to(self.device)
                masks = masks.to(self.device)

                # Forward pass
                outputs = model(images)
                loss = loss_fn(outputs, masks)

                # Convert outputs to predictions
                pred_masks = torch.argmax(outputs, dim=1)

                # Update metrics
                metrics_calculator.update(pred_masks, masks, loss.item())

                # Save predictions if requested
                if self.config.save_predictions:
                    self._save_predictions(images, masks, pred_masks, image_ids)

        # Compute and log final metrics
        metrics = metrics_calculator.compute_metrics()
        metrics_calculator.log_metrics(metrics, self.console)

        return metrics

    def _save_predictions(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        predictions: torch.Tensor,
        image_ids: list[str],
    ) -> None:
        """Saves visualization of images with their corresponding masks and predictions.

        Args:
            images: Batch of input images (N, C, H, W)
            masks: Ground truth masks (N, H, W)
            predictions: Predicted masks (N, H, W)
            image_ids: List of image identifiers
        """
        # Constants
        IMAGE_WEIGHT = 0.7
        MASK_WEIGHT = 1 - IMAGE_WEIGHT

        def denormalize_images(images: torch.Tensor) -> np.ndarray:
            """Denormalize images and convert to numpy array."""
            mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(self.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(self.device)

            images = images.clone()
            images = images * std + mean
            images = images.clamp(0, 1)
            return images.cpu().numpy().transpose(0, 2, 3, 1)

        def create_colored_mask(label_mask: np.ndarray, colors: list) -> np.ndarray:
            """Create RGB colored mask from label mask."""
            h, w = label_mask.shape
            colored = np.zeros((h, w, 3))
            for class_idx, color in enumerate(colors):
                colored[label_mask == class_idx] = color
            return colored

        def blend_images(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
            """Blend image with colored mask."""
            blended = IMAGE_WEIGHT * image + MASK_WEIGHT * mask
            return np.clip(blended, 0, 1)

        def save_image(image: np.ndarray, path: str) -> None:
            """Save image to disk."""
            cv2.imwrite(path, image * 255)

        # Process images
        images = denormalize_images(images)
        masks = masks.cpu().numpy()
        predictions = predictions.cpu().numpy()

        for image, mask, prediction, image_id in zip(
            images, masks, predictions, image_ids
        ):
            # Convert image from RGB to BGR for OpenCV
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Create colored versions of masks
            colored_mask = create_colored_mask(mask, self.colors)
            colored_prediction = create_colored_mask(prediction, self.colors)

            # Create blended versions
            blended_mask = blend_images(image, colored_mask)
            blended_prediction = blend_images(image, colored_prediction)

            # Save all versions
            output_dir = self.config.output_dir
            save_image(image, os.path.join(output_dir, f"{image_id}.png"))
            save_image(blended_mask, os.path.join(output_dir, f"{image_id}_mask.png"))
            save_image(
                blended_prediction,
                os.path.join(output_dir, f"{image_id}_prediction.png"),
            )


def parse_eval_args() -> argparse.Namespace:
    """Parse command line arguments for evaluation"""
    parser = argparse.ArgumentParser(
        description="TPA Segmentation Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Experiment settings
    exp_group = parser.add_argument_group("Evaluation Configuration")
    exp_group.add_argument(
        "--checkpoint-path", type=str, required=True, help="Path to model checkpoint"
    )

    # Model settings
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--n-classes", type=int, default=4, help="Number of segmentation classes"
    )
    model_group.add_argument(
        "--class-names",
        type=str,
        nargs="+",
        default=["background", "trailer", "trailer_bar", "trailer_ball"],
        help="Names of the classes for reporting",
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

    # Evaluation settings
    eval_group = parser.add_argument_group("Evaluation Parameters")
    eval_group.add_argument(
        "--batch-size", type=int, default=1, help="Batch size for evaluation"
    )
    eval_group.add_argument(
        "--num-workers", type=int, default=4, help="Number of data loading workers"
    )
    eval_group.add_argument(
        "--save-predictions", action="store_true", help="Save prediction visualizations"
    )
    eval_group.add_argument(
        "--output-dir", type=str, default=None, help="Directory to save predictions"
    )

    # Model components
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
        help="Device to use for evaluation",
    )
    misc_group.add_argument(
        "--top_k_percent",
        type=float,
        default=0.8,
        help="Top-k percentage for TPACriterion",
    )

    args = parser.parse_args()

    # Validate device selection
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA is not available, falling back to CPU")
        args.device = "cpu"

    assert args.n_classes == len(args.class_names), (
        f"Number of classes ({args.n_classes}) does not match "
        f"number of class names ({len(args.class_names)})"
    )

    return args


def main():
    """Main entry point for evaluation"""
    args = parse_eval_args()
    config = EvaluationConfig(args)

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

    # Initialize evaluator and load checkpoint
    evaluator = Evaluator(config)
    epoch = evaluator.load_checkpoint(model, config.checkpoint_path)
    model = model.to(config.device)

    # Build validation dataloader
    _, val_loader = build_dataloaders(
        config=DataloaderConfig(
            batch_size=1, val_batch_size=args.batch_size, num_workers=args.num_workers
        )
    )

    # Initialize loss function
    loss_fn = TPACriterion(top_k_percent=args.top_k_percent)

    # Run evaluation
    evaluator.evaluate(model, val_loader, loss_fn)


if __name__ == "__main__":
    main()
