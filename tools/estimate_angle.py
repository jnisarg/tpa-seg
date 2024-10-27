from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from tqdm import tqdm

from tpa.models import build_network  # Assuming this is a custom module


@dataclass
class ModelConfig:
    """Configuration for the neural network model."""

    n_classes: int = 4
    width_list: list[int] = (8, 16, 32, 64, 128)
    depth_list: list[int] = (1, 2, 2, 2, 2)
    head_width: int = 32
    head_depth: int = 1
    norm: str = "bn2d"
    act: str = "relu6"


@dataclass
class ProcessingConfig:
    """Configuration for image processing."""

    normalize_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    reference_point: tuple[int, int] = (258, 311)
    plot_line_length: int = 300
    # Color configuration for visualization (in BGR format for OpenCV)
    colors: dict = None

    def __post_init__(self):
        self.colors = {
            1: (0, 255, 0),  # Green
            2: (0, 165, 255),  # Orange
            3: (255, 0, 0),  # Blue
        }


class ImageProcessor:
    """Handles image loading and preprocessing."""

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.config.normalize_mean, std=self.config.normalize_std
                ),
            ]
        )

    def load_and_preprocess(self, image_path: Path) -> Tuple[np.ndarray, torch.Tensor]:
        """Load and preprocess image for model inference."""
        cv2_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

        tensor_image = self.transform(rgb_image)
        tensor_image = tensor_image.unsqueeze(0)

        return cv2_image, tensor_image


class AngleDetector:
    """Main class for angle detection in images."""

    def __init__(
        self,
        model_config: ModelConfig,
        processing_config: ProcessingConfig,
        checkpoint_path: Path,
    ):
        self.model = self._initialize_model(model_config, checkpoint_path)
        self.processor = ImageProcessor(processing_config)
        self.config = processing_config

    @staticmethod
    def _initialize_model(config: ModelConfig, checkpoint_path: Path) -> nn.Module:
        """Initialize and load the model from checkpoint."""
        model = build_network(
            n_classes=config.n_classes,
            width_list=config.width_list,
            depth_list=config.depth_list,
            head_width=config.head_width,
            head_depth=config.head_depth,
            norm=config.norm,
            act=config.act,
        )

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model

    def _compute_centroids(self, prediction: np.ndarray) -> Dict[int, np.ndarray]:
        """Compute centroids for each class in the prediction."""
        centroids = {}
        for class_id in range(1, 4):  # Classes 1, 2, 3
            ys, xs = np.where(prediction == class_id)
            if len(xs) > 0 and len(ys) > 0:
                centroids[class_id] = np.array([xs.mean(), ys.mean()])
        return centroids

    def _calculate_angle(self, centroids: Dict[int, np.ndarray]) -> float:
        """Calculate the angle based on centroids and reference point."""
        ref_point = np.array(self.config.reference_point)

        trailer_bar_vector = ref_point - centroids[2]
        trailer_ball_vector = ref_point - centroids[3]

        angle_bar = np.rad2deg(np.arctan2(trailer_bar_vector[1], trailer_bar_vector[0]))
        # angle_ball = np.rad2deg(
        #     np.arctan2(trailer_ball_vector[1], trailer_ball_vector[0])
        # )

        # return 180 - (angle_ball + angle_bar) / 2
        return 180 - angle_bar

    def _visualize_results(
        self,
        cv2_image: np.ndarray,
        prediction: np.ndarray,
        centroids: Dict[int, np.ndarray],
        angle: float,
        output_path: Path
    ) -> None:
        """
        Visualize and save the results using OpenCV with side-by-side comparison.
        Left side: Colored mask overlay
        Right side: Angle visualization with logical AND of image and mask
        """
        h, w = cv2_image.shape[:2]
        
        # Create a double-width canvas for side-by-side visualization
        visualization = np.zeros((h, w * 2, 3), dtype=np.uint8)
        
        # Left side - Colored mask overlay
        left_side = cv2_image.copy()
        overlay = np.zeros_like(cv2_image)
        
        # Create colored overlay for each class
        for class_id, color in self.config.colors.items():
            mask = (prediction == class_id).astype(np.uint8)
            overlay[mask == 1] = color
        
        # Blend the overlay with the original image for left side
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, left_side, 1 - alpha, 0, left_side)
        
        # Right side - Angle visualization with logical AND
        right_side = cv2_image.copy()
        
        # Create binary mask (all classes combined)
        combined_mask = (prediction > 0).astype(np.uint8)
        # Apply logical AND between original image and mask
        right_side = cv2.bitwise_and(right_side, right_side, mask=combined_mask)
        
        # Draw reference point
        ref_x, ref_y = self.config.reference_point
        cv2.circle(right_side, (ref_x, ref_y), 5, (0, 255, 0), -1)
        cv2.circle(right_side, (ref_x, ref_y), 7, (255, 255, 255), 2)
        
        # Draw angle line
        line_length = self.config.plot_line_length
        end_x = int(ref_x + line_length * np.cos(np.deg2rad(angle)))
        end_y = int(ref_y - line_length * np.sin(np.deg2rad(angle)))
        cv2.line(right_side, (ref_x, ref_y), (end_x, end_y), (255, 0, 0), 2)
        
        # Add angle text to both sides
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Estimated Angle: {angle:.2f}"
        text_size = cv2.getTextSize(text, font, 0.7, 2)[0]
        
        # Add text background and text to both sides
        for side in [left_side, right_side]:
            cv2.rectangle(side, (10, 10),
                        (10 + text_size[0], 10 + text_size[1] + 10),
                        (0, 0, 0), -1)
            cv2.putText(side, text,
                        (10, 30), font, 0.7, (255, 255, 255), 2)
        
        # Draw centroids on both sides
        for class_id, centroid in centroids.items():
            color = self.config.colors[class_id]
            center = tuple(map(int, centroid))
            # Draw on left side
            cv2.circle(left_side, center, 5, color, -1)
            cv2.circle(left_side, center, 7, (255, 255, 255), 2)
            # Draw on right side
            cv2.circle(right_side, center, 5, color, -1)
            cv2.circle(right_side, center, 7, (255, 255, 255), 2)
        
        # Combine the two visualizations side by side
        visualization[:, :w] = left_side
        visualization[:, w:] = right_side
        
        # Add a vertical line to separate the two sides
        cv2.line(visualization, (w, 0), (w, h), (255, 255, 255), 2)
        
        # Add labels for each side
        label_left = "Mask Overlay"
        label_right = "Angle Detection"
        font_scale = 0.8
        thickness = 2
        
        # Add labels with background
        for label, x_pos in [(label_left, 10), (label_right, w + 10)]:
            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            cv2.rectangle(visualization, 
                        (x_pos, h - 40),
                        (x_pos + text_size[0], h - 10),
                        (0, 0, 0), -1)
            cv2.putText(visualization, label,
                    (x_pos, h - 20), font, font_scale,
                    (255, 255, 255), thickness)
        
        # Save the result
        cv2.imwrite(str(output_path), visualization)

    def process_image(self, image_path: Path, output_path: Path) -> Optional[float]:
        """Process a single image and return the detected angle."""
        cv2_image, tensor_image = self.processor.load_and_preprocess(image_path)

        with torch.no_grad():
            prediction = self.model(tensor_image).squeeze()
            prediction = prediction.argmax(dim=0).numpy().astype(np.uint8)

        centroids = self._compute_centroids(prediction)

        # Check if all required centroids were found
        if not all(k in centroids for k in [1, 2, 3]):
            return None

        angle = self._calculate_angle(centroids)

        self._visualize_results(cv2_image, prediction, centroids, angle, output_path)

        return angle


def main():
    """Main function to process all images in a directory."""
    # Configuration
    model_config = ModelConfig()
    processing_config = ProcessingConfig()

    # Paths setup
    base_path = Path("data/BEV/images/Clip_2")
    output_path = Path("angle_results/Clip_2")
    checkpoint_path = Path(
        "experiments/exp003-all/checkpoints/best/best_checkpoint_e93_loss_0.0273_iou_0.9347.pth"
    )

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize detector
    detector = AngleDetector(model_config, processing_config, checkpoint_path)

    # Process all images
    image_paths = sorted(base_path.glob("*.png"))
    for image_path in tqdm(image_paths, desc="Processing images"):
        output_file = output_path / image_path.name
        angle = detector.process_image(image_path, output_file)
        if angle is None:
            print(f"Warning: Could not detect angle in {image_path}")


if __name__ == "__main__":
    main()
