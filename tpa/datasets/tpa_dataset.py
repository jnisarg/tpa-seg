import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import albumentations as A
import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

__all__ = [
    "TPADataset",
    "DatasetMode",
    "DatasetConfig",
    "DataloaderConfig",
    "DatasetVisualizer",
    "visualize_dataset_samples",
    "build_dataloaders",
]


class DatasetMode(Enum):
    """Enum for dataset modes."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


@dataclass
class DatasetConfig:
    """Configuration class for dataset parameters."""

    root_dir: Path = Path("data/BEV")
    image_size: Tuple[int, int] = (320, 512)  # Added configurable image size
    class_names: List[str] = ("background", "trailer", "trailer_bar", "trailer_ball")
    id_to_trainid: Dict[int, int] = field(
        default_factory=lambda: {0: 0, 1: 1, 2: 2, 3: 3}
    )
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    std: Tuple[float, ...] = (0.229, 0.224, 0.225)


class TPADataset(Dataset):
    """
    Dataset class for Trailer Position Alignment (TPA) segmentation task.

    This dataset handles loading and preprocessing of images and masks for
    semantic segmentation of trailer-related objects in bird's eye view images.

    Attributes:
        mode (DatasetMode): Dataset split mode (train/val/test)
        config (DatasetConfig): Configuration parameters
        transforms (Optional[A.Compose]): Albumentations transformations
        samples (List[Tuple[str, str]]): List of (image_path, mask_path) pairs
    """

    def __init__(
        self,
        mode: Literal["train", "val", "test"] = "train",
        config: Optional[DatasetConfig] = None,
    ):
        super().__init__()
        self.mode = DatasetMode(mode)
        self.config = config or DatasetConfig()
        self.transforms = self._build_transforms()
        self.samples = self._load_samples()

        # Cache normalize transform
        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=self.config.mean, std=self.config.std),
            ]
        )

    def _build_transforms(self) -> Optional[A.Compose]:
        """Build augmentation pipeline based on dataset mode."""
        if self.mode != DatasetMode.TRAIN:
            return None

        return A.Compose(
            [
                # Spatial augmentations
                A.RandomResizedCrop(*self.config.image_size, scale=(0.8, 1.0), p=0.5),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.2,
                    rotate_limit=30,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=0.5,
                ),
                # Color augmentations
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(
                            brightness_limit=0.2, contrast_limit=0.2, p=0.5
                        ),
                        A.ColorJitter(
                            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5
                        ),
                    ],
                    p=0.5,
                ),
                # Noise augmentations
                A.OneOf(
                    [
                        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                        A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=0.5),
                    ],
                    p=0.3,
                ),
                # Weather augmentations
                A.OneOf(
                    [
                        A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.5),
                        A.RandomShadow(p=0.5),
                        A.RandomSunFlare(
                            flare_roi=(0, 0, 1, 0.5),
                            angle_lower=0,
                            angle_upper=1,
                            num_flare_circles_lower=6,
                            num_flare_circles_upper=10,
                            src_radius=400,
                            src_color=(255, 255, 255),
                            p=0.5,
                        ),
                    ],
                    p=0.2,
                ),
                # Quality augmentations
                A.OneOf(
                    [
                        A.ImageCompression(quality_lower=80, quality_upper=100, p=0.5),
                        A.Blur(blur_limit=3, p=0.5),
                    ],
                    p=0.2,
                ),
            ]
        )

    def _load_samples(self) -> List[Tuple[str, str]]:
        """Load and validate dataset samples."""
        samples_file = self.config.root_dir / f"{self.mode.value}.txt"
        if not samples_file.exists():
            raise FileNotFoundError(f"Samples file not found: {samples_file}")

        with open(samples_file, "r") as f:
            samples = [tuple(line.strip().split()) for line in f]

        # Validate paths
        for img_path, mask_path in samples:
            full_img_path = self.config.root_dir / img_path
            full_mask_path = self.config.root_dir / mask_path
            if not full_img_path.exists() or not full_mask_path.exists():
                raise FileNotFoundError(
                    f"Missing files: {full_img_path} or {full_mask_path}"
                )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample from the dataset.

        Args:
            idx (int): Sample index

        Returns:
            Tuple[torch.Tensor, torch.Tensor, str]: Normalized image, segmentation mask, and image ID (Name)
        """
        img_path, mask_path = self.samples[idx]
        img_path = self.config.root_dir / img_path
        mask_path = self.config.root_dir / mask_path

        # Load images with error handling
        img = self._load_image(img_path)
        mask = self._load_mask(mask_path)

        # Apply augmentations
        if self.transforms is not None:
            transformed = self.transforms(image=img, mask=mask)
            img, mask = transformed["image"], transformed["mask"]

        # Map IDs
        mask = self._map_ids(mask)

        # Convert to tensors
        img = self.normalize(img)
        mask = torch.from_numpy(mask).long()

        img_id = img_path.stem

        return img, mask, img_id

    def _load_image(self, path: Path) -> np.ndarray:
        """Load and preprocess image with error handling."""
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"Failed to load image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _load_mask(self, path: Path) -> np.ndarray:
        """Load mask with error handling."""
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise IOError(f"Failed to load mask: {path}")
        return mask

    def _map_ids(self, mask: np.ndarray) -> np.ndarray:
        """Map segmentation IDs to training IDs efficiently."""
        output = np.zeros_like(mask)
        for k, v in self.config.id_to_trainid.items():
            output[mask == k] = v
        return output


@dataclass
class DataloaderConfig:
    """Configuration for data loaders."""

    batch_size: int = 32
    val_batch_size: int = 1
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2


def build_dataloaders(
    config: DataloaderConfig = DataloaderConfig(),
) -> Tuple[DataLoader, DataLoader]:
    """
    Build training and validation dataloaders.

    Args:
        config (DataloaderConfig): Dataloader configuration

    Returns:
        Tuple[DataLoader, DataLoader]: Training and validation dataloaders
    """
    train_dataset = TPADataset(mode="train")
    val_dataset = TPADataset(mode="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        prefetch_factor=config.prefetch_factor,
        drop_last=True,
        persistent_workers=True,  # Keep workers alive between epochs
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.val_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        prefetch_factor=config.prefetch_factor,
        drop_last=False,
        persistent_workers=True,
    )

    return train_loader, val_loader


class DatasetVisualizer:
    """
    Utility class for visualizing samples from the TPA Dataset.

    Attributes:
        class_names (List[str]): List of class names
        colors (List[Tuple[float, float, float]]): RGB colors for each class
    """

    def __init__(self):
        self.class_names = ["background", "trailer", "trailer_bar", "trailer_ball"]
        # Define distinct colors for each class (background, trailer, trailer_parts)
        self.colors = [
            (0.1, 0.1, 0.1),  # Dark gray for background
            (0.2, 0.7, 0.2),  # Green for trailer
            (0.9, 0.3, 0.1),  # Red-orange for trailer bar
            (0.1, 0.3, 0.9),  # Blue for trailer ball
        ]

    def denormalize_image(self, img: torch.Tensor) -> np.ndarray:
        """
        Denormalize a PyTorch image tensor back to numpy array.

        Args:
            img (torch.Tensor): Normalized image tensor

        Returns:
            np.ndarray: Denormalized image array
        """
        mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)

        img = img.clone()
        img = img * std + mean
        img = img.clamp(0, 1)
        img = img.numpy().transpose(1, 2, 0)
        return img

    def create_mask_overlay(self, mask: np.ndarray) -> np.ndarray:
        """
        Create a colored overlay from a segmentation mask.

        Args:
            mask (np.ndarray): Segmentation mask

        Returns:
            np.ndarray: Colored mask overlay
        """
        h, w = mask.shape
        colored_mask = np.zeros((h, w, 3))

        for class_idx, color in enumerate(self.colors):
            colored_mask[mask == class_idx] = color

        return colored_mask

    def plot_samples(
        self, dataset: TPADataset, num_samples: int = 6, save_path: Optional[str] = None
    ) -> None:
        """
        Plot samples from the dataset with their corresponding masks.

        Args:
            dataset (TPADataset): Dataset to visualize
            num_samples (int): Number of samples to plot
            save_path (Optional[str]): Path to save the plot
        """
        num_cols = 3
        # num_rows = (num_samples + num_cols - 1) // num_cols
        fig, axes = plt.subplots(num_samples, num_cols, figsize=(15, 5 * num_samples))
        axes = axes.flatten()

        # Create legend patches
        patches = [
            mpatches.Patch(color=color, label=name)
            for color, name in zip(self.colors, self.class_names)
        ]

        for idx in range(num_samples):
            img, mask, img_id = dataset[idx]

            # Get denormalized image and colored mask
            img_np = self.denormalize_image(img)
            mask_overlay = self.create_mask_overlay(mask.numpy())

            # Plot original image
            axes[idx * num_cols].imshow(img_np)
            axes[idx * num_cols].set_title(f"Sample {idx + 1} - Original")
            axes[idx * num_cols].axis("off")

            # Plot mask overlay
            axes[idx * num_cols + 1].imshow(mask_overlay)
            axes[idx * num_cols + 1].set_title(f"Sample {idx + 1} - Segmentation")
            axes[idx * num_cols + 1].axis("off")

            # Plot blended image
            blended = 0.7 * img_np + 0.3 * mask_overlay
            blended = np.clip(blended, 0, 1)
            axes[idx * num_cols + 2].imshow(blended)
            axes[idx * num_cols + 2].set_title(f"Sample {idx + 1} - Blended")
            axes[idx * num_cols + 2].axis("off")

        # Remove empty subplots
        for idx in range(num_samples * 3, len(axes)):
            fig.delaxes(axes[idx])

        # Add legend
        fig.legend(handles=patches, loc="center right", bbox_to_anchor=(0.98, 0.5))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            plt.close()
        else:
            plt.show()

    def plot_augmentations(
        self,
        dataset: TPADataset,
        sample_idx: int = 0,
        num_augmentations: int = 8,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot multiple augmentations of the same sample.

        Args:
            dataset (TPADataset): Dataset to visualize
            sample_idx (int): Index of the sample to augment
            num_augmentations (int): Number of augmentations to show
            save_path (Optional[str]): Path to save the plot
        """
        num_cols = 4
        num_rows = (num_augmentations + num_cols - 1) // num_cols
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
        axes = axes.flatten()

        # Get the original sample
        orig_img, orig_mask, orig_img_id = dataset[sample_idx]
        orig_img_np = self.denormalize_image(orig_img)
        orig_mask_overlay = self.create_mask_overlay(orig_mask.numpy())

        # Plot original
        axes[0].imshow(orig_img_np)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        # Plot original mask
        axes[1].imshow(orig_mask_overlay)
        axes[1].set_title("Original Mask")
        axes[1].axis("off")

        # Create and plot augmentations
        for idx in range(2, num_augmentations):
            img, mask, img_id = dataset[sample_idx]  # This will apply random augmentations
            img_np = self.denormalize_image(img)

            axes[idx].imshow(img_np)
            axes[idx].set_title(f"Augmentation {idx-1}")
            axes[idx].axis("off")

        # Remove empty subplots
        for idx in range(num_augmentations, len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            plt.close()
        else:
            plt.show()


def visualize_dataset_samples():
    """
    Visualize samples from the dataset with their segmentation masks.
    """
    # Create dataset
    dataset = TPADataset(mode="train")

    # Create visualizer
    visualizer = DatasetVisualizer()

    # Plot samples
    print("Plotting dataset samples...")
    visualizer.plot_samples(dataset, num_samples=5, save_path="dataset_samples.png")

    # Plot augmentations of a single sample
    print("Plotting augmentations...")
    visualizer.plot_augmentations(
        dataset, sample_idx=0, num_augmentations=8, save_path="augmentations.png"
    )


if __name__ == "__main__":
    visualize_dataset_samples()
