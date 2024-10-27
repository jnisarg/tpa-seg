import logging
from typing import Dict, List

import torch
from rich.console import Console
from rich.table import Table


class MetricsCalculator:
    """Handles calculation and aggregation of evaluation metrics"""

    def __init__(self, n_classes: int, class_names: List[str], device: torch.device):
        self.n_classes = n_classes
        self.class_names = class_names
        self.device = device
        self.reset()

    def reset(self):
        """Resets all metric counters"""
        self.confusion_matrix = torch.zeros(
            (self.n_classes, self.n_classes), dtype=torch.int64
        ).to(self.device)
        self.total_loss = 0.0
        self.batch_count = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor, loss: float = None):
        """Updates metrics with batch results"""
        # Update confusion matrix
        # for t, p in zip(target.flatten(), pred.flatten()):
        #     self.confusion_matrix[t, p] += 1

        pred = pred.view(-1)
        target = target.view(-1)
        self.confusion_matrix += torch.bincount(
            target * self.n_classes + pred, minlength=self.n_classes**2
        ).reshape(self.n_classes, self.n_classes)

        # Update loss if provided
        if loss is not None:
            self.total_loss += loss
            self.batch_count += 1

    def compute_metrics(self) -> Dict[str, float]:
        """Computes final metrics from accumulated statistics"""
        # Per-class metrics
        per_class_metrics = {}
        for i in range(self.n_classes):
            tp = self.confusion_matrix[i, i]
            fp = self.confusion_matrix[:, i].sum() - tp
            fn = self.confusion_matrix[i, :].sum() - tp

            # IoU for each class
            iou = tp / (tp + fp + fn + 1e-10)
            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)

            class_name = (
                self.class_names[i] if i < len(self.class_names) else f"Class {i}"
            )
            per_class_metrics[class_name] = {
                "iou": iou,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

        # Global metrics
        total_tp = torch.diag(self.confusion_matrix).sum()
        total_pixels = self.confusion_matrix.sum()

        metrics = {
            "per_class": per_class_metrics,
            "mean_iou": torch.mean(
                torch.tensor([m["iou"] for m in per_class_metrics.values()])
            ),
            "pixel_accuracy": total_tp / total_pixels,
            "mean_loss": (
                self.total_loss / self.batch_count if self.batch_count > 0 else 0
            ),
        }

        return metrics

    @staticmethod
    def log_metrics(metrics: Dict[str, float], console: Console):
        """Logs evaluation metrics in a formatted table"""
        console.print("\n📊 Evaluation Results:")

        # Create and populate metrics table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="dim")
        table.add_column("Value", justify="right")

        # Global metrics
        table.add_row("Mean IoU", f"{metrics['mean_iou']:.4f}")
        table.add_row("Pixel Accuracy", f"{metrics['pixel_accuracy']:.4f}")
        table.add_row("Mean Loss", f"{metrics['mean_loss']:.4f}")

        console.print(table)

        # Create per-class metrics table
        class_table = Table(show_header=True, header_style="bold magenta")
        class_table.add_column("Class")
        class_table.add_column("IoU", justify="right")
        class_table.add_column("Precision", justify="right")
        class_table.add_column("Recall", justify="right")
        class_table.add_column("F1", justify="right")

        for class_name, class_metrics in metrics["per_class"].items():
            class_table.add_row(
                class_name,
                f"{class_metrics['iou']:.4f}",
                f"{class_metrics['precision']:.4f}",
                f"{class_metrics['recall']:.4f}",
                f"{class_metrics['f1']:.4f}",
            )

        console.print("\n📊 Per-Class Metrics:")
        console.print(class_table)
