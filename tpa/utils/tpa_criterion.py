import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["TPACriterion"]


class TPACriterion(nn.Module):
    """
    Implements Online Hard Example Mining (OHEM) for Cross Entropy Loss.
    """

    def __init__(self, top_k_percent: float = 0.8):
        super().__init__()

        assert 0.5 < top_k_percent <= 1.0, "top_k_percent must be between 0.5 and 1.0"

        self.top_k_percent = top_k_percent

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = preds.size(1)
        preds = preds.permute(0, 2, 3, 1).reshape(-1, n_classes)
        targets = targets.view(-1)

        loss = F.cross_entropy(preds, targets, reduction="none")

        # Sort the loss to get the hardest pixels
        sorted_loss, _ = loss.sort(descending=True)
        num_keep = int(self.top_k_percent * sorted_loss.numel())
        top_loss = sorted_loss[:num_keep]

        return top_loss.mean()
