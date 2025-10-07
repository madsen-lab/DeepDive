from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyWithMasking(torch.nn.Module):
    def __init__(self, reduction="mean", mask_value=-999):
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError('Reduction must be one of: "mean", "sum", "none".')

        self.reduction = reduction
        self.mask_value = mask_value

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

        # Mask values matching the mask value
        valid_mask = targets != self.mask_value  # .float()

        # Make the targets fit the mask value, insert 0 - to stay inside the bounds of the predictions
        targets = torch.where(
            valid_mask,
            targets,
            torch.tensor(0, device=targets.device, dtype=targets.dtype),
        )

        # Calculate cross-entropy loss only on valid samples
        loss = F.cross_entropy(predictions, targets, reduction="none")

        # Apply mask: zero out loss for invalid samples
        loss = loss * valid_mask.float()

        # Reduce the loss (mean or sum), considering only valid samples
        if self.reduction == "mean":
            valid_count = valid_mask.sum().float()
            return loss.sum() / valid_count.clamp(min=1)
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class MaskedFocalLoss(nn.Module):
    """Inspired by https://github.com/AdeelH/pytorch-multi-class-focal-loss/blob/master/focal_loss.py

    Focal Loss, as described in https://arxiv.org/abs/1708.02002.
    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.
    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        mask_value=-999,
    ):
        """
        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
        """
        if reduction not in ("mean", "sum", "none"):
            raise ValueError('Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.mask_value = mask_value

        self.nll_loss = nn.NLLLoss(weight=alpha, reduction="none")

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if len(targets) == 0:
            return torch.tensor(0.0)

        # Mask values matching the mask value
        valid_mask = targets != self.mask_value  # .float()
        targets_drop_masked = torch.where(
            valid_mask,
            targets,
            torch.tensor(0, device=targets.device, dtype=targets.dtype),
        )

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(predictions, dim=-1)
        ce = self.nll_loss(log_p, targets_drop_masked)

        # get true class column from each row
        all_rows = torch.arange(len(predictions))
        log_pt = log_p[all_rows, targets_drop_masked]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt) ** self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        # Apply mask: zero out loss for invalid samples
        loss = loss * valid_mask

        if self.reduction == "mean":
            valid_count = valid_mask.sum().float()
            loss = loss.sum() / valid_count.clamp(min=1)
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class MSELossWithMask(nn.Module):
    def __init__(self, reduction="mean", mask_value=-999):
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError('Reduction must be one of: "mean", "sum", "none".')
        self.reduction = reduction
        self.mask_value = mask_value

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Compute the squared difference
        squared_diff = F.mse_loss(predictions, targets, reduction="none")

        if self.mask_value is torch.nan:
            mask = ~torch.isnan(targets)
        else:
            mask = targets != self.mask_value
        # Apply mask: only consider valid (non-missing) values
        masked_squared_diff = squared_diff * mask

        # Calculate the mean or sum of the masked values
        if self.reduction == "mean":
            valid_count = mask.sum().float()
            return masked_squared_diff.sum() / valid_count.clamp(min=1)
        elif self.reduction == "sum":
            return masked_squared_diff.sum()
        else:
            return masked_squared_diff
