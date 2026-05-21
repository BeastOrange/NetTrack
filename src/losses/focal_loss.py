"""
Sigmoid Focal Loss for the YOLOv8 classification head.

YOLOv8's default `v8DetectionLoss` uses `BCEWithLogitsLoss(reduction="none")`
on per-anchor class logits, then averages per-anchor and per-class. We swap
that single line for Focal Loss while preserving the rest of the pipeline,
so the behaviour outside the classification head is unchanged.

Default α and γ follow Lin et al. 2017 (RetinaNet): α=0.25, γ=2.0.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SigmoidFocalLoss(nn.Module):
    """
    Compute a per-element focal loss; reduction is left to the caller so the
    drop-in replacement for `nn.BCEWithLogitsLoss(reduction="none")` is
    behaviourally identical apart from the focal weighting.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0,1], got {alpha}")
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: raw class logits, shape (..., C).
            targets: same shape as logits, values in [0, 1].
        Returns:
            Per-element focal loss, same shape as logits.
        """
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        modulating = (1.0 - p_t).pow(self.gamma)
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            return alpha_t * modulating * bce
        return modulating * bce
