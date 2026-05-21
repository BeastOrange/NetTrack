"""
Online Hard Example Mining (OHEM) for the YOLOv8 classification loss.

After per-anchor classification loss is computed (Focal or BCE), only the
top-k anchors by loss magnitude are retained for backprop. Positive anchors
(rows where any class target > 0) are always retained — OHEM only prunes
"easy negatives" by sorting all negative anchors by loss and keeping the
top `neg_pos_ratio * num_positives`.

If there are no positives in a batch, OHEM falls back to keeping the global
top-k negatives (k clamped to a configurable minimum) so the loss signal
is not zero.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class OHEMMask(nn.Module):
    """
    Build a 0/1 mask over per-anchor losses such that:
    - All anchors that contain at least one positive target are kept.
    - The hardest `neg_pos_ratio * num_positives` negatives are kept.
    - Everything else is masked out.

    Apply by element-wise multiplying with the per-anchor loss tensor before
    summing/averaging.
    """

    def __init__(self, neg_pos_ratio: float = 3.0, min_keep: int = 16) -> None:
        super().__init__()
        if neg_pos_ratio <= 0:
            raise ValueError(f"neg_pos_ratio must be > 0, got {neg_pos_ratio}")
        if min_keep < 1:
            raise ValueError(f"min_keep must be >= 1, got {min_keep}")
        self.neg_pos_ratio = neg_pos_ratio
        self.min_keep = min_keep

    @torch.no_grad()
    def forward(
        self,
        per_anchor_loss: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            per_anchor_loss: shape (B, A) or (B, A, C); reduced to (B, A) by
                summing over the class dim if needed.
            targets: same shape as per_anchor_loss (or pre-summed), used to
                identify positive vs negative anchors.
        Returns:
            Mask of same shape as `per_anchor_loss` (broadcasting-friendly).
        """
        if per_anchor_loss.dim() == 3:
            anchor_loss = per_anchor_loss.sum(dim=-1)
            anchor_targets = targets.sum(dim=-1)
        else:
            anchor_loss = per_anchor_loss
            anchor_targets = targets if targets.dim() == 2 else targets.sum(dim=-1)

        positive_mask = anchor_targets > 0
        negative_mask = ~positive_mask

        num_pos = int(positive_mask.sum().item())
        target_neg = max(self.min_keep, int(num_pos * self.neg_pos_ratio))

        keep_mask = positive_mask.clone()

        if negative_mask.any():
            neg_losses = anchor_loss.masked_fill(~negative_mask, float("-inf"))
            flat = neg_losses.flatten()
            k = min(target_neg, int(negative_mask.sum().item()))
            if k > 0:
                topk_idx = torch.topk(flat, k=k, sorted=False).indices
                flat_keep = torch.zeros_like(flat, dtype=torch.bool)
                flat_keep[topk_idx] = True
                keep_mask = keep_mask | flat_keep.view_as(anchor_loss)

        if per_anchor_loss.dim() == 3:
            keep_mask = keep_mask.unsqueeze(-1).expand_as(per_anchor_loss)
        return keep_mask.to(per_anchor_loss.dtype)
