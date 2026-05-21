"""
WeightedRandomSampler integration for the Ultralytics YOLO data loader.

Goal: down-weight pure-background images (the "正常煤流" subset, marked by
empty .txt label files) so they don't dominate the training distribution
while still contributing as hard negatives.

Strategy: build a per-sample weight vector by reading each label file once,
then return a `torch.utils.data.WeightedRandomSampler` instance. The training
script monkey-patches `ultralytics.data.build.build_dataloader` to inject
this sampler when the `weighted_sampler` flag is on; that lets us avoid
forking Ultralytics internals while still getting a deterministic switch.
"""

from __future__ import annotations

from pathlib import Path

import torch
from loguru import logger
from torch.utils.data import WeightedRandomSampler


def _label_dir_for_image(image_path: Path) -> Path:
    """Mirror Ultralytics' images/<split>/ ↔ labels/<split>/ convention."""
    parts = list(image_path.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def is_background_label(label_path: Path) -> bool:
    """A label file is "background" if it is missing or has no non-empty lines."""
    if not label_path.exists():
        return True
    try:
        text = label_path.read_text(encoding="utf-8").strip()
    except OSError:
        return True
    return len(text) == 0


def compute_sample_weights(
    image_paths: list[str | Path],
    positive_weight: float = 1.0,
    background_weight: float = 0.3,
) -> torch.Tensor:
    if positive_weight <= 0 or background_weight <= 0:
        raise ValueError("Sample weights must be positive.")

    weights = torch.empty(len(image_paths), dtype=torch.float32)
    n_bg = 0
    for i, raw_path in enumerate(image_paths):
        label_path = _label_dir_for_image(Path(raw_path))
        if is_background_label(label_path):
            weights[i] = background_weight
            n_bg += 1
        else:
            weights[i] = positive_weight
    logger.info(
        f"WeightedRandomSampler: {n_bg}/{len(image_paths)} background images "
        f"(weight={background_weight}) vs {len(image_paths) - n_bg} positive "
        f"(weight={positive_weight})"
    )
    return weights


def build_weighted_sampler(
    image_paths: list[str | Path],
    positive_weight: float = 1.0,
    background_weight: float = 0.3,
    num_samples: int | None = None,
    replacement: bool = True,
) -> WeightedRandomSampler:
    weights = compute_sample_weights(image_paths, positive_weight, background_weight)
    return WeightedRandomSampler(
        weights=weights,
        num_samples=num_samples if num_samples is not None else len(weights),
        replacement=replacement,
    )
