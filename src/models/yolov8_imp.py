"""
Importance-sampling enhancements for Ultralytics YOLOv8.

This module exposes three independent toggles plumbed through Ultralytics'
internals without forking them:

- `use_focal`: replace the BCE classification loss in `v8DetectionLoss`
  with `SigmoidFocalLoss` from `src/losses/focal_loss.py`.
- `use_ohem`: mask per-anchor classification loss to keep only positives
  plus the top-k hardest negatives (`src/samplers/ohem.py`).
- `use_wsampler`: install a `WeightedRandomSampler` over the training
  DataLoader so background-only images contribute less per epoch
  (`src/samplers/weighted_sampler.py`).

Public entry point: `apply_improvements(model, ...)` returns the same model
instance with patches active for that interpreter session. Patches are
installed lazily and are idempotent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from loguru import logger

from src.losses.focal_loss import SigmoidFocalLoss
from src.samplers.ohem import OHEMMask
from src.samplers.weighted_sampler import build_weighted_sampler


@dataclass
class ImprovementConfig:
    use_focal: bool = False
    use_ohem: bool = False
    use_wsampler: bool = False
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    ohem_neg_pos_ratio: float = 3.0
    sampler_positive_weight: float = 1.0
    sampler_background_weight: float = 0.3


def _patch_v8_detection_loss(cfg: ImprovementConfig) -> None:
    """Swap the BCE call inside `v8DetectionLoss.__call__` for our Focal+OHEM combo."""
    from ultralytics.utils.loss import v8DetectionLoss

    if getattr(v8DetectionLoss, "_imp_patched", False):
        return

    focal_module = SigmoidFocalLoss(alpha=cfg.focal_alpha, gamma=cfg.focal_gamma)
    ohem_module = OHEMMask(neg_pos_ratio=cfg.ohem_neg_pos_ratio)
    use_focal = cfg.use_focal
    use_ohem = cfg.use_ohem

    original_init = v8DetectionLoss.__init__

    def patched_init(self, model, tal_topk: int = 10):  # noqa: D401
        original_init(self, model, tal_topk=tal_topk)
        device = getattr(self, "device", torch.device("cpu"))
        self._focal = focal_module.to(device)
        self._ohem = ohem_module.to(device)
        self._use_focal_imp = use_focal
        self._use_ohem_imp = use_ohem

    def cls_loss(self, pred_scores: torch.Tensor, target_scores: torch.Tensor) -> torch.Tensor:
        """
        Replacement for the inline `self.bce(pred_scores, target_scores).sum()` call.
        Returns a scalar (sum reduction) divided by `target_scores_sum` upstream.
        """
        if self._use_focal_imp:
            per = self._focal(pred_scores, target_scores)
        else:
            per = torch.nn.functional.binary_cross_entropy_with_logits(
                pred_scores, target_scores, reduction="none"
            )
        if self._use_ohem_imp:
            mask = self._ohem(per, target_scores)
            per = per * mask
        return per.sum()

    original_call = v8DetectionLoss.__call__

    def patched_call(self, preds, batch):
        bce_backup = self.bce

        class _BCEShim:
            def __init__(self, owner):
                self.owner = owner

            def __call__(self, pred_scores, target_scores):
                return cls_loss(self.owner, pred_scores, target_scores)

        self.bce = _BCEShim(self)
        try:
            return original_call(self, preds, batch)
        finally:
            self.bce = bce_backup

    v8DetectionLoss.__init__ = patched_init
    v8DetectionLoss.__call__ = patched_call
    v8DetectionLoss._imp_patched = True
    logger.info(
        "Patched v8DetectionLoss with "
        f"focal={cfg.use_focal} (α={cfg.focal_alpha}, γ={cfg.focal_gamma}), "
        f"ohem={cfg.use_ohem} (neg_pos_ratio={cfg.ohem_neg_pos_ratio})"
    )


def _patch_dataloader_sampler(cfg: ImprovementConfig) -> None:
    """Wrap `ultralytics.data.build.build_dataloader` to inject WeightedRandomSampler."""
    import ultralytics.data.build as build_mod

    if getattr(build_mod, "_imp_sampler_patched", False):
        return

    original_build = build_mod.build_dataloader
    pos_w = cfg.sampler_positive_weight
    bg_w = cfg.sampler_background_weight

    def patched_build_dataloader(dataset, batch, workers, shuffle: bool = True, rank: int = -1):
        if not shuffle:
            return original_build(dataset, batch, workers, shuffle=shuffle, rank=rank)

        try:
            image_paths = list(getattr(dataset, "im_files", []))
        except Exception as exc:
            logger.warning(f"Could not introspect dataset for weighted sampler: {exc}")
            return original_build(dataset, batch, workers, shuffle=shuffle, rank=rank)

        if not image_paths:
            return original_build(dataset, batch, workers, shuffle=shuffle, rank=rank)

        sampler = build_weighted_sampler(
            image_paths,
            positive_weight=pos_w,
            background_weight=bg_w,
        )
        from torch.utils.data import DataLoader

        nd = torch.cuda.device_count()
        nw = min([workers, batch if batch > 1 else 0, 8])
        if rank != -1:
            nw = 0

        return DataLoader(
            dataset=dataset,
            batch_size=batch,
            sampler=sampler,
            shuffle=False,
            num_workers=nw,
            pin_memory=True,
            collate_fn=getattr(dataset, "collate_fn", None),
            persistent_workers=False if nw == 0 else True,
        )

    build_mod.build_dataloader = patched_build_dataloader
    build_mod._imp_sampler_patched = True
    logger.info(
        f"Patched build_dataloader with WeightedRandomSampler "
        f"(positive={pos_w}, background={bg_w})"
    )


def apply_improvements(cfg: ImprovementConfig) -> None:
    """Install whichever patches `cfg` enables. Safe to call multiple times."""
    if cfg.use_focal or cfg.use_ohem:
        _patch_v8_detection_loss(cfg)
    if cfg.use_wsampler:
        _patch_dataloader_sampler(cfg)


def config_from_variant(variant: str) -> ImprovementConfig:
    """Map a CLI `--variant` flag to an ImprovementConfig."""
    table: dict[str, dict[str, Any]] = {
        "baseline": {},
        "focal": {"use_focal": True},
        "focal_ohem": {"use_focal": True, "use_ohem": True},
        "full": {"use_focal": True, "use_ohem": True, "use_wsampler": True},
    }
    if variant not in table:
        raise ValueError(
            f"Unknown YOLOv8 variant '{variant}'. "
            f"Choose from: {sorted(table)}"
        )
    return ImprovementConfig(**table[variant])
