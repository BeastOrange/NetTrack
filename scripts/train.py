"""
Unified training entrypoint for the comparison + ablation matrix.

Variants:
    yolov8 baselines / improvements (this script):
        --variant baseline      YOLOv8s, no enhancements
        --variant focal         YOLOv8s + Focal Loss
        --variant focal_ohem    YOLOv8s + Focal + OHEM
        --variant full          YOLOv8s + Focal + OHEM + WeightedSampler

    Comparison group:
        --variant yolov5        YOLOv5s via Ultralytics
        --variant frcnn         Faster R-CNN R50-FPN via torchvision

All YOLO variants share the same data.yaml / imgsz / epochs / batch so
the matrix is apples-to-apples. Faster R-CNN reads the same YOLO labels
through a thin adapter.

This script must run on a CUDA host (e.g. SSH to the user's Windows box).
On macOS without CUDA it will surface the right command rather than
silently fall back.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from loguru import logger

from src.models.yolov8_imp import apply_improvements, config_from_variant


YOLOV8_VARIANTS = {"baseline", "focal", "focal_ohem", "full"}


def _check_cuda(device: str) -> None:
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.error("CUDA is not available on this host.")
        logger.error("Run training on the remote Windows box instead, e.g.:")
        logger.error("  ssh win 'cd ~/repo && uv run scripts/train.py ...'")
        sys.exit(2)


def train_yolov8(args: argparse.Namespace) -> None:
    from ultralytics import YOLO

    cfg = config_from_variant(args.variant)
    logger.info(f"YOLOv8 variant={args.variant}, improvements={cfg}")
    apply_improvements(cfg)

    model = YOLO(args.weights or "yolov8s.pt")
    model.train(
        data=str(args.data),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        project=str(args.project),
        name=args.variant,
        workers=args.workers,
        seed=args.seed,
        deterministic=True,
        pretrained=True,
        exist_ok=True,
        patience=args.patience,
    )


def train_yolov5(args: argparse.Namespace) -> None:
    """YOLOv5s via Ultralytics' multi-model registry (same training API)."""
    from ultralytics import YOLO

    model = YOLO(args.weights or "yolov5su.pt")
    model.train(
        data=str(args.data),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        project=str(args.project),
        name="yolov5",
        workers=args.workers,
        seed=args.seed,
        deterministic=True,
        pretrained=True,
        exist_ok=True,
        patience=args.patience,
    )


def train_frcnn(args: argparse.Namespace) -> None:
    """Faster R-CNN R50-FPN via torchvision; lightweight loop matching the YOLO matrix."""
    from src.training.frcnn_runner import run_frcnn_training

    run_frcnn_training(
        data_yaml=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        project=args.project,
        seed=args.seed,
        workers=args.workers,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        type=str,
        required=True,
        choices=sorted(YOLOV8_VARIANTS | {"yolov5", "frcnn"}),
    )
    parser.add_argument(
        "--data", type=Path,
        default=Path("data/cumt_belt_yolo/data.yaml"),
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument(
        "--project", type=Path, default=Path("runs/detect"),
        help="Output directory under which each variant gets its own subdir.",
    )
    parser.add_argument(
        "--weights", type=str, default=None,
        help="Optional pretrained checkpoint override.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise SystemExit(
            f"data.yaml not found at {args.data}. "
            "Run scripts/prepare_dataset.py first."
        )

    _check_cuda(args.device)

    if args.variant in YOLOV8_VARIANTS:
        train_yolov8(args)
    elif args.variant == "yolov5":
        train_yolov5(args)
    elif args.variant == "frcnn":
        train_frcnn(args)
    else:
        raise SystemExit(f"Unhandled variant: {args.variant}")


if __name__ == "__main__":
    main()
