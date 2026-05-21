"""
Drive the four-row YOLOv8 ablation matrix in one shot.

Runs `scripts/train.py` for: baseline, focal, focal_ohem, full — each into
its own `runs/detect/<variant>` subdirectory. After all four finish,
launches `scripts/eval.py --all-variants` to assemble a single CSV.

Usage:
    uv run scripts/ablation.py --epochs 100
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from loguru import logger

ABLATION_VARIANTS = ("baseline", "focal", "focal_ohem", "full")


def run_one(variant: str, args: argparse.Namespace) -> int:
    cmd = [
        sys.executable, "scripts/train.py",
        "--variant", variant,
        "--data", str(args.data),
        "--imgsz", str(args.imgsz),
        "--epochs", str(args.epochs),
        "--batch", str(args.batch),
        "--device", args.device,
        "--workers", str(args.workers),
        "--seed", str(args.seed),
        "--patience", str(args.patience),
        "--project", str(args.project),
    ]
    logger.info(f"[ablation] starting variant={variant}")
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        logger.error(f"[ablation] variant={variant} failed (code={completed.returncode})")
    return completed.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/cumt_belt_yolo/data.yaml"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--project", type=Path, default=Path("runs/detect"))
    parser.add_argument(
        "--skip-eval", action="store_true",
        help="Don't auto-run scripts/eval.py after the matrix.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    failures: list[str] = []
    for variant in ABLATION_VARIANTS:
        if run_one(variant, args) != 0:
            failures.append(variant)

    if failures:
        logger.error(f"Ablation incomplete; failed variants: {failures}")
        sys.exit(1)

    logger.info("All four ablation variants finished.")

    if not args.skip_eval:
        eval_cmd = [
            sys.executable, "scripts/eval.py",
            "--data", str(args.data),
            "--project", str(args.project),
            "--device", args.device,
            "--imgsz", str(args.imgsz),
            "--all-variants",
        ]
        logger.info(f"Running: {' '.join(eval_cmd)}")
        subprocess.run(eval_cmd, check=False)


if __name__ == "__main__":
    main()
