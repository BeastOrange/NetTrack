"""
Run the full 6-row comparison + ablation matrix sequentially.

Order chosen so the most-likely-to-fail (frcnn, custom loop) runs last —
if YOLO variants succeed first we keep their results even if frcnn breaks.

Each variant logs to its own file under `--log-dir`; per-variant exit codes
are captured but never abort the matrix. Final summary printed at end.

Usage:
    python scripts/run_matrix.py --epochs 100
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from loguru import logger

# (variant, friendly_name)
MATRIX = (
    ("baseline", "YOLOv8s vanilla"),
    ("focal", "YOLOv8s + Focal"),
    ("focal_ohem", "YOLOv8s + Focal + OHEM"),
    ("full", "YOLOv8s + Focal + OHEM + WSampler"),
    ("yolov5", "YOLOv5s (Ultralytics)"),
    ("frcnn", "Faster R-CNN R50-FPN"),
)


def run_one(variant: str, args: argparse.Namespace) -> tuple[int, float]:
    log_path = args.log_dir / f"{variant}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
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
    logger.info(f"[matrix] {variant}  log -> {log_path}")
    t0 = time.time()
    with log_path.open("w") as log_f:
        completed = subprocess.run(cmd, check=False, stdout=log_f, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    return completed.returncode, elapsed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/cumt_belt_yolo/data.yaml"))
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--project", type=Path, default=Path("runs/matrix"))
    p.add_argument("--log-dir", type=Path, default=Path("runs/matrix/_logs"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results: list[tuple[str, int, float]] = []
    for variant, friendly in MATRIX:
        rc, secs = run_one(variant, args)
        status = "OK" if rc == 0 else f"FAIL(rc={rc})"
        logger.info(f"[matrix] {variant:12s} ({friendly:36s}) {status:14s} elapsed={secs / 60:.1f}min")
        results.append((variant, rc, secs))

    logger.info("=" * 70)
    logger.info("matrix summary:")
    for variant, rc, secs in results:
        marker = "✓" if rc == 0 else "✗"
        logger.info(f"  {marker}  {variant:12s}  rc={rc:>3}  {secs / 60:>5.1f} min")
    fails = [v for v, rc, _ in results if rc != 0]
    if fails:
        logger.error(f"FAILED: {fails}")
        sys.exit(1)
    logger.info("all variants succeeded.")


if __name__ == "__main__":
    main()
