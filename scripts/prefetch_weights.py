"""
Prefetch all model weights on a host with internet access (your mac with VPN),
so the AutoDL training box never has to leave the firewall.

Downloads:
  - Ultralytics YOLO weights (yolov8s.pt, yolov5su.pt, yolov8l-world.pt)
  - torchvision Faster R-CNN R50-FPN COCO weights
  - (optional) Grounding DINO SwinB weights (already in legacy weights/)

Output:
  weights/cache/ultralytics/*.pt
  weights/cache/torchvision/*.pth
  weights/cache/MANIFEST.txt

This directory is then bundled into the AutoDL upload. The bootstrap script
on AutoDL points Ultralytics + torchvision at this cache via env vars and
symlinks, so no model weight ever needs to be re-downloaded.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

ULTRALYTICS_BASE = "https://github.com/ultralytics/assets/releases/download/v8.3.0"
ULTRALYTICS_WEIGHTS = (
    "yolov8s.pt",
    "yolov8n.pt",
    "yolov5su.pt",
    "yolov8l-world.pt",
)

TORCHVISION_WEIGHTS = (
    (
        "fasterrcnn_resnet50_fpn_coco-258fb6c6.pth",
        "https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth",
    ),
    (
        "resnet50-0676ba61.pth",
        "https://download.pytorch.org/models/resnet50-0676ba61.pth",
    ),
)


def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return
    print(f"[get ] {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=120) as r, tmp.open("wb") as f:
            shutil.copyfileobj(r, f, length=1 << 20)
        tmp.rename(dest)
        print(f"       -> {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("weights/cache"))
    return p.parse_args()


def main() -> None:
    args = parse_args()

    ultra_dir = args.out / "ultralytics"
    for name in ULTRALYTICS_WEIGHTS:
        download(f"{ULTRALYTICS_BASE}/{name}", ultra_dir / name)

    tv_dir = args.out / "torchvision"
    for name, url in TORCHVISION_WEIGHTS:
        download(url, tv_dir / name)

    manifest = args.out / "MANIFEST.txt"
    lines = []
    for sub in sorted(args.out.glob("*/")):
        for f in sorted(sub.glob("*")):
            if f.is_file():
                lines.append(f"{f.relative_to(args.out)} {f.stat().st_size}")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nDone. Cache at {args.out.resolve()}")
    print(f"Manifest:\n{manifest.read_text(encoding='utf-8')}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
