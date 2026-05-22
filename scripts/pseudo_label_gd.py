"""
Grounding-DINO zero-shot pseudo-labeling for CUMT-BelT.

Used instead of pseudo_label.py (YOLO-World) because YOLO-World did not fire
on this underground-coal-mine domain. GD-SwinB does, with these operating
thresholds (empirically tuned on a held-out subset):

  锚杆  (anchor rod):  box_thresh=0.25  text_thresh=0.20  → ~6-10 boxes/image
  大块  (large coal):  box_thresh=0.22  text_thresh=0.18  → ~1-3 boxes/image
  正常煤流 (normal flow): treated as background, empty label written.

Outputs per-image YOLO label files and an images mirror (symlinked).
Class IDs are hard-assigned from the source directory, not from the caption
phrases — phrase resolution from GD is noisy.

Pipeline:
  CUMT-BelT/<split>/<chinese_class_dir>/*.jpg
    → run GD with class-specific caption
    → filter aspect ratio + area outliers
    → write <stem>.txt with `cls cx cy w h` (normalized)

This script depends on the legacy GD package layout: it injects
`legacy/` onto sys.path so `from groundingdino.util.inference import ...` works
without any pip install. The MS-DeformAttn CUDA extension is not required —
ms_deform_attn.py has been patched to always use the pure-PyTorch fallback.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "legacy"))


@dataclass(frozen=True)
class ClassSpec:
    class_id: int
    source_dir: str
    caption: str
    box_thresh: float
    text_thresh: float
    is_background: bool = False


CLASS_SPECS: tuple[ClassSpec, ...] = (
    ClassSpec(
        class_id=0,
        source_dir="锚杆",
        caption="anchor rod . metal bolt . steel rod .",
        box_thresh=0.25,
        text_thresh=0.20,
    ),
    ClassSpec(
        class_id=1,
        source_dir="大块",
        caption="large coal block . large coal lump . big rock .",
        box_thresh=0.22,
        text_thresh=0.18,
    ),
    ClassSpec(
        class_id=-1,
        source_dir="正常煤流",
        caption="",
        box_thresh=0.0,
        text_thresh=0.0,
        is_background=True,
    ),
)


def _filter_box_norm(
    cx: float, cy: float, bw: float, bh: float,
    img_w: int, img_h: int,
    aspect_min: float = 1 / 8,
    aspect_max: float = 8.0,
    area_min_ratio: float = 1e-3,
    area_max_ratio: float = 0.9,
) -> bool:
    if bw <= 1.0 / img_w or bh <= 1.0 / img_h:
        return False
    aspect = bw / bh if bh > 0 else 0.0
    if aspect < aspect_min or aspect > aspect_max:
        return False
    if bw * bh < area_min_ratio or bw * bh > area_max_ratio:
        return False
    return True


def _clamp(v: float) -> float:
    return min(max(v, 0.0), 1.0)


def load_gd(config: Path, weights: Path, device: str):
    from groundingdino.util.inference import load_model
    model = load_model(str(config), str(weights), device=device)
    model = model.to(device)
    model.eval()
    return model


def predict_gd(model, image_path: Path, caption: str, box_thresh: float,
               text_thresh: float, device: str) -> np.ndarray:
    """Returns boxes in (cx, cy, w, h) normalized form, filtered by box_thresh."""
    from groundingdino.util.inference import load_image, predict
    _, image = load_image(str(image_path))
    image = image.to(device)
    with torch.no_grad():
        boxes, logits, _ = predict(
            model=model, image=image, caption=caption,
            box_threshold=box_thresh, text_threshold=text_thresh,
            device=device,
        )
    if len(boxes) == 0:
        return np.empty((0, 4), dtype=np.float32)
    return boxes.cpu().numpy().astype(np.float32)


def label_one(model, image_path: Path, spec: ClassSpec, device: str) -> list[str]:
    if spec.is_background:
        return []
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        logger.warning(f"unreadable: {image_path}")
        return []
    img_h, img_w = image_bgr.shape[:2]

    boxes_cxcywh = predict_gd(
        model, image_path, spec.caption, spec.box_thresh, spec.text_thresh, device
    )
    lines: list[str] = []
    for cx, cy, bw, bh in boxes_cxcywh:
        cx, cy, bw, bh = float(cx), float(cy), float(bw), float(bh)
        if not _filter_box_norm(cx, cy, bw, bh, img_w, img_h):
            continue
        lines.append(
            f"{spec.class_id} {_clamp(cx):.6f} {_clamp(cy):.6f} "
            f"{_clamp(bw):.6f} {_clamp(bh):.6f}"
        )
    return lines


def mirror_image(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        dst.symlink_to(src.resolve())
    except (OSError, NotImplementedError):
        import shutil
        shutil.copy2(src, dst)


def run_split(model, src_split: Path, out_labels: Path, out_images: Path, device: str):
    out_labels.mkdir(parents=True, exist_ok=True)
    out_images.mkdir(parents=True, exist_ok=True)

    totals = {"images": 0, "boxes": 0, "bg": 0, "no_box": 0, "secs": 0.0}

    for spec in CLASS_SPECS:
        class_dir = src_split / spec.source_dir
        if not class_dir.is_dir():
            logger.warning(f"missing: {class_dir}")
            continue
        images = sorted(class_dir.glob("*.jpg"))
        logger.info(
            f"[{src_split.name}/{spec.source_dir}] {len(images)} imgs "
            f"class_id={'bg' if spec.is_background else spec.class_id} "
            f"box={spec.box_thresh} text={spec.text_thresh}"
        )

        t0 = time.time()
        for image_path in tqdm(images, desc=spec.source_dir, leave=False):
            mirror_image(image_path, out_images / image_path.name)
            lines = label_one(model, image_path, spec, device)
            (out_labels / f"{image_path.stem}.txt").write_text(
                "\n".join(lines), encoding="utf-8"
            )
            totals["images"] += 1
            totals["boxes"] += len(lines)
            if spec.is_background:
                totals["bg"] += 1
            elif not lines:
                totals["no_box"] += 1
        totals["secs"] += time.time() - t0

    return totals


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, default=Path("CUMT-BelT/训练集"))
    p.add_argument("--out-labels", type=Path,
                   default=Path("data/cumt_belt_yolo/labels/_train_raw"))
    p.add_argument("--out-images", type=Path,
                   default=Path("data/cumt_belt_yolo/images/_train_raw"))
    p.add_argument("--config", type=Path,
                   default=Path("weights/groundingdino/GroundingDINO_SwinB_cfg.py"))
    p.add_argument("--weights", type=Path,
                   default=Path("weights/groundingdino/groundingdino_swinb_cogcoor.pth"))
    p.add_argument("--device", type=str, default="cuda:0")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.src.is_dir():
        raise SystemExit(f"source split not found: {args.src}")

    logger.info(f"loading GD: {args.weights}")
    model = load_gd(args.config, args.weights, args.device)
    logger.info("loaded.")

    totals = run_split(
        model, args.src, args.out_labels, args.out_images, args.device,
    )
    logger.info(
        "pseudo-label done. "
        f"images={totals['images']}, boxes={totals['boxes']}, "
        f"no_box_imgs={totals['no_box']}, bg_imgs={totals['bg']}, "
        f"foreground_secs={totals['secs']:.1f} "
        f"({totals['secs'] / max(1, totals['images'] - totals['bg']):.2f} s/img)"
    )


if __name__ == "__main__":
    main()
