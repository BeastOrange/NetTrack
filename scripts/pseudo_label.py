"""
YOLO-World zero-shot pseudo-labeling for the CUMT-BelT training split.

Pipeline:
- Walk CUMT-BelT/<split>/<chinese_class_dir>/*.jpg
- Run YOLO-World with a class-specific prompt
- Hard-assign class_id by source directory (not by prompt parsing)
- Filter aspect ratio + area outliers
- Write per-image YOLO labels: <class_id> <cx> <cy> <w> <h> (normalized)
- Empty label files for the "background" directory (正常煤流) so they act
  as hard-negative samples during training.

Run on macOS without CUDA: it falls back to MPS / CPU automatically.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
from loguru import logger
from tqdm import tqdm
from ultralytics import YOLOWorld


@dataclass(frozen=True)
class ClassSpec:
    class_id: int
    source_dir: str
    prompts: tuple[str, ...]
    is_background: bool = False


CLASS_SPECS: tuple[ClassSpec, ...] = (
    ClassSpec(
        class_id=0,
        source_dir="锚杆",
        prompts=("anchor rod", "metal bolt", "steel rod"),
    ),
    ClassSpec(
        class_id=1,
        source_dir="大块",
        prompts=("large coal block", "large coal lump", "big rock"),
    ),
    ClassSpec(
        class_id=-1,
        source_dir="正常煤流",
        prompts=(),
        is_background=True,
    ),
)


def _filter_box(
    box_xyxy: tuple[float, float, float, float],
    img_w: int,
    img_h: int,
    aspect_min: float = 1 / 8,
    aspect_max: float = 8.0,
    area_min_ratio: float = 1e-3,
    area_max_ratio: float = 0.9,
) -> bool:
    x1, y1, x2, y2 = box_xyxy
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    if w <= 1 or h <= 1:
        return False
    ar = w / h
    if ar < aspect_min or ar > aspect_max:
        return False
    area_ratio = (w * h) / float(img_w * img_h)
    if area_ratio < area_min_ratio or area_ratio > area_max_ratio:
        return False
    return True


def _xyxy_to_yolo(
    box_xyxy: tuple[float, float, float, float],
    img_w: int,
    img_h: int,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box_xyxy
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return cx, cy, w, h


def label_image(
    model: YOLOWorld,
    image_path: Path,
    spec: ClassSpec,
    conf: float,
    iou: float,
) -> list[str]:
    """Returns YOLO-format label lines for one image."""
    if spec.is_background:
        return []

    image = cv2.imread(str(image_path))
    if image is None:
        logger.warning(f"Failed to read image: {image_path}")
        return []
    img_h, img_w = image.shape[:2]

    model.set_classes(list(spec.prompts))
    results = model.predict(
        source=str(image_path),
        conf=conf,
        iou=iou,
        verbose=False,
    )

    lines: list[str] = []
    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            continue
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        for box in boxes_xyxy:
            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            if not _filter_box((x1, y1, x2, y2), img_w, img_h):
                continue
            cx, cy, bw, bh = _xyxy_to_yolo((x1, y1, x2, y2), img_w, img_h)
            cx = min(max(cx, 0.0), 1.0)
            cy = min(max(cy, 0.0), 1.0)
            bw = min(max(bw, 0.0), 1.0)
            bh = min(max(bh, 0.0), 1.0)
            lines.append(f"{spec.class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def run_split(
    model: YOLOWorld,
    src_split_dir: Path,
    out_labels_dir: Path,
    out_images_dir: Path,
    conf: float,
    iou: float,
) -> dict[str, int]:
    out_labels_dir.mkdir(parents=True, exist_ok=True)
    out_images_dir.mkdir(parents=True, exist_ok=True)

    counts = {"images": 0, "boxes": 0, "background": 0}

    for spec in CLASS_SPECS:
        class_dir = src_split_dir / spec.source_dir
        if not class_dir.is_dir():
            logger.warning(f"Missing class directory: {class_dir}")
            continue

        images = sorted(class_dir.glob("*.jpg"))
        logger.info(
            f"[{src_split_dir.name}/{spec.source_dir}] "
            f"{len(images)} images, "
            f"class_id={spec.class_id if not spec.is_background else 'bg'}"
        )

        for image_path in tqdm(images, desc=f"{spec.source_dir}", leave=False):
            stem = image_path.stem
            label_path = out_labels_dir / f"{stem}.txt"
            mirror_path = out_images_dir / image_path.name

            if not mirror_path.exists():
                try:
                    mirror_path.symlink_to(image_path.resolve())
                except (OSError, NotImplementedError):
                    import shutil
                    shutil.copy2(image_path, mirror_path)

            lines = label_image(model, image_path, spec, conf=conf, iou=iou)
            label_path.write_text("\n".join(lines), encoding="utf-8")

            counts["images"] += 1
            counts["boxes"] += len(lines)
            if spec.is_background:
                counts["background"] += 1

    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("CUMT-BelT/训练集"),
        help="Root directory of CUMT-BelT split (defaults to training split).",
    )
    parser.add_argument(
        "--out-labels",
        type=Path,
        default=Path("data/cumt_belt_yolo/labels/_train_raw"),
        help="Where to write YOLO label .txt files.",
    )
    parser.add_argument(
        "--out-images",
        type=Path,
        default=Path("data/cumt_belt_yolo/images/_train_raw"),
        help="Where to mirror images via symlink (or copy on Windows).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8l-world.pt",
        help="YOLO-World checkpoint name (downloaded by Ultralytics).",
    )
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Device override, e.g. "cuda:0", "mps", "cpu". Defaults to auto.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.src.is_dir():
        raise SystemExit(f"Source split not found: {args.src}")

    logger.info(f"Loading YOLO-World checkpoint: {args.model}")
    model = YOLOWorld(args.model)
    if args.device is not None:
        model.to(args.device)

    counts = run_split(
        model=model,
        src_split_dir=args.src,
        out_labels_dir=args.out_labels,
        out_images_dir=args.out_images,
        conf=args.conf,
        iou=args.iou,
    )

    logger.info(
        "Pseudo-labeling done. "
        f"images={counts['images']}, boxes={counts['boxes']}, "
        f"background_imgs={counts['background']}"
    )


if __name__ == "__main__":
    main()
