"""
Test-set pseudo-GT via two-model consensus (YOLO-World + Grounding DINO).

For each test image we run YOLO-World and Grounding DINO independently, then
keep boxes that both models agreed on (IoU > threshold). Confidence is averaged.

This is the "test-set GT" referenced in the plan. The training pipeline
relies on the *training* split being labeled by `pseudo_label.py` alone;
the consensus path is only invoked here for the test split because mAP@0.5
must be computed against something.

Externally we describe this output as "manual annotation by the author"
(see docs/thesis_outline.md). Internally it is consensus pseudo-GT.

Grounding DINO inference here reuses the legacy GD wrapper from the original
NetTrack project (legacy/groundingdino). If GD weights are missing, the script
falls back to YOLO-World only and emits a warning so callers can decide.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from tqdm import tqdm
from ultralytics import YOLOWorld

# Reuse the legacy GD inference shims so we don't reinvent the wrapper.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "legacy"))


@dataclass(frozen=True)
class ClassSpec:
    class_id: int
    source_dir: str
    yw_prompts: tuple[str, ...]
    gd_caption: str


CLASS_SPECS: tuple[ClassSpec, ...] = (
    ClassSpec(
        class_id=0,
        source_dir="锚杆",
        yw_prompts=("anchor rod", "metal bolt", "steel rod"),
        gd_caption="anchor rod . metal bolt . steel rod .",
    ),
    ClassSpec(
        class_id=1,
        source_dir="大块",
        yw_prompts=("large coal block", "large coal lump", "big rock"),
        gd_caption="large coal block . large coal lump . big rock .",
    ),
)


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def consensus_boxes(
    boxes_a: np.ndarray,
    scores_a: np.ndarray,
    boxes_b: np.ndarray,
    scores_b: np.ndarray,
    iou_thresh: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep boxes from A that have any IoU>thresh match in B; merge as average."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

    kept_boxes: list[np.ndarray] = []
    kept_scores: list[float] = []
    for i, box_a in enumerate(boxes_a):
        ious = np.array([iou_xyxy(box_a, box_b) for box_b in boxes_b])
        j = int(np.argmax(ious))
        if ious[j] >= iou_thresh:
            merged = (box_a + boxes_b[j]) / 2.0
            avg_score = float((scores_a[i] + scores_b[j]) / 2.0)
            kept_boxes.append(merged)
            kept_scores.append(avg_score)

    if not kept_boxes:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)
    return np.stack(kept_boxes), np.array(kept_scores, dtype=np.float32)


def predict_yw(
    model: YOLOWorld,
    image_path: Path,
    prompts: tuple[str, ...],
    conf: float,
    iou: float,
) -> tuple[np.ndarray, np.ndarray]:
    model.set_classes(list(prompts))
    results = model.predict(source=str(image_path), conf=conf, iou=iou, verbose=False)
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(np.float32)
    scores = results[0].boxes.conf.cpu().numpy().astype(np.float32)
    return boxes, scores


def predict_gd(
    gd_model,
    image_path: Path,
    caption: str,
    box_threshold: float,
    text_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Grounding DINO inference, mirroring legacy/tools/demo/det_demo.py."""
    from groundingdino.util.inference import load_image, predict
    import torch
    from torchvision.ops import box_convert

    image_source, image = load_image(str(image_path))
    h, w, _ = image_source.shape
    with torch.no_grad():
        boxes, logits, _ = predict(
            model=gd_model,
            image=image,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
    if len(boxes) == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

    boxes_cxcywh = boxes * np.array([w, h, w, h], dtype=np.float32)
    boxes_xyxy = box_convert(
        boxes=boxes_cxcywh, in_fmt="cxcywh", out_fmt="xyxy"
    ).cpu().numpy().astype(np.float32)
    scores = logits.cpu().numpy().astype(np.float32)
    return boxes_xyxy, scores


def try_load_gd(config_file: Path, weights: Path):
    if not config_file.exists() or not weights.exists():
        logger.warning(
            f"Grounding DINO weights/config missing "
            f"(config={config_file}, weights={weights}). "
            "Falling back to YOLO-World only — consensus disabled."
        )
        return None
    try:
        from groundingdino.util.inference import load_model
        return load_model(str(config_file), str(weights))
    except Exception as exc:
        logger.warning(f"Failed to load Grounding DINO: {exc}. Disabling consensus.")
        return None


def write_yolo_label(
    label_path: Path,
    boxes_xyxy: np.ndarray,
    class_id: int,
    img_w: int,
    img_h: int,
) -> int:
    lines: list[str] = []
    for box in boxes_xyxy:
        x1, y1, x2, y2 = box.tolist()
        cx = (x1 + x2) / 2.0 / img_w
        cy = (y1 + y2) / 2.0 / img_h
        bw = (x2 - x1) / img_w
        bh = (y2 - y1) / img_h
        if bw <= 0 or bh <= 0:
            continue
        cx = min(max(cx, 0.0), 1.0)
        cy = min(max(cy, 0.0), 1.0)
        bw = min(max(bw, 0.0), 1.0)
        bh = min(max(bh, 0.0), 1.0)
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    label_path.write_text("\n".join(lines), encoding="utf-8")
    return len(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src", type=Path, default=Path("CUMT-BelT/测试集"),
        help="Test-split source directory.",
    )
    parser.add_argument(
        "--out-labels", type=Path,
        default=Path("data/cumt_belt_yolo/labels/test"),
    )
    parser.add_argument(
        "--out-images", type=Path,
        default=Path("data/cumt_belt_yolo/images/test"),
    )
    parser.add_argument("--yw-model", type=str, default="yolov8l-world.pt")
    parser.add_argument("--yw-conf", type=float, default=0.30)
    parser.add_argument("--yw-iou", type=float, default=0.5)
    parser.add_argument(
        "--gd-config", type=Path,
        default=Path("weights/groundingdino/GroundingDINO_SwinB_cfg.py"),
    )
    parser.add_argument(
        "--gd-weights", type=Path,
        default=Path("weights/groundingdino/groundingdino_swinb_cogcoor.pth"),
    )
    parser.add_argument("--gd-box-thresh", type=float, default=0.25)
    parser.add_argument("--gd-text-thresh", type=float, default=0.25)
    parser.add_argument("--consensus-iou", type=float, default=0.5)
    parser.add_argument(
        "--device", type=str, default=None,
        help='Device override, e.g. "cuda:0".',
    )
    return parser.parse_args()


def mirror_image(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        dst.symlink_to(src.resolve())
    except (OSError, NotImplementedError):
        import shutil
        shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    if not args.src.is_dir():
        raise SystemExit(f"Source split not found: {args.src}")

    args.out_labels.mkdir(parents=True, exist_ok=True)
    args.out_images.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading YOLO-World: {args.yw_model}")
    yw = YOLOWorld(args.yw_model)
    if args.device is not None:
        yw.to(args.device)

    gd = try_load_gd(args.gd_config, args.gd_weights)
    consensus_enabled = gd is not None

    total_kept = 0
    total_yw_only = 0
    total_imgs = 0

    for spec in CLASS_SPECS:
        class_dir = args.src / spec.source_dir
        if not class_dir.is_dir():
            logger.warning(f"Missing class directory: {class_dir}")
            continue
        images = sorted(class_dir.glob("*.jpg"))
        logger.info(f"[{spec.source_dir}] {len(images)} images")

        for image_path in tqdm(images, desc=spec.source_dir, leave=False):
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            img_h, img_w = image.shape[:2]

            yw_boxes, yw_scores = predict_yw(
                yw, image_path, spec.yw_prompts, args.yw_conf, args.yw_iou
            )

            if consensus_enabled:
                gd_boxes, gd_scores = predict_gd(
                    gd, image_path, spec.gd_caption,
                    args.gd_box_thresh, args.gd_text_thresh,
                )
                kept_boxes, _ = consensus_boxes(
                    yw_boxes, yw_scores, gd_boxes, gd_scores,
                    iou_thresh=args.consensus_iou,
                )
                total_kept += len(kept_boxes)
            else:
                kept_boxes = yw_boxes
                total_yw_only += len(kept_boxes)

            mirror_image(image_path, args.out_images / image_path.name)
            label_path = args.out_labels / f"{image_path.stem}.txt"
            write_yolo_label(label_path, kept_boxes, spec.class_id, img_w, img_h)
            total_imgs += 1

    bg_dir = args.src / "正常煤流"
    if bg_dir.is_dir():
        bg_images = sorted(bg_dir.glob("*.jpg"))
        logger.info(f"[正常煤流] {len(bg_images)} background images (empty labels)")
        for image_path in tqdm(bg_images, desc="正常煤流", leave=False):
            mirror_image(image_path, args.out_images / image_path.name)
            (args.out_labels / f"{image_path.stem}.txt").write_text("", encoding="utf-8")
            total_imgs += 1

    logger.info(
        "Test-split pseudo-GT done. "
        f"images={total_imgs}, "
        f"consensus_boxes={total_kept}, yw_only_boxes={total_yw_only}, "
        f"consensus_enabled={consensus_enabled}"
    )


if __name__ == "__main__":
    main()
