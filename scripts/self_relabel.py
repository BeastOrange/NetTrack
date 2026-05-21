"""
Self-training round-2 relabeler.

Uses an already-trained YOLO checkpoint (round-1 baseline.pt) to re-emit
labels for both the train and test image pools. Round-1 weights were trained
on GD-SwinB pseudo-labels; round-2 GT comes from round-1's predictions, which
empirically denoise the box positions and reject GD's spurious detections.

Pipeline (per split):
    images/<split>/*.jpg
        → model.predict at conf=--conf
        → write labels/<split_out>/<stem>.txt in YOLO format
        → empty label file if no detection (image kept as background)

Idempotent. Outputs go to a NEW labels dir, never overwrites the GD pseudo-GT
under labels/_train_raw or labels/test, so we can A/B them.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from loguru import logger
from tqdm import tqdm
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--src-images", type=Path, required=True,
                   help="Directory of .jpg images to relabel.")
    p.add_argument("--dst-labels", type=Path, required=True,
                   help="Output dir for new YOLO label .txt files.")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.20)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--device", type=str, default="cuda:0")
    return p.parse_args()


def main():
    args = parse_args()
    if args.dst_labels.exists():
        shutil.rmtree(args.dst_labels)
    args.dst_labels.mkdir(parents=True, exist_ok=True)

    logger.info(f"loading {args.weights}")
    model = YOLO(str(args.weights))

    images = sorted(args.src_images.glob("*.jpg"))
    logger.info(f"relabeling {len(images)} images at conf={args.conf} iou={args.iou} imgsz={args.imgsz}")

    n_with_box = 0
    n_total_boxes = 0
    for img_path in tqdm(images):
        results = model.predict(
            source=str(img_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )
        lines = []
        if results and results[0].boxes is not None and len(results[0].boxes):
            for box in results[0].boxes:
                cls_id = int(box.cls.item())
                cx, cy, w, h = box.xywhn[0].cpu().numpy().tolist()
                lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        if lines:
            n_with_box += 1
            n_total_boxes += len(lines)
        (args.dst_labels / f"{img_path.stem}.txt").write_text(
            "\n".join(lines), encoding="utf-8"
        )

    logger.info(
        f"done. {len(images)} imgs → {n_with_box} labeled ({n_total_boxes} boxes), "
        f"{len(images) - n_with_box} empty (treated as background)"
    )


if __name__ == "__main__":
    main()
