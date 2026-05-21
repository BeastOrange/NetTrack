"""
Probe Grounding DINO on coal-mine sample images.

Sweeps a small grid of (box_threshold, text_threshold) and prints the number of
boxes and max logit per image per (caption, threshold) pair. Optionally dumps
visualized boxes to /tmp/gd_probe/ for sanity check.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "legacy"))


CAPTIONS = {
    "锚杆": "anchor rod . metal bolt . steel rod .",
    "大块": "large coal block . large coal lump . big rock .",
}

THRESH_GRID = (
    (0.35, 0.25),
    (0.25, 0.20),
    (0.20, 0.15),
    (0.15, 0.10),
)


def load_gd(config: Path, weights: Path, device: str):
    from groundingdino.util.inference import load_model
    model = load_model(str(config), str(weights), device=device)
    model = model.to(device)
    model.eval()
    return model


def run_one(model, image_path: Path, caption: str, device: str):
    from groundingdino.util.inference import load_image, predict
    image_source, image = load_image(str(image_path))
    image = image.to(device)
    with torch.no_grad():
        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=caption,
            box_threshold=0.05,  # collect everything, filter later
            text_threshold=0.05,
            device=device,
        )
    return image_source, boxes.cpu().numpy(), logits.cpu().numpy(), phrases


def sweep(model, src_root: Path, n_per_class: int, device: str, dump_dir: Path | None):
    for chinese_dir, caption in CAPTIONS.items():
        class_dir = src_root / chinese_dir
        if not class_dir.is_dir():
            print(f"[miss] {class_dir}")
            continue
        images = sorted(class_dir.glob("*.jpg"))[:n_per_class]
        print(f"\n=== {chinese_dir}  caption='{caption}'  n={len(images)} ===")
        for image_path in images:
            try:
                source, boxes_cxcywh, logits, phrases = run_one(
                    model, image_path, caption, device
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  [err] {image_path.name}: {exc}")
                continue
            print(f"  {image_path.name}  total_raw={len(logits)}  max_logit={(logits.max() if len(logits) else 0):.3f}")
            for box_thr, text_thr in THRESH_GRID:
                mask = logits > box_thr
                n_kept = int(mask.sum())
                kept_max = float(logits[mask].max()) if n_kept else 0.0
                print(
                    f"    box={box_thr:.2f} text={text_thr:.2f}  n_boxes={n_kept:>3}  "
                    f"max_kept={kept_max:.3f}"
                )

            if dump_dir is not None and len(logits):
                h, w = source.shape[:2]
                kept = logits > 0.20
                if kept.any():
                    out = source.copy()
                    for box, score in zip(boxes_cxcywh[kept], logits[kept]):
                        cx, cy, bw, bh = box
                        x1 = int((cx - bw / 2) * w); y1 = int((cy - bh / 2) * h)
                        x2 = int((cx + bw / 2) * w); y2 = int((cy + bh / 2) * h)
                        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(
                            out, f"{score:.2f}", (x1, max(15, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                        )
                    out_path = dump_dir / f"{chinese_dir}_{image_path.stem}.jpg"
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(out_path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, default=Path("CUMT-BelT/训练集"))
    p.add_argument("--config", type=Path, default=Path("weights/groundingdino/GroundingDINO_SwinB_cfg.py"))
    p.add_argument("--weights", type=Path, default=Path("weights/groundingdino/groundingdino_swinb_cogcoor.pth"))
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--dump", type=Path, default=Path("/tmp/gd_probe"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"loading GD: cfg={args.config}, w={args.weights}, device={args.device}")
    model = load_gd(args.config, args.weights, args.device)
    print("loaded.")
    sweep(model, args.src, args.n, args.device, args.dump)


if __name__ == "__main__":
    main()
