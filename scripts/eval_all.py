"""
Evaluate every trained variant in the matrix on the test split.

Discovers weights at:
    runs/detect/runs/matrix/<variant>/weights/best.pt   (yolov8 variants + yolov5)
    runs/matrix/frcnn/best.pt                           (faster R-CNN)
    runs/detect/runs/matrix_v8m/baseline/weights/best.pt (tuned full+)

Outputs a single CSV at runs/detect/runs/matrix/_eval/metrics.csv with
mAP@0.5, mAP@0.5:0.95, P, R, FPS for each variant.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import yaml
from loguru import logger

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


# (variant_name, weights_path_relative_to_repo, eval_kind)
VARIANTS = (
    ("baseline",   "runs/detect/runs/matrix/baseline/weights/best.pt",    "yolo_imp"),
    ("focal",      "runs/detect/runs/matrix/focal/weights/best.pt",       "yolo_imp"),
    ("focal_ohem", "runs/detect/runs/matrix/focal_ohem/weights/best.pt",  "yolo_imp"),
    ("full",       "runs/detect/runs/matrix/full/weights/best.pt",        "yolo_imp"),
    ("yolov5",     "runs/detect/runs/matrix/yolov5/weights/best.pt",      "yolo"),
    ("v8m_imgsz960", "runs/detect/runs/matrix_v8m/baseline/weights/best.pt", "yolo_v8m"),
    ("frcnn",      "runs/matrix/frcnn/best.pt",                           "frcnn"),
)


@dataclass
class Metrics:
    variant: str
    map50: float
    map5095: float
    precision: float
    recall: float
    fps: float
    n_test_images: int


def measure_fps_yolo(model, sample_paths, device, iters, imgsz):
    if not sample_paths:
        return 0.0
    for p in sample_paths[:3]:
        model.predict(source=str(p), imgsz=imgsz, device=device, verbose=False)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for i in range(iters):
            p = sample_paths[i % len(sample_paths)]
            model.predict(source=str(p), imgsz=imgsz, device=device, verbose=False)
        end.record()
        torch.cuda.synchronize()
        elapsed_s = start.elapsed_time(end) / 1000.0
    else:
        t0 = time.perf_counter()
        for i in range(iters):
            p = sample_paths[i % len(sample_paths)]
            model.predict(source=str(p), imgsz=imgsz, device=device, verbose=False)
        elapsed_s = time.perf_counter() - t0
    return iters / elapsed_s if elapsed_s > 0 else 0.0


def eval_yolo(variant, weights, kind, args):
    from ultralytics import YOLO

    if kind == "yolo_imp":
        from src.models.yolov8_imp import apply_improvements, config_from_variant
        if variant in {"focal", "focal_ohem", "full"}:
            apply_improvements(config_from_variant(variant))

    imgsz = 960 if kind == "yolo_v8m" else args.imgsz

    logger.info(f"[eval] {variant}: loading {weights} (imgsz={imgsz})")
    model = YOLO(str(weights))
    val_results = model.val(
        data=str(args.data),
        split="test",
        imgsz=imgsz,
        device=args.device,
        project=str(args.out_dir),
        name=variant,
        plots=True,
        verbose=False,
        exist_ok=True,
    )
    box = val_results.box

    with args.data.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    test_dir = Path(cfg["path"]) / "images" / "test"
    sample_paths = sorted(test_dir.glob("*.jpg"))
    fps = measure_fps_yolo(model, sample_paths, args.device, args.fps_iters, imgsz)

    return Metrics(
        variant=variant,
        map50=float(box.map50),
        map5095=float(box.map),
        precision=float(box.mp),
        recall=float(box.mr),
        fps=fps,
        n_test_images=len(sample_paths),
    )


def eval_frcnn(weights, args):
    from torch.utils.data import DataLoader
    from torchmetrics.detection import MeanAveragePrecision
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    from src.training.frcnn_runner import DatasetSpec, YoloDirDataset, collate_fn

    with args.data.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    root = Path(cfg["path"]).expanduser().resolve()
    nc = int(cfg["nc"])
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    test_ds = YoloDirDataset(DatasetSpec(root=root, split="test", nc=nc), imgsz=args.imgsz)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)

    model = fasterrcnn_resnet50_fpn(weights=None, num_classes=91)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, nc + 1)
    model.load_state_dict(torch.load(weights, map_location=dev))
    model.to(dev).eval()

    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox").to(dev)
    n_imgs = 0
    t0 = time.perf_counter()
    with torch.inference_mode():
        for images, targets in test_loader:
            images = [im.to(dev) for im in images]
            outputs = model(images)
            preds = [{"boxes": o["boxes"], "scores": o["scores"], "labels": o["labels"]} for o in outputs]
            gts = [{"boxes": t["boxes"].to(dev), "labels": t["labels"].to(dev)} for t in targets]
            metric.update(preds, gts)
            n_imgs += len(images)
    elapsed = time.perf_counter() - t0
    out = metric.compute()
    fps = n_imgs / elapsed if elapsed > 0 else 0.0
    return Metrics(
        variant="frcnn",
        map50=float(out["map_50"].item()),
        map5095=float(out["map"].item()),
        precision=float("nan"),
        recall=float("nan"),
        fps=fps,
        n_test_images=n_imgs,
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/cumt_belt_yolo/data.yaml"))
    p.add_argument("--out-dir", type=Path, default=Path("runs/detect/runs/matrix/_eval_all"))
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--fps-iters", type=int, default=200)
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results: list[Metrics] = []
    for variant, w_rel, kind in VARIANTS:
        weights = REPO / w_rel
        if not weights.exists():
            logger.warning(f"[eval] {variant}: weights not found at {weights}, skipping.")
            continue
        try:
            if kind == "frcnn":
                m = eval_frcnn(weights, args)
            else:
                m = eval_yolo(variant, weights, kind, args)
        except Exception as exc:
            logger.error(f"[eval] {variant} failed: {exc!r}")
            continue
        results.append(m)
        logger.info(
            f"[{variant:14s}] mAP@0.5={m.map50:.4f}  "
            f"mAP@0.5:0.95={m.map5095:.4f}  "
            f"P={m.precision:.4f}  R={m.recall:.4f}  FPS={m.fps:.1f}"
        )

    if not results:
        logger.error("No variants produced metrics.")
        sys.exit(1)

    csv_path = args.out_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        w.writeheader()
        for m in results:
            w.writerow(asdict(m))
    logger.info(f"Wrote {csv_path}")

    print("\n" + "=" * 80)
    print(f"{'variant':14s} {'mAP@0.5':>9s} {'mAP@.5:.95':>11s} {'P':>7s} {'R':>7s} {'FPS':>7s}")
    print("=" * 80)
    for m in results:
        print(f"{m.variant:14s} {m.map50:>9.4f} {m.map5095:>11.4f} "
              f"{m.precision:>7.4f} {m.recall:>7.4f} {m.fps:>7.1f}")


if __name__ == "__main__":
    main()
