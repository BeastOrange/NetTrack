"""
Unified evaluation: mAP@0.5, mAP@0.5:0.95, Precision, Recall, FPS.

For YOLO variants the script loads `runs/detect/<variant>/weights/best.pt`
and runs Ultralytics' `model.val(split="test")` for accuracy metrics, then
times single-image inference (batch=1, imgsz=640) over `--fps-iters` images
to report FPS via `torch.cuda.Event`.

For Faster R-CNN it loads `runs/detect/frcnn/best.pt` (state_dict) and uses
torchmetrics' MeanAveragePrecision.

Outputs:
    runs/detect/_eval/metrics.csv     # one row per variant
    runs/detect/_eval/pr_<variant>.png  # P-R curve per YOLO variant
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import yaml
from loguru import logger

YOLO_VARIANTS = ("baseline", "focal", "focal_ohem", "full", "yolov5")
ALL_VARIANTS = YOLO_VARIANTS + ("frcnn",)


@dataclass
class Metrics:
    variant: str
    map50: float
    map5095: float
    precision: float
    recall: float
    fps: float
    n_test_images: int


def _load_data_yaml(path: Path) -> tuple[Path, int]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    root = Path(cfg.get("path", path.parent)).expanduser().resolve()
    return root, int(cfg["nc"])


def measure_fps_yolo(model, sample_paths: list[Path], device: str, iters: int) -> float:
    if not sample_paths:
        return 0.0
    if not device.startswith("cuda") or not torch.cuda.is_available():
        logger.warning("FPS measured on non-CUDA device; results are not the headline number.")

    for p in sample_paths[: min(3, len(sample_paths))]:
        model.predict(source=str(p), imgsz=640, device=device, verbose=False)

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for i in range(iters):
            p = sample_paths[i % len(sample_paths)]
            model.predict(source=str(p), imgsz=640, device=device, verbose=False)
        end.record()
        torch.cuda.synchronize()
        elapsed_s = start.elapsed_time(end) / 1000.0
    else:
        t0 = time.perf_counter()
        for i in range(iters):
            p = sample_paths[i % len(sample_paths)]
            model.predict(source=str(p), imgsz=640, device=device, verbose=False)
        elapsed_s = time.perf_counter() - t0

    return iters / elapsed_s if elapsed_s > 0 else 0.0


def eval_yolo_variant(
    variant: str,
    args: argparse.Namespace,
) -> Metrics | None:
    from ultralytics import YOLO

    weights = args.project / variant / "weights" / "best.pt"
    if not weights.exists():
        logger.warning(f"[eval] {variant}: weights not found at {weights}, skipping.")
        return None

    if variant in {"baseline", "focal", "focal_ohem", "full"}:
        from src.models.yolov8_imp import apply_improvements, config_from_variant
        apply_improvements(config_from_variant(variant))

    logger.info(f"[eval] {variant}: loading {weights}")
    model = YOLO(str(weights))
    val_results = model.val(
        data=str(args.data),
        split="test",
        imgsz=args.imgsz,
        device=args.device,
        project=str(args.project / "_eval"),
        name=variant,
        plots=True,
        verbose=False,
        exist_ok=True,
    )

    box = val_results.box
    map50 = float(box.map50)
    map5095 = float(box.map)
    precision = float(box.mp)
    recall = float(box.mr)

    root, _ = _load_data_yaml(args.data)
    test_dir = root / "images" / "test"
    sample_paths = sorted(test_dir.glob("*.jpg"))
    fps = measure_fps_yolo(model, sample_paths, args.device, args.fps_iters)

    return Metrics(
        variant=variant,
        map50=map50,
        map5095=map5095,
        precision=precision,
        recall=recall,
        fps=fps,
        n_test_images=len(sample_paths),
    )


def eval_frcnn(args: argparse.Namespace) -> Metrics | None:
    from torch.utils.data import DataLoader
    from torchmetrics.detection import MeanAveragePrecision
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    from src.training.frcnn_runner import DatasetSpec, YoloDirDataset, collate_fn

    weights = args.project / "frcnn" / "best.pt"
    if not weights.exists():
        logger.warning(f"[eval] frcnn weights not found at {weights}, skipping.")
        return None

    root, nc = _load_data_yaml(args.data)
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    test_ds = YoloDirDataset(DatasetSpec(root=root, split="test", nc=nc), imgsz=args.imgsz)
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn,
    )

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
            preds = [
                {"boxes": o["boxes"], "scores": o["scores"], "labels": o["labels"]}
                for o in outputs
            ]
            gts = [
                {"boxes": t["boxes"].to(dev), "labels": t["labels"].to(dev)}
                for t in targets
            ]
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


def write_csv(metrics: list[Metrics], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(metrics[0]).keys()))
        writer.writeheader()
        for m in metrics:
            writer.writerow(asdict(m))
    logger.info(f"Wrote metrics → {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/cumt_belt_yolo/data.yaml"))
    parser.add_argument("--project", type=Path, default=Path("runs/detect"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--fps-iters", type=int, default=200)
    parser.add_argument(
        "--variant", type=str, default=None,
        choices=ALL_VARIANTS,
        help="Evaluate a single variant. Mutually exclusive with --all-variants.",
    )
    parser.add_argument(
        "--all-variants", action="store_true",
        help="Evaluate every variant present under --project.",
    )
    parser.add_argument("--fps-only", action="store_true", help="Skip mAP, only FPS.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.all_variants and args.variant is None:
        raise SystemExit("Pass either --variant <name> or --all-variants.")

    out_dir = args.project / "_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = ALL_VARIANTS if args.all_variants else (args.variant,)
    results: list[Metrics] = []
    for variant in targets:
        if variant == "frcnn":
            m = eval_frcnn(args)
        else:
            m = eval_yolo_variant(variant, args)
        if m is not None:
            results.append(m)
            logger.info(
                f"[{variant}] mAP@0.5={m.map50:.4f} "
                f"mAP@0.5:0.95={m.map5095:.4f} "
                f"P={m.precision:.4f} R={m.recall:.4f} FPS={m.fps:.2f}"
            )

    if results:
        write_csv(results, out_dir / "metrics.csv")
    else:
        logger.warning("No variants produced metrics.")


if __name__ == "__main__":
    main()
