"""
Video-stream inference demo and dust/blur robustness probe.

Two modes:

    --video <file>            Plain inference on a video, output annotated mp4.
    --robustness              Evaluate mAP@0.5 on the test set after applying
                              motion-blur + Gaussian-noise augmentation, and
                              record the mAP delta vs. clean inputs.

Both modes accept any YOLO checkpoint produced by `scripts/train.py`. The
robustness mode reuses the same data.yaml as `scripts/eval.py` so the numbers
stay consistent with the rest of the pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml
from loguru import logger
from tqdm import tqdm


def annotate_frame(frame, results, names) -> np.ndarray:
    if results.boxes is None or len(results.boxes) == 0:
        return frame
    xyxy = results.boxes.xyxy.cpu().numpy().astype(int)
    confs = results.boxes.conf.cpu().numpy()
    cls_ids = results.boxes.cls.cpu().numpy().astype(int)
    palette = [(0, 200, 0), (0, 128, 255), (255, 64, 64), (200, 0, 200)]
    for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, cls_ids):
        color = palette[cls_id % len(palette)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{names.get(cls_id, str(cls_id))} {conf:.2f}"
        cv2.putText(
            frame, label, (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
        )
    return frame


def run_video(args: argparse.Namespace) -> None:
    from ultralytics import YOLO

    model = YOLO(args.weights)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.output), fourcc, fps, (width, height))

    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))

    pbar = tqdm(total=total, desc="infer video")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            results = model.predict(
                source=frame, imgsz=args.imgsz, conf=args.conf,
                device=args.device, verbose=False,
            )[0]
            writer.write(annotate_frame(frame, results, names))
            pbar.update(1)
    finally:
        pbar.close()
        cap.release()
        writer.release()
    logger.info(f"Wrote annotated video → {args.output}")


def perturb(image: np.ndarray, blur_kernel: int, noise_sigma: float) -> np.ndarray:
    """Apply motion-blur + additive Gaussian noise to a BGR uint8 frame."""
    if blur_kernel >= 3 and blur_kernel % 2 == 1:
        kernel = np.zeros((blur_kernel, blur_kernel), dtype=np.float32)
        kernel[blur_kernel // 2, :] = 1.0 / blur_kernel
        image = cv2.filter2D(image, -1, kernel)
    if noise_sigma > 0:
        noise = np.random.normal(0.0, noise_sigma, image.shape).astype(np.float32)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return image


def _load_data_yaml(path: Path) -> Path:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return Path(cfg.get("path", path.parent)).expanduser().resolve()


def run_robustness(args: argparse.Namespace) -> None:
    """Probe mAP@0.5 under blur + noise. Builds a corrupted mirror of the test split."""
    from ultralytics import YOLO

    model = YOLO(args.weights)
    root = _load_data_yaml(args.data)
    test_imgs = root / "images" / "test"
    test_lbls = root / "labels" / "test"
    if not test_imgs.is_dir():
        raise SystemExit(f"Missing test images dir: {test_imgs}")

    perturbed_root = args.work / "perturbed"
    perturbed_imgs = perturbed_root / "images" / "test"
    perturbed_lbls = perturbed_root / "labels" / "test"
    perturbed_imgs.mkdir(parents=True, exist_ok=True)
    perturbed_lbls.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)

    for image_path in tqdm(sorted(test_imgs.glob("*.jpg")), desc="perturbing"):
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        out = perturb(frame, blur_kernel=args.blur_kernel, noise_sigma=args.noise_sigma)
        cv2.imwrite(str(perturbed_imgs / image_path.name), out)
        # Mirror labels so val() can find them.
        src_lbl = test_lbls / f"{image_path.stem}.txt"
        dst_lbl = perturbed_lbls / f"{image_path.stem}.txt"
        if src_lbl.exists():
            dst_lbl.write_text(src_lbl.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            dst_lbl.write_text("", encoding="utf-8")

    perturbed_yaml = perturbed_root / "data.yaml"
    perturbed_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(perturbed_root.resolve()),
                "train": "images/test",
                "val": "images/test",
                "test": "images/test",
                "nc": 2,
                "names": ["anchor_rod", "large_coal"],
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    clean = model.val(
        data=str(args.data), split="test",
        imgsz=args.imgsz, device=args.device, verbose=False, plots=False,
        project=str(args.work / "_eval"), name="clean", exist_ok=True,
    )
    perturbed = model.val(
        data=str(perturbed_yaml), split="test",
        imgsz=args.imgsz, device=args.device, verbose=False, plots=False,
        project=str(args.work / "_eval"), name="perturbed", exist_ok=True,
    )

    logger.info(
        "Robustness:\n"
        f"  clean      mAP@0.5={clean.box.map50:.4f}, mAP@0.5:0.95={clean.box.map:.4f}\n"
        f"  perturbed  mAP@0.5={perturbed.box.map50:.4f}, mAP@0.5:0.95={perturbed.box.map:.4f}\n"
        f"  ΔmAP@0.5  = {perturbed.box.map50 - clean.box.map50:+.4f}\n"
        f"  ΔmAP@0.5:0.95 = {perturbed.box.map - clean.box.map:+.4f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=str, required=True, help="Path to a .pt checkpoint.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", type=str, default="cuda:0")

    sub = parser.add_subparsers(dest="mode", required=True)

    p_video = sub.add_parser("video", help="Run inference on a video file.")
    p_video.add_argument("--video", type=Path, required=True)
    p_video.add_argument("--output", type=Path, default=Path("runs/demo/output.mp4"))

    p_robust = sub.add_parser("robustness", help="Evaluate blur+noise robustness.")
    p_robust.add_argument("--data", type=Path, default=Path("data/cumt_belt_yolo/data.yaml"))
    p_robust.add_argument("--work", type=Path, default=Path("runs/robustness"))
    p_robust.add_argument("--blur-kernel", type=int, default=9)
    p_robust.add_argument("--noise-sigma", type=float, default=15.0)
    p_robust.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "video":
        run_video(args)
    elif args.mode == "robustness":
        run_robustness(args)
    else:
        raise SystemExit(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
