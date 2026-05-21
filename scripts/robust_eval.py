"""
Robustness test: measure mAP@0.5 degradation under motion blur + Gaussian noise.

Builds three perturbed copies of the test split (mild blur, strong blur, mild
noise), then runs `model.val(split=...)` against each. Outputs a CSV row per
perturbation level so the thesis can report the degradation curve.

Inputs:
    weights/best.pt (YOLOv8 .pt)
    data/cumt_belt_yolo/data.yaml

Outputs:
    runs/detect/runs/matrix/_robust/{level}/  (Ultralytics val artefacts)
    runs/detect/runs/matrix/_robust/robust.csv
"""

from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml
from loguru import logger
from tqdm import tqdm


@dataclass
class Perturbation:
    name: str
    fn: callable


def motion_blur(image: np.ndarray, ksize: int) -> np.ndarray:
    if ksize <= 1:
        return image
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    kernel[ksize // 2, :] = 1.0 / ksize
    return cv2.filter2D(image, -1, kernel)


def gauss_noise(image: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return image
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    out = image.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


PERTURBATIONS = (
    Perturbation("clean",        lambda img: img),
    Perturbation("blur_k7",      lambda img: motion_blur(img, 7)),
    Perturbation("blur_k15",     lambda img: motion_blur(img, 15)),
    Perturbation("noise_s10",    lambda img: gauss_noise(img, 10)),
    Perturbation("noise_s25",    lambda img: gauss_noise(img, 25)),
    Perturbation("blur7_noise10", lambda img: gauss_noise(motion_blur(img, 7), 10)),
)


@dataclass
class RobustRow:
    perturb: str
    map50: float
    map5095: float
    precision: float
    recall: float


def build_perturbed_split(
    src_images: Path,
    src_labels: Path,
    dst_root: Path,
    perturb: Perturbation,
) -> tuple[Path, Path]:
    np.random.seed(42)  # deterministic noise per run
    dst_images = dst_root / "images" / perturb.name
    dst_labels = dst_root / "labels" / perturb.name
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    for src in tqdm(sorted(src_images.glob("*.jpg")), desc=perturb.name, leave=False):
        dst = dst_images / src.name
        if not dst.exists():
            img = cv2.imread(str(src))
            if img is None:
                continue
            cv2.imwrite(str(dst), perturb.fn(img))
        # Symlink labels (same GT, different image)
        label_src = src_labels / f"{src.stem}.txt"
        label_dst = dst_labels / f"{src.stem}.txt"
        if label_src.exists() and not label_dst.exists():
            try:
                label_dst.symlink_to(label_src.resolve())
            except OSError:
                shutil.copy2(label_src, label_dst)
    return dst_images, dst_labels


def write_perturbed_data_yaml(perturb_root: Path, base_yaml: Path, perturb_name: str) -> Path:
    with base_yaml.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = dict(cfg)
    cfg["path"] = str(perturb_root.resolve())
    # All three keys point at the perturbed dir so Ultralytics' load-time
    # existence check passes; only `test` is actually evaluated when we call
    # model.val(split="test").
    cfg["train"] = f"images/{perturb_name}"
    cfg["val"] = f"images/{perturb_name}"
    cfg["test"] = f"images/{perturb_name}"
    out_path = perturb_root / f"data_{perturb_name}.yaml"
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--data", type=Path, default=Path("data/cumt_belt_yolo/data.yaml"))
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--out-root", type=Path, default=Path("runs/detect/runs/matrix/_robust"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    with args.data.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base_root = Path(cfg["path"]).expanduser().resolve()
    src_images = base_root / "images" / "test"
    src_labels = base_root / "labels" / "test"

    from ultralytics import YOLO
    logger.info(f"loading {args.weights}")
    model = YOLO(str(args.weights))

    rows: list[RobustRow] = []
    for perturb in PERTURBATIONS:
        logger.info(f"--- perturb={perturb.name} ---")
        build_perturbed_split(src_images, src_labels, args.out_root, perturb)
        data_yaml = write_perturbed_data_yaml(args.out_root, args.data, perturb.name)

        results = model.val(
            data=str(data_yaml),
            split="test",
            imgsz=args.imgsz,
            device=args.device,
            project=str(args.out_root),
            name=f"val_{perturb.name}",
            plots=False,
            verbose=False,
            exist_ok=True,
        )
        box = results.box
        rows.append(RobustRow(
            perturb=perturb.name,
            map50=float(box.map50),
            map5095=float(box.map),
            precision=float(box.mp),
            recall=float(box.mr),
        ))
        logger.info(
            f"[{perturb.name:14s}] mAP@0.5={float(box.map50):.4f}  "
            f"mAP@0.5:0.95={float(box.map):.4f}  P={float(box.mp):.4f}  R={float(box.mr):.4f}"
        )

    csv_path = args.out_root / "robust.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for row in rows:
            w.writerow(asdict(row))
    logger.info(f"wrote {csv_path}")

    print("\n" + "=" * 70)
    print(f"{'perturb':14s} {'mAP@0.5':>9s} {'mAP@.5:.95':>11s} {'P':>7s} {'R':>7s}")
    print("=" * 70)
    for row in rows:
        print(f"{row.perturb:14s} {row.map50:>9.4f} {row.map5095:>11.4f} "
              f"{row.precision:>7.4f} {row.recall:>7.4f}")


if __name__ == "__main__":
    main()
