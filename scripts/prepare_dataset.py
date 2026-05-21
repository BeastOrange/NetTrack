"""
Assemble the YOLO-format dataset directory after pseudo-labeling.

Inputs (produced by pseudo_label.py):
    data/cumt_belt_yolo/images/_train_raw/*.jpg
    data/cumt_belt_yolo/labels/_train_raw/*.txt

Inputs (produced by pseudo_label_consensus.py):
    data/cumt_belt_yolo/images/test/*.jpg
    data/cumt_belt_yolo/labels/test/*.txt

Outputs:
    data/cumt_belt_yolo/images/train/*.jpg
    data/cumt_belt_yolo/images/val/*.jpg
    data/cumt_belt_yolo/labels/train/*.txt
    data/cumt_belt_yolo/labels/val/*.txt
    data/cumt_belt_yolo/data.yaml

Behavior:
- Splits the pseudo-labeled training pool 9:1 into train/val (deterministic
  via `random.Random(seed)`, default seed=42).
- Mirrors images via symlink (or copy on Windows).
- Skips files that already exist at the destination.
- Validates that every image has a matching label file (empty file = bg).
- Writes data.yaml with absolute path to the dataset root so Ultralytics
  picks it up regardless of cwd.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import yaml
from loguru import logger
from tqdm import tqdm

CLASS_NAMES = ("anchor_rod", "large_coal")


def mirror(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        dst.symlink_to(src.resolve())
    except (OSError, NotImplementedError):
        import shutil
        shutil.copy2(src, dst)


def split_train_val(
    raw_images: Path,
    raw_labels: Path,
    out_root: Path,
    val_ratio: float,
    seed: int,
) -> tuple[int, int]:
    image_paths = sorted(raw_images.glob("*.jpg"))
    if not image_paths:
        raise SystemExit(f"No images in {raw_images}. Run pseudo_label.py first.")

    rng = random.Random(seed)
    indices = list(range(len(image_paths)))
    rng.shuffle(indices)
    n_val = max(1, int(len(image_paths) * val_ratio))
    val_idx = set(indices[:n_val])

    train_imgs = out_root / "images" / "train"
    val_imgs = out_root / "images" / "val"
    train_lbls = out_root / "labels" / "train"
    val_lbls = out_root / "labels" / "val"
    for d in (train_imgs, val_imgs, train_lbls, val_lbls):
        d.mkdir(parents=True, exist_ok=True)

    n_train = 0
    n_valid = 0
    for i, image_path in enumerate(tqdm(image_paths, desc="split train/val")):
        label_path = raw_labels / f"{image_path.stem}.txt"
        if not label_path.exists():
            logger.warning(f"Missing label for {image_path.name}, skipping")
            continue
        if i in val_idx:
            mirror(image_path, val_imgs / image_path.name)
            mirror(label_path, val_lbls / label_path.name)
            n_valid += 1
        else:
            mirror(image_path, train_imgs / image_path.name)
            mirror(label_path, train_lbls / label_path.name)
            n_train += 1
    return n_train, n_valid


def validate_pairs(images_dir: Path, labels_dir: Path) -> int:
    missing = 0
    for image_path in images_dir.glob("*.jpg"):
        if not (labels_dir / f"{image_path.stem}.txt").exists():
            logger.warning(f"Image without label: {image_path}")
            missing += 1
    return missing


def write_data_yaml(out_root: Path) -> Path:
    out_root_abs = out_root.resolve()
    data_yaml = {
        "path": str(out_root_abs),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(CLASS_NAMES),
        "names": list(CLASS_NAMES),
    }
    yaml_path = out_root / "data.yaml"
    yaml_path.write_text(
        yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return yaml_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/cumt_belt_yolo"),
        help="Output dataset root.",
    )
    parser.add_argument(
        "--raw-images",
        type=Path,
        default=Path("data/cumt_belt_yolo/images/_train_raw"),
    )
    parser.add_argument(
        "--raw-labels",
        type=Path,
        default=Path("data/cumt_belt_yolo/labels/_train_raw"),
    )
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    n_train, n_val = split_train_val(
        raw_images=args.raw_images,
        raw_labels=args.raw_labels,
        out_root=args.root,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    test_imgs = args.root / "images" / "test"
    test_lbls = args.root / "labels" / "test"
    n_test = len(list(test_imgs.glob("*.jpg"))) if test_imgs.is_dir() else 0
    if n_test == 0:
        logger.warning(
            "No test images found at "
            f"{test_imgs}. Run pseudo_label_consensus.py before training."
        )

    missing_train = validate_pairs(args.root / "images" / "train", args.root / "labels" / "train")
    missing_val = validate_pairs(args.root / "images" / "val", args.root / "labels" / "val")
    if test_imgs.is_dir():
        missing_test = validate_pairs(test_imgs, test_lbls)
    else:
        missing_test = 0

    yaml_path = write_data_yaml(args.root)

    logger.info(
        f"Done. train={n_train}, val={n_val}, test={n_test}, "
        f"missing_labels(train/val/test)={missing_train}/{missing_val}/{missing_test}"
    )
    logger.info(f"data.yaml -> {yaml_path}")


if __name__ == "__main__":
    main()
