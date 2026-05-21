"""
Build a round-2 dataset variant by reusing the round-1 train/val image split
but swapping in self-relabeled ground-truth from labels/_train_r2 + labels/test_r2.

Round-1 layout (existing):
    data/cumt_belt_yolo/
        images/{train,val,test}/*.jpg     (already split 9:1)
        labels/{train,val,test}/*.txt     (GD-SwinB pseudo-GT)
        labels/_train_r2/*.txt            (round-1 model preds on train+val pool)
        labels/test_r2/*.txt              (round-1 model preds on test)

Round-2 layout (new sibling, so Ultralytics' images↔labels path swap works):
    data/cumt_belt_yolo_r2/
        images/{train,val,test}/*.jpg     (symlinks to round-1 images)
        labels/{train,val,test}/*.txt     (copied from round-1's *_r2 dirs)
        data.yaml

The round-1 split is preserved so any train→test eval comparison stays fair.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from loguru import logger
from tqdm import tqdm

ROOT = Path("data/cumt_belt_yolo")
ROOT_R2 = Path("data/cumt_belt_yolo_r2")
CLASS_NAMES = ("anchor_rod", "large_coal")


def link_dir(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if not f.name.endswith(".jpg"):
            continue
        target = dst / f.name
        if target.exists():
            continue
        try:
            target.symlink_to(f.resolve())
        except (OSError, NotImplementedError):
            import shutil
            shutil.copy2(f, target)


def copy_labels(src: Path, dst: Path, image_dir: Path) -> int:
    """Copy a label file for each .jpg in image_dir from src/<stem>.txt → dst/<stem>.txt.
    If src lacks the stem, write an empty file (background)."""
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for img in image_dir.glob("*.jpg"):
        s = src / f"{img.stem}.txt"
        d = dst / f"{img.stem}.txt"
        if d.exists():
            continue
        if s.exists():
            d.write_text(s.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            d.write_text("", encoding="utf-8")
        n += 1
    return n


def main() -> None:
    img_train = ROOT / "images" / "train"
    img_val = ROOT / "images" / "val"
    img_test = ROOT / "images" / "test"

    # New layout: sibling dataset root with vanilla images/+labels/ so
    # Ultralytics' image↔label path replacement (literal substring "images"→"labels")
    # works without any patching.
    img_r2_train = ROOT_R2 / "images" / "train"
    img_r2_val = ROOT_R2 / "images" / "val"
    img_r2_test = ROOT_R2 / "images" / "test"
    lbl_r2_train = ROOT_R2 / "labels" / "train"
    lbl_r2_val = ROOT_R2 / "labels" / "val"
    lbl_r2_test = ROOT_R2 / "labels" / "test"

    logger.info("symlinking images...")
    link_dir(img_train, img_r2_train)
    link_dir(img_val, img_r2_val)
    link_dir(img_test, img_r2_test)

    src_train_r2 = ROOT / "labels" / "_train_r2"
    src_test_r2 = ROOT / "labels" / "test_r2"

    logger.info("copying r2 labels for train/val/test...")
    copy_labels(src_train_r2, lbl_r2_train, img_r2_train)
    copy_labels(src_train_r2, lbl_r2_val, img_r2_val)
    copy_labels(src_test_r2, lbl_r2_test, img_r2_test)

    # Stats
    def count_boxes(d: Path) -> tuple[int, int, int]:
        n_files = 0; n_empty = 0; n_boxes = 0
        for f in d.glob("*.txt"):
            n_files += 1
            body = f.read_text().strip()
            if body:
                n_boxes += len(body.splitlines())
            else:
                n_empty += 1
        return n_files, n_empty, n_boxes

    for name, d in (("train", lbl_r2_train), ("val", lbl_r2_val), ("test", lbl_r2_test)):
        n_files, n_empty, n_boxes = count_boxes(d)
        logger.info(f"  {name}: {n_files} files, {n_empty} empty, {n_boxes} boxes")

    yaml_path = ROOT_R2 / "data.yaml"
    cfg = {
        "path": str(ROOT_R2.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(CLASS_NAMES),
        "names": list(CLASS_NAMES),
    }
    yaml_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    logger.info(f"wrote {yaml_path}")


if __name__ == "__main__":
    main()
