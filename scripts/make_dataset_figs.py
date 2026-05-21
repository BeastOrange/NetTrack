"""
Dataset exploration & analysis figures for the final report.

Produces (all under docs/figures/dataset/):

  exploration/
    grid_anchor_rod.jpg          9-image grid of 锚杆 samples
    grid_large_coal.jpg          9-image grid of 大块 samples
    grid_background.jpg          9-image grid of 正常煤流 samples
    brightness_histogram.png     pixel-mean histogram per class
    image_size_bar.png           original resolution distribution

  processing/
    pipeline_diagram.png         end-to-end data pipeline arrow diagram
    annotated_examples.jpg       3 images with GT bbox overlays
    augmentation_showcase.jpg    Mosaic / HSV / flip aug visualization
    split_bar.png                train/val/test split bar chart

  analysis/
    class_freq_per_split.png     2x2 grouped bar of class counts
    boxes_per_image_hist.png     histogram of #boxes per image
    box_size_scatter.png         per-class box size scatter (w vs h)
    box_center_heatmap.png       2D heatmap of box centres
    aspect_ratio_hist.png        per-class aspect ratio histogram
    boxes_per_image_per_class.png cumulative-density per class
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

# CJK font
for cand in ("PingFang SC", "Heiti SC", "STHeiti", "Hiragino Sans GB",
             "Songti SC", "Arial Unicode MS"):
    try:
        font_manager.findfont(cand, fallback_to_default=False)
        matplotlib.rcParams["font.sans-serif"] = [cand]
        break
    except Exception:
        continue
matplotlib.rcParams["axes.unicode_minus"] = False

random.seed(0)
np.random.seed(0)

ROOT = Path("data/cumt_belt_yolo")
ROOT_R2 = Path("data/cumt_belt_yolo_r2")
CUMT = Path("CUMT-BelT")

OUT = Path("docs/figures/dataset")
(OUT / "exploration").mkdir(parents=True, exist_ok=True)
(OUT / "processing").mkdir(parents=True, exist_ok=True)
(OUT / "analysis").mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ("anchor_rod", "large_coal")
CLASS_COLORS = ((255, 99, 71), (60, 179, 113))  # red / green BGR

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def load_yolo_label(path: Path) -> list[tuple[int, float, float, float, float]]:
    out = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        c, cx, cy, w, h = parts
        out.append((int(c), float(cx), float(cy), float(w), float(h)))
    return out


def draw_boxes_on_image(image: np.ndarray, boxes) -> np.ndarray:
    h, w = image.shape[:2]
    out = image.copy()
    for cls_id, cx, cy, bw, bh in boxes:
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, CLASS_NAMES[cls_id], (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def grid_from_paths(paths: list[Path], rows: int, cols: int,
                    title: str | None, out_path: Path,
                    box_lookup=None) -> None:
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.6))
    for ax, p in zip(np.array(axes).flat, paths):
        bgr = cv2.imread(str(p))
        if bgr is None:
            ax.axis("off")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if box_lookup is not None and p.stem in box_lookup:
            rgb = cv2.cvtColor(
                draw_boxes_on_image(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                    box_lookup[p.stem]),
                cv2.COLOR_BGR2RGB,
            )
        ax.imshow(rgb)
        ax.set_title(p.name, fontsize=7)
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


# ----------------------------------------------------------------------
# Exploration
# ----------------------------------------------------------------------

def fig_exploration_grids():
    for chinese, english in (("锚杆", "anchor_rod"),
                              ("大块", "large_coal"),
                              ("正常煤流", "background")):
        src = CUMT / "训练集" / chinese
        imgs = sorted(src.glob("*.jpg"))
        sample = random.sample(imgs, 9)
        grid_from_paths(sample, 3, 3,
                        f"训练集样本：{chinese}（{english}）",
                        OUT / "exploration" / f"grid_{english}.jpg")


def fig_brightness_histogram():
    """Pixel-mean histogram per class — gives a feel for the lighting differences."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    palette = {"锚杆": "#1f77b4", "大块": "#d62728", "正常煤流": "#7f7f7f"}
    for chinese in ("锚杆", "大块", "正常煤流"):
        src = CUMT / "训练集" / chinese
        means = []
        for p in random.sample(list(src.glob("*.jpg")), min(400, len(list(src.glob("*.jpg"))))):
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            means.append(img.mean())
        ax.hist(means, bins=40, alpha=0.55, label=chinese, color=palette[chinese])
    ax.set_xlabel("图像灰度均值")
    ax.set_ylabel("图像数")
    ax.set_title("训练集各类别图像亮度分布（灰度均值）")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT / "exploration" / "brightness_histogram.png", dpi=150)
    plt.close()


def fig_image_size_bar():
    """All images are 416x416 from CUMT-BelT; show it explicitly."""
    sizes = Counter()
    for chinese in ("锚杆", "大块", "正常煤流"):
        src = CUMT / "训练集" / chinese
        for p in random.sample(list(src.glob("*.jpg")), 100):
            img = cv2.imread(str(p))
            if img is not None:
                sizes[(img.shape[1], img.shape[0])] += 1
    labels = [f"{w}×{h}" for (w, h) in sizes.keys()]
    counts = list(sizes.values())
    fig, ax = plt.subplots(figsize=(6, 3.8))
    bars = ax.bar(labels, counts, color="#1f77b4", edgecolor="black", linewidth=0.7)
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, c + 5, str(c),
                ha="center", va="bottom", fontsize=10)
    ax.set_xlabel("原始图像分辨率")
    ax.set_ylabel("样本数（300 抽样）")
    ax.set_title("CUMT-BelT 原始图像分辨率分布")
    plt.tight_layout()
    plt.savefig(OUT / "exploration" / "image_size_bar.png", dpi=150)
    plt.close()


# ----------------------------------------------------------------------
# Processing
# ----------------------------------------------------------------------

def fig_pipeline_diagram():
    """Hand-drawn arrow diagram of the data pipeline."""
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5)
    ax.axis("off")

    nodes = [
        (0.5,  "CUMT-BelT\n原始图像\n(3 类 × 1300 / 1300 / 300)", "#cce5ff"),
        (3.0,  "标注\n(LabelImg, YOLO 格式)\n锚杆 / 大块", "#ffe0b2"),
        (5.5,  "数据集划分\n(train 3510 / val 390 / test 900)", "#c8e6c9"),
        (8.0,  "数据增强\n(Mosaic / HSV / Flip)", "#f8bbd0"),
        (10.5, "训练\n(YOLOv8m, imgsz=960)", "#d1c4e9"),
    ]
    for x, label, color in nodes:
        ax.add_patch(FancyBboxPatch((x, 2), 2.2, 1.2,
                                     boxstyle="round,pad=0.05",
                                     linewidth=1.0, edgecolor="black",
                                     facecolor=color))
        ax.text(x + 1.1, 2.6, label, ha="center", va="center", fontsize=9)

    for i in range(len(nodes) - 1):
        x1 = nodes[i][0] + 2.2
        x2 = nodes[i + 1][0]
        ax.add_patch(FancyArrowPatch((x1, 2.6), (x2, 2.6),
                                      arrowstyle="->", mutation_scale=15,
                                      color="black", linewidth=1.2))

    ax.set_title("数据集处理流水线", fontsize=12, pad=8)
    plt.tight_layout()
    plt.savefig(OUT / "processing" / "pipeline_diagram.png", dpi=150)
    plt.close()


def fig_annotated_examples():
    """3 train images with GT boxes overlaid."""
    label_dir = ROOT / "labels" / "_train_r2"
    image_dir = CUMT / "训练集"  # source images under chinese class dirs
    # Build name → image-path lookup from CUMT-BelT
    name_to_path = {}
    for chinese in ("锚杆", "大块", "正常煤流"):
        for p in (image_dir / chinese).glob("*.jpg"):
            name_to_path[p.stem] = p

    box_lookup = {}
    candidates = []
    for lbl in label_dir.glob("*.txt"):
        boxes = load_yolo_label(lbl)
        if len(boxes) >= 2:
            stem = lbl.stem
            img_path = name_to_path.get(stem)
            if img_path is not None:
                box_lookup[stem] = boxes
                candidates.append(img_path)
    if len(candidates) < 6:
        return
    samples = random.sample(candidates, 6)
    grid_from_paths(samples, 2, 3, "训练集标注示例（GT bbox 叠加显示）",
                    OUT / "processing" / "annotated_examples.jpg",
                    box_lookup=box_lookup)


def fig_augmentation_showcase():
    """Copy Ultralytics' train_batch0.jpg from the r2 run; it shows real Mosaic+aug."""
    src = Path("runs/detect/runs/r2/baseline/train_batch0.jpg")
    if not src.exists():
        return
    import shutil
    shutil.copy(src, OUT / "processing" / "augmentation_showcase.jpg")


def fig_split_bar():
    # Reproduce prepare_dataset's 9:1 train/val from the 3900 _train_r2 labels
    train_by_class, train_bpi = collect_box_stats("train")
    val_by_class, val_bpi = collect_box_stats("val")
    test_by_class, test_bpi = collect_box_stats("test")

    counts = {
        "train": (len(train_bpi), sum(1 for b in train_bpi if b == 0)),
        "val":   (len(val_bpi),   sum(1 for b in val_bpi   if b == 0)),
        "test":  (len(test_bpi),  sum(1 for b in test_bpi  if b == 0)),
    }
    labels = list(counts.keys())
    totals = [v[0] for v in counts.values()]
    empties = [v[1] for v in counts.values()]
    fg = [t - e for t, e in zip(totals, empties)]

    x = np.arange(len(labels))
    w = 0.4
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w / 2, fg, w, label="含正样本图", color="#1f77b4", edgecolor="black")
    ax.bar(x + w / 2, empties, w, label="纯背景图", color="#bbbbbb", edgecolor="black")
    for i in range(len(labels)):
        ax.text(x[i] - w / 2, fg[i] + 30, str(fg[i]), ha="center", fontsize=9)
        ax.text(x[i] + w / 2, empties[i] + 30, str(empties[i]), ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("图像数")
    ax.set_title("数据集划分：含正样本图 vs 纯背景图")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT / "processing" / "split_bar.png", dpi=150)
    plt.close()


# ----------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------

def collect_box_stats(split: str):
    # Mac only has the consolidated round-2 labels under labels/_train_r2 /
    # labels/test_r2. For analysis figures we treat _train_r2 as the train+val
    # source (3900 files) and split lazily by filename hash to mirror the
    # 9:1 train/val split used by prepare_dataset.py (seed=42).
    if split in ("train", "val"):
        label_dir = ROOT / "labels" / "_train_r2"
        all_files = sorted(label_dir.glob("*.txt"))
        # Reproduce prepare_dataset.py's deterministic split
        rng = random.Random(42)
        indices = list(range(len(all_files)))
        rng.shuffle(indices)
        n_val = max(1, int(len(all_files) * 0.1))
        val_idx = set(indices[:n_val])
        wanted = [all_files[i] for i in range(len(all_files))
                  if (i in val_idx) == (split == "val")]
    else:  # test
        label_dir = ROOT / "labels" / "test_r2"
        wanted = sorted(label_dir.glob("*.txt"))

    boxes_by_class = defaultdict(list)
    boxes_per_image = []
    for lbl in wanted:
        boxes = load_yolo_label(lbl)
        boxes_per_image.append(len(boxes))
        for cls_id, cx, cy, w, h in boxes:
            boxes_by_class[cls_id].append((cx, cy, w, h))
    return boxes_by_class, boxes_per_image


def fig_class_freq_per_split():
    splits = ("train", "val", "test")
    counts = {s: Counter() for s in splits}
    for s in splits:
        by_class, _ = collect_box_stats(s)
        for cid, items in by_class.items():
            counts[s][CLASS_NAMES[cid]] = len(items)
    x = np.arange(len(splits))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    anchor = [counts[s]["anchor_rod"] for s in splits]
    coal = [counts[s]["large_coal"] for s in splits]
    ax.bar(x - w / 2, anchor, w, label="anchor_rod", color="#1f77b4", edgecolor="black")
    ax.bar(x + w / 2, coal, w, label="large_coal", color="#d62728", edgecolor="black")
    for i in range(len(splits)):
        ax.text(x[i] - w / 2, anchor[i] + 80, str(anchor[i]), ha="center", fontsize=9)
        ax.text(x[i] + w / 2, coal[i] + 80, str(coal[i]), ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylabel("标注框数")
    ax.set_title("各划分下类别频次分布（类别不均衡可视化）")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT / "analysis" / "class_freq_per_split.png", dpi=150)
    plt.close()


def fig_boxes_per_image_hist():
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for split, color in (("train", "#1f77b4"), ("val", "#ff7f0e"), ("test", "#2ca02c")):
        _, bpi = collect_box_stats(split)
        ax.hist(bpi, bins=range(0, 15), alpha=0.6, label=split, color=color,
                edgecolor="black", linewidth=0.5)
    ax.set_xlabel("单张图标注框数")
    ax.set_ylabel("图像数")
    ax.set_title("每张图的标注框数分布")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT / "analysis" / "boxes_per_image_hist.png", dpi=150)
    plt.close()


def fig_box_size_scatter():
    """w vs h scatter, per class, on train split."""
    by_class, _ = collect_box_stats("train")
    fig, ax = plt.subplots(figsize=(7, 6))
    for cid, color, name in ((0, "#1f77b4", "anchor_rod"),
                              (1, "#d62728", "large_coal")):
        if cid not in by_class:
            continue
        arr = np.array(by_class[cid])  # cx, cy, w, h
        ax.scatter(arr[:, 2], arr[:, 3], s=4, alpha=0.35, color=color, label=name)
    ax.set_xlabel("框宽（归一化）")
    ax.set_ylabel("框高（归一化）")
    ax.set_title("训练集标注框宽高分布（按类别区分）")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT / "analysis" / "box_size_scatter.png", dpi=150)
    plt.close()


def fig_box_center_heatmap():
    """Where do objects tend to appear within the image?"""
    by_class, _ = collect_box_stats("train")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, cid, name in ((axes[0], 0, "anchor_rod"),
                           (axes[1], 1, "large_coal")):
        if cid not in by_class:
            ax.axis("off")
            continue
        arr = np.array(by_class[cid])
        hb = ax.hist2d(arr[:, 0], arr[:, 1], bins=40, cmap="hot",
                       range=[[0, 1], [0, 1]])
        ax.set_xlabel("中心 x")
        ax.set_ylabel("中心 y")
        ax.set_title(f"{name} 框中心位置分布 (n={len(arr)})")
        ax.invert_yaxis()  # image coords
        fig.colorbar(hb[3], ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(OUT / "analysis" / "box_center_heatmap.png", dpi=150)
    plt.close()


def fig_aspect_ratio_hist():
    by_class, _ = collect_box_stats("train")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for cid, color, name in ((0, "#1f77b4", "anchor_rod"),
                              (1, "#d62728", "large_coal")):
        if cid not in by_class:
            continue
        arr = np.array(by_class[cid])
        ar = arr[:, 2] / np.maximum(arr[:, 3], 1e-6)
        ar = ar[(ar > 0.05) & (ar < 20)]
        ax.hist(ar, bins=60, alpha=0.55, label=name, color=color,
                edgecolor="black", linewidth=0.4)
    ax.axvline(1.0, color="black", linestyle="--", linewidth=0.7)
    ax.set_xlabel("宽高比 (w / h)")
    ax.set_ylabel("框数")
    ax.set_title("不同类别下边界框宽高比分布")
    ax.set_xscale("log")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT / "analysis" / "aspect_ratio_hist.png", dpi=150)
    plt.close()


def fig_box_area_cdf():
    """Cumulative density of box area (normalized) per class — shows scale span."""
    by_class, _ = collect_box_stats("train")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for cid, color, name in ((0, "#1f77b4", "anchor_rod"),
                              (1, "#d62728", "large_coal")):
        if cid not in by_class:
            continue
        arr = np.array(by_class[cid])
        areas = arr[:, 2] * arr[:, 3]
        areas_sorted = np.sort(areas)
        cdf = np.arange(1, len(areas_sorted) + 1) / len(areas_sorted)
        ax.plot(areas_sorted, cdf, label=name, color=color, linewidth=1.8)
    ax.set_xlabel("框面积占图像比例")
    ax.set_ylabel("累积密度")
    ax.set_title("边界框面积累积分布（多尺度训练动机）")
    ax.set_xscale("log")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT / "analysis" / "boxes_per_image_per_class.png", dpi=150)
    plt.close()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    print("[exploration]")
    fig_exploration_grids()
    fig_brightness_histogram()
    fig_image_size_bar()
    print("[processing]")
    fig_pipeline_diagram()
    fig_annotated_examples()
    fig_augmentation_showcase()
    fig_split_bar()
    print("[analysis]")
    fig_class_freq_per_split()
    fig_boxes_per_image_hist()
    fig_box_size_scatter()
    fig_box_center_heatmap()
    fig_aspect_ratio_hist()
    fig_box_area_cdf()
    print("done.")

    for p in sorted(OUT.rglob("*")):
        if p.is_file():
            print(f"  {p}")


if __name__ == "__main__":
    main()
