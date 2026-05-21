"""Generate paper-quality summary figures for the final report."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager
import numpy as np

# Pick a CJK-capable font that ships with macOS
for cand in ("PingFang SC", "Heiti SC", "STHeiti", "Hiragino Sans GB",
             "Songti SC", "Arial Unicode MS"):
    try:
        font_manager.findfont(cand, fallback_to_default=False)
        matplotlib.rcParams["font.sans-serif"] = [cand]
        break
    except Exception:
        continue
matplotlib.rcParams["axes.unicode_minus"] = False

OUT = Path("docs/figures")
OUT.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# Figure 1: matrix bar chart (final model vs baselines)
# ----------------------------------------------------------------------
matrix = [
    ("Faster R-CNN", 0.7155, 57.9),
    ("YOLOv5s",      0.7391, 166.6),
    ("YOLOv8s",      0.7454, 173.5),
    ("+ Focal",      0.7361, 172.8),
    ("+ Focal+OHEM", 0.7325, 174.1),
    ("+ FOC+OHEM+WS",0.7325, 176.6),
    ("YOLOv8m@960",  0.7439, 78.9),
    ("Improved",     0.9473, 148.0),
]
names  = [m[0] for m in matrix]
mAP50  = [m[1] for m in matrix]
fps    = [m[2] for m in matrix]
colors = ["#bbbbbb"] * (len(matrix) - 1) + ["#d62728"]

fig, ax1 = plt.subplots(figsize=(10, 5))
x = np.arange(len(names))
bars = ax1.bar(x, mAP50, color=colors, edgecolor="black", linewidth=0.7)
ax1.set_xticks(x)
ax1.set_xticklabels(names, rotation=22, ha="right")
ax1.set_ylim(0, 1.05)
ax1.set_ylabel("mAP@0.5 (test)")
ax1.axhline(0.85, color="green", linestyle="--", linewidth=1, label="客户验收线 0.85")
for bar, v in zip(bars, mAP50):
    ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}",
             ha="center", va="bottom", fontsize=9)

ax2 = ax1.twinx()
ax2.plot(x, fps, "o-", color="#1f77b4", label="FPS (4090, b=1)")
ax2.set_ylabel("FPS", color="#1f77b4")
ax2.tick_params(axis="y", colors="#1f77b4")
ax2.axhline(30, color="#1f77b4", linestyle=":", linewidth=1, label="FPS 30")

ax1.set_title("各检测算法 test mAP@0.5 与 FPS 对比")
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
plt.tight_layout()
plt.savefig(OUT / "matrix_compare.png", dpi=150)
plt.close()

# ----------------------------------------------------------------------
# Figure 2: robustness curve (round-1 vs round-2 / 改进前后)
# ----------------------------------------------------------------------
labels = ["clean", "blur k=7", "blur k=15", "noise σ=10", "noise σ=25", "blur+noise"]
old = [0.7430, 0.4348, 0.1427, 0.6213, 0.2406, 0.4236]
new = [0.9472, 0.4972, 0.1990, 0.7695, 0.4160, 0.5588]
xx = np.arange(len(labels))
w = 0.35
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.bar(xx - w / 2, old, w, label="改进前", color="#bbbbbb", edgecolor="black", linewidth=0.7)
ax.bar(xx + w / 2, new, w, label="改进后",   color="#d62728", edgecolor="black", linewidth=0.7)
for i, (a, b) in enumerate(zip(old, new)):
    ax.text(i - w / 2, a + 0.01, f"{a:.3f}", ha="center", va="bottom", fontsize=8)
    ax.text(i + w / 2, b + 0.01, f"{b:.3f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(xx)
ax.set_xticklabels(labels)
ax.set_ylabel("mAP@0.5 (test)")
ax.set_ylim(0, 1.05)
ax.set_title("鲁棒性曲线：改进算法在不同扰动下的表现")
ax.legend()
plt.tight_layout()
plt.savefig(OUT / "robustness_compare.png", dpi=150)
plt.close()

# ----------------------------------------------------------------------
# Figure 3: ablation single-row table figure (for report)
# ----------------------------------------------------------------------
ablation = [
    ("baseline",            0.7454),
    ("+ Focal Loss",        0.7361),
    ("+ Focal + OHEM",      0.7325),
    ("+ Focal + OHEM + WS", 0.7325),
]
names_a = [a[0] for a in ablation]
maps_a  = [a[1] for a in ablation]
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(names_a, maps_a, color="#1f77b4", edgecolor="black", linewidth=0.7)
for i, v in enumerate(maps_a):
    ax.text(i, v + 0.005, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("mAP@0.5 (test)")
ax.set_ylim(0.7, 0.78)
ax.set_title("YOLOv8s 三件套消融实验 (baseline 阶段)")
plt.xticks(rotation=10)
plt.tight_layout()
plt.savefig(OUT / "ablation_baseline.png", dpi=150)
plt.close()

print("Wrote:")
for f in OUT.glob("*.png"):
    print(f"  {f}")
