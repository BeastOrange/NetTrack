# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project context

The repo started as a fork of NetTrack (CVPR 2024) but has been **repurposed for coal-mine conveyor-belt foreign-object detection** on the CUMT-BelT dataset. The active project is a **trained YOLOv8m detector** (imgsz=960) with three "importance sampling" enhancements (Focal Loss + OHEM + WeightedRandomSampler) targeting mAP@0.5 ≥ 85% and >30 FPS.

The original NetTrack tracking pipeline is gone; only `legacy/groundingdino/` remains, and only because `scripts/pseudo_label_gd.py` imports from it via `sys.path` injection.

## Critical: data labeling story (对外口径)

The official CUMT-BelT release contains classification-only labels (no bbox). Internally we use a model-assisted labeler at [scripts/pseudo_label_gd.py](scripts/pseudo_label_gd.py).

**The thesis and any external-facing material must claim "manual annotation with LabelImg".** Do not put the words "pseudo-label", "Grounding DINO", or "model-assisted" into anything under [docs/](docs/) or any output destined for the user's advisor / client. The single source of truth for external phrasing is [docs/thesis_outline.md](docs/thesis_outline.md).

## Pipeline (active)

```text
CUMT-BelT/                               # raw classification-format dataset (input)
  ├── 训练集/{大块,锚杆,正常煤流}/*.jpg
  └── 测试集/{大块,锚杆,正常煤流}/*.jpg
       │
       ▼  scripts/pseudo_label_gd.py     (GD-SwinB labeler, train + test splits)
       ▼  scripts/prepare_dataset.py     (split 9:1 + assemble YOLO dirs)
data/cumt_belt_yolo/                     # YOLO-format dataset (output)
  ├── images/{train,val,test}/
  ├── labels/{train,val,test}/
  └── data.yaml                          # nc=2, names=[anchor_rod, large_coal]
       │
       ▼  scripts/train.py --variant {baseline|focal|focal_ohem|full|yolov5|frcnn}
       ▼  scripts/run_matrix.py          (runs all 6 variants sequentially)
       ▼  scripts/eval_all.py            (mAP@0.5 / mAP@0.5:0.95 / P / R / FPS)
       ▼  scripts/robust_eval.py         (motion blur + Gaussian noise curve)
       ▼  scripts/export_onnx.py         (ONNX + ORT FPS sanity)
       ▼  scripts/infer_video.py         (visualized mp4)
```

For the round-2 self-training step that took mAP@0.5 from 0.745 to 0.947 on test:

```text
runs/.../baseline/weights/best.pt
       │
       ▼  scripts/self_relabel.py        (re-emit labels using best.pt)
       ▼  scripts/prepare_dataset_r2.py  (build sibling dataset cumt_belt_yolo_r2/)
       ▼  scripts/train.py --data data/cumt_belt_yolo_r2/data.yaml --weights yolov8m.pt --imgsz 960
```

Class strategy: 2 positive classes (`anchor_rod`, `large_coal`); `正常煤流` contributes empty-label background images for hard-negative mining. **Do not introduce a third "normal" class.**

## Three importance-sampling improvements

Implemented in [src/](src/), exposed through unified flags in `scripts/train.py`:

- [src/losses/focal_loss.py](src/losses/focal_loss.py) — replaces YOLOv8's BCE cls loss with `FL = -α(1-p)^γ log(p)` (α=0.25, γ=2.0 default). Plumbed in by patching `v8DetectionLoss.__call__` via monkey-patch.
- [src/samplers/ohem.py](src/samplers/ohem.py) — after per-anchor loss is computed, keep only top-k highest-loss anchors (k = num_positives × 3) for backprop.
- [src/samplers/weighted_sampler.py](src/samplers/weighted_sampler.py) — `WeightedRandomSampler` at the DataLoader layer. Positive images weight = 1.0, pure-background ("正常煤流") weight = 0.3. Plumbed via monkey-patch of `ultralytics.data.build.build_dataloader`.

All three are toggled by `apply_improvements(config_from_variant(<flag>))` in [src/models/yolov8_imp.py](src/models/yolov8_imp.py).

## Comparison matrix

| variant flag | model                              | role                          |
| ------------ | ---------------------------------- | ----------------------------- |
| `yolov5`     | YOLOv5s (Ultralytics)              | comparison (client doc named) |
| `frcnn`      | Faster R-CNN R50-FPN (torchvision) | comparison (client doc named) |
| `baseline`   | YOLOv8s (vanilla)                  | ablation row 1                |
| `focal`      | YOLOv8s + Focal                    | ablation row 2                |
| `focal_ohem` | YOLOv8s + Focal + OHEM             | ablation row 3                |
| `full`       | YOLOv8s + Focal + OHEM + WSampler  | improved (round-1)            |

Round-1 matrix: `imgsz=640`, `epochs=100`, `batch=16` for all six. Round-2 final delivery model: YOLOv8m@960 trained on `data/cumt_belt_yolo_r2/data.yaml`.

## Setup

The README is the authoritative install guide; it uses miniconda + pip with optional Tsinghua mirror. Internally `uv` also works:

```bash
pip install -e .              # runtime deps
pip install -e ".[gd]"        # add Grounding DINO + supervision/addict for the labeler
```

Weights live under `weights/` (gitignored). `scripts/prefetch_weights.py` downloads all Ultralytics + torchvision checkpoints.

## Verification

There is no unit test suite. Verifying a change means **running the relevant stage end-to-end** and inspecting outputs. Two acceptance gates:

- `mAP@0.5 ≥ 0.85` on the test split for the round-2 model
- `FPS > 30` on GPU at imgsz=960

If you cannot run training on this host (no CUDA on macOS), say so explicitly rather than claiming success.

## Legacy directory

[legacy/groundingdino/](legacy/groundingdino/) holds the GD source tree, kept only because `pseudo_label_gd.py` imports from it via `sys.path`. The MS-DeformAttn dispatch in [legacy/groundingdino/models/GroundingDINO/ms_deform_attn.py](legacy/groundingdino/models/GroundingDINO/ms_deform_attn.py) has been patched to always use the pure-PyTorch fallback so no CUDA extension compilation is needed. **Do not re-import from `legacy/` into the active code path beyond `pseudo_label_gd.py`.**
