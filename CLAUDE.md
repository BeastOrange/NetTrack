# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

This repo started life as a fork of NetTrack (CVPR 2024) — a training-free MOT pipeline (Grounding DINO + CoTracker + ByteTrack) — but has since been **repurposed for coal-mine conveyor-belt foreign-object detection** (煤矿输送带异物检测) on the CUMT-BelT dataset.

The tracking pipeline is now considered **legacy** (see [legacy/](legacy/)) and is not part of the active code path. The active project is a **trained YOLOv8 detector** with three "importance sampling" enhancements (Focal Loss + OHEM + WeightedRandomSampler) targeting mAP@0.5 ≥ 85% and >30 FPS on a GPU.

The full implementation plan, including risks and acceptance criteria, lives at `~/.claude/plans/yolo-faster-rippling-wreath.md`. Read it before making structural changes.

## Critical: data labeling story (对外口径)

**The official CUMT-BelT release contains classification-only labels (no bbox).** This project uses **YOLO-World as a zero-shot pseudo-labeler** for training data and **YOLO-World + Grounding DINO consensus** as pseudo-GT for the test set. This pipeline lives in [scripts/pseudo_label.py](scripts/pseudo_label.py) and [scripts/pseudo_label_consensus.py](scripts/pseudo_label_consensus.py).

**The thesis and any external-facing material must claim "manual annotation with LabelImg"**. Do not put the words "pseudo-label", "YOLO-World", "Grounding DINO", or "consensus" into anything under [docs/](docs/) or any output destined for the user's advisor / client. The single source of truth for external phrasing is [docs/thesis_outline.md](docs/thesis_outline.md).

## Compute split

- The user's primary machine is **macOS (this repo lives here)**. Training does not run here.
- Training, ONNX export, and FPS benchmarking happen on a **rented mainland-China GPU box (AutoDL or similar)** reachable via SSH. The box has no firewall workaround, so all out-of-network downloads must be pre-fetched on macOS (which has a VPN) and bundled along.
- Local-only operations: code editing, dataset prep on small samples, plotting, documentation, weight prefetch.

## AutoDL deployment (one-shot)

The deployment story is built around a single tarball that contains everything the GPU box needs offline-ish:

1. **On macOS** (with VPN, once) — prefetch Ultralytics + torchvision weights:

   ```bash
   uv run scripts/prefetch_weights.py
   ```

   Outputs to [weights/cache/](weights/cache/). These are GitHub + pytorch.org downloads that AutoDL cannot reach directly.

2. **On macOS** — assemble the bundle:

   ```bash
   bash scripts/pack_bundle.sh           # ~753MB, no GD weights
   bash scripts/pack_bundle.sh --with-gd # ~1.3GB, includes GD SwinB for consensus
   ```

   Produces `/tmp/coal-belt-bundle.tar.gz`.

3. **On AutoDL** — upload + bootstrap:

   ```bash
   # from mac:
   scp /tmp/coal-belt-bundle.tar.gz autodl:~/coal-belt-bundle.tar.gz
   # on AutoDL:
   mkdir -p ~/coal-belt && cd ~/coal-belt
   tar -xzf ~/coal-belt-bundle.tar.gz
   bash scripts/bootstrap_autodl.sh
   ```

   `bootstrap_autodl.sh` is idempotent. It configures Tsinghua PyPI + SJTU PyTorch + hf-mirror, runs `uv sync`, replaces CPU torch with `torch==2.5.1+cu121` from SJTU, and symlinks the prefetched weights into `weights/` and `~/.cache/torch/hub/checkpoints/` so Ultralytics + torchvision never trigger a download.

Mirror table (used by `bootstrap_autodl.sh`, persisted in `~/.bashrc` / `~/.pip/pip.conf` / `~/.config/uv/uv.toml`):

| Source                          | Mirror                                            |
|---------------------------------|---------------------------------------------------|
| PyPI                            | `https://pypi.tuna.tsinghua.edu.cn/simple`        |
| PyTorch CUDA 12.1 wheels        | `https://mirror.sjtu.edu.cn/pytorch-wheels/cu121` |
| Hugging Face                    | `https://hf-mirror.com`                           |
| Ultralytics + torchvision .pt   | bundled in `weights/cache/` (no live download)    |

## Pipeline (active)

```text
CUMT-BelT/                               # raw classification-format dataset (input)
  ├── 训练集/{大块,锚杆,正常煤流}/*.jpg
  └── 测试集/{大块,锚杆,正常煤流}/*.jpg
       │
       ▼  scripts/pseudo_label.py           (YOLO-World, train split)
       ▼  scripts/pseudo_label_consensus.py (YOLO-World ∩ GD, test split)
       ▼  scripts/prepare_dataset.py        (split + assemble YOLO dirs)
data/cumt_belt_yolo/                     # YOLO-format dataset (output)
  ├── images/{train,val,test}/
  ├── labels/{train,val,test}/
  └── data.yaml                          # nc=2, names=[anchor_rod, large_coal]
       │
       ▼  scripts/train.py --variant {baseline|focal|focal_ohem|full|yolov5|frcnn}
       ▼  scripts/ablation.py             (runs the four variants)
       ▼  scripts/eval.py                 (mAP@0.5 / mAP@0.5:0.95 / P / R / FPS)
runs/                                    # Ultralytics-style training output
       │
       ▼  scripts/infer_video.py          (visualized mp4)
       ▼  yolo export model=best.pt format=onnx
```

Class strategy: 2 positive classes (`anchor_rod`, `large_coal`); the `正常煤流` directory contributes empty-label background images for hard-negative mining. **Do not introduce a third "normal" class.**

## Three importance-sampling improvements

Implemented in [src/](src/), exposed through unified flags in `scripts/train.py`:

- [src/losses/focal_loss.py](src/losses/focal_loss.py) — replaces YOLOv8's BCE cls loss with `FL = -α(1-p)^γ log(p)` (α=0.25, γ=2.0 default). Plumbed in by subclassing `v8DetectionLoss` and registering on the model's criterion.
- [src/samplers/ohem.py](src/samplers/ohem.py) — after per-anchor loss is computed, keep only top-k highest-loss anchors (k = num_positives × 3) for backprop. Hooks into the Focal Loss subclass.
- [src/samplers/weighted_sampler.py](src/samplers/weighted_sampler.py) — `WeightedRandomSampler` at the DataLoader layer. Positive images weight = 1.0, pure-background ("正常煤流") images weight = 0.3. Plumbed via monkey-patch of Ultralytics' default sampler.

The "重要性采样" framing in the thesis collapses these three into one coherent story; the three knobs map 1:1 to ablation rows.

## Comparison matrix

| variant flag | model                              | role                          |
|--------------|------------------------------------|-------------------------------|
| `yolov5`     | YOLOv5s (Ultralytics)              | comparison (client doc named) |
| `frcnn`      | Faster R-CNN R50-FPN (torchvision) | comparison (client doc named) |
| `baseline`   | YOLOv8s (vanilla)                  | ablation row 1                |
| `focal`      | YOLOv8s + Focal                    | ablation row 2                |
| `focal_ohem` | YOLOv8s + Focal + OHEM             | ablation row 3                |
| `full`       | YOLOv8s + Focal + OHEM + WSampler  | improved (delivery)           |

All six runs use the same `data.yaml`, `imgsz=640`, `epochs=100`, `batch=16` for fair comparison.

## Setup

Use [`uv`](https://github.com/astral-sh/uv) — `requirements.txt` was retired in favor of `pyproject.toml`.

```bash
uv sync                       # install runtime deps
uv sync --extra gd            # add Grounding DINO deps for the consensus pseudo-labeler
```

Weights live under `weights/` (already gitignored at the project root). The pseudo-labeling scripts download YOLO-World checkpoints lazily on first run via Ultralytics; Grounding DINO weights ([weights/groundingdino/groundingdino_swinb_cogcoor.pth](weights/groundingdino/)) are still needed for the consensus path.

## Verification

There is no unit test suite. Verifying a change means **running the relevant stage end-to-end** and inspecting outputs. The plan file's "验证方法" section enumerates the nine stages; the two acceptance gates are:

- `mAP@0.5 ≥ 85%` on the test split for the `full` variant
- `FPS > 30` for the `full` variant on GPU at imgsz=640

If you cannot run training locally (the macOS host has no CUDA), state that explicitly and surface the SSH command for the remote Windows box rather than claiming success.

## Legacy pipeline

[legacy/](legacy/) holds the original NetTrack tracker, the old `det_demo.py` / `track_demo.py` / `demo_seq.sh` scripts, the `groundingdino/` source tree from the GD installation, and the original `config/NetTrack_SwinB_cfg.py`. **Do not re-import from `legacy/` into the active code path.** Treat it as reference material only — useful for understanding what was tried before, useful for reusing GD inference code in the consensus pseudo-labeler, but otherwise frozen.
