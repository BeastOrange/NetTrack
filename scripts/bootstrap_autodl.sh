#!/usr/bin/env bash
# bootstrap_autodl.sh — minimal setup on AutoDL.
#
# Strategy: AutoDL's miniconda already ships torch 2.8.0+cu128 with working
# CUDA. We install our remaining deps into the same miniconda env using
# Tsinghua PyPI mirror, then symlink prefetched weights. No venv, no NCCL
# fight, no cu121/cu118 wheel hunt.
#
# Idempotent.

set -euo pipefail

PY=/root/miniconda3/bin/python
PIP=/root/miniconda3/bin/pip
TSINGHUA_PYPI="https://pypi.tuna.tsinghua.edu.cn/simple"
HF_MIRROR="https://hf-mirror.com"

# ── 1. Mirrors ───────────────────────────────────────────────────────────────
mkdir -p ~/.pip
cat > ~/.pip/pip.conf <<EOF
[global]
index-url = ${TSINGHUA_PYPI}
trusted-host = pypi.tuna.tsinghua.edu.cn
timeout = 60
EOF

PROFILE="$HOME/.bashrc"
grep -q "HF_ENDPOINT" "$PROFILE" 2>/dev/null || cat >> "$PROFILE" <<EOF

# coal-belt-detection runtime
export HF_ENDPOINT="${HF_MIRROR}"
export HUGGINGFACE_HUB_CACHE="\$HOME/.cache/huggingface"
export TORCH_HOME="\$HOME/.cache/torch"
EOF
export HF_ENDPOINT="${HF_MIRROR}"
export TORCH_HOME="$HOME/.cache/torch"

echo "[1/4] mirrors configured"

# ── 2. Verify miniconda torch+CUDA ──────────────────────────────────────────
echo "[2/4] verifying pre-installed torch+CUDA..."
$PY - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not available in miniconda env"
print(f"  torch={torch.__version__}  cuda={torch.version.cuda}  device={torch.cuda.get_device_name(0)}")
PY

# ── 3. Install project deps into miniconda env ──────────────────────────────
echo "[3/4] installing project deps into miniconda env..."
# --no-deps for ultralytics so it doesn't try to drag torch back in;
# we install its actual transitive deps explicitly below.
$PIP install --upgrade --no-deps \
    "ultralytics>=8.3.0" \
    "torchmetrics>=1.3.0" \
    "loguru>=0.7.0" \
    "onnx>=1.15.0" \
    "onnxruntime>=1.17.0"
$PIP install \
    "opencv-python>=4.9.0" \
    "pyyaml>=6.0" \
    "matplotlib>=3.8.0" \
    "pandas>=2.1.0" \
    "pillow>=10.0.0" \
    "scipy>=1.11.0" \
    "polars>=0.20.0" \
    "ultralytics-thop>=2.0.18" \
    "lightning-utilities>=0.10.0" \
    "psutil>=5.9.0" \
    "requests>=2.31.0" \
    "seaborn>=0.13.0" \
    "py-cpuinfo>=9.0.0"

# ── 4. Wire prefetched weights so no live download is triggered ──────────────
echo "[4/4] linking prefetched weights..."

# Ultralytics looks in cwd and in SETTINGS.weights_dir (default: "weights").
# We symlink into both for redundancy.
mkdir -p weights
if [ -d weights/cache/ultralytics ]; then
    for w in weights/cache/ultralytics/*.pt; do
        ln -sf "$(realpath "$w")" "weights/$(basename "$w")"
        ln -sf "$(realpath "$w")" "$(basename "$w")"
    done
fi

# torchvision looks in $TORCH_HOME/hub/checkpoints/.
HUB_CKPT="$TORCH_HOME/hub/checkpoints"
mkdir -p "$HUB_CKPT"
if [ -d weights/cache/torchvision ]; then
    for w in weights/cache/torchvision/*.pth; do
        ln -sf "$(realpath "$w")" "$HUB_CKPT/$(basename "$w")"
    done
fi

# ── Final verification ──────────────────────────────────────────────────────
$PY - <<'PY'
import torch, ultralytics, torchvision
print(f"\nFinal env:")
print(f"  torch={torch.__version__}, cuda_ok={torch.cuda.is_available()}, device={torch.cuda.get_device_name(0)}")
print(f"  vram={torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
print(f"  torchvision={torchvision.__version__}, ultralytics={ultralytics.__version__}")
PY

echo
echo "✅ bootstrap done. Use \"$PY\" as the project's python."
echo "   Next steps:"
echo "   $PY scripts/pseudo_label.py --device cuda:0"
echo "   $PY scripts/pseudo_label_consensus.py --device cuda:0"
echo "   $PY scripts/prepare_dataset.py"
echo "   $PY scripts/ablation.py --device cuda:0 --epochs 100"
