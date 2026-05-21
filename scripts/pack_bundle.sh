#!/usr/bin/env bash
# pack_bundle.sh — assemble a single uploadable tarball for AutoDL.
#
# Output:
#   /tmp/coal-belt-bundle.tar.gz
#
# Contents (relative paths, extracts cleanly into ~/coal-belt/):
#   - pyproject.toml, uv.lock, .gitignore, CLAUDE.md, LICENSE, README.md
#   - scripts/   src/   docs/
#   - CUMT-BelT/                    (3900 train + 900 test images)
#   - weights/cache/                (prefetched ultralytics + torchvision .pt/.pth)
#   - weights/groundingdino/        (only if --with-gd is passed; +904MB)
#
# Usage:
#   bash scripts/pack_bundle.sh           # default, no GD weights (~430MB)
#   bash scripts/pack_bundle.sh --with-gd # include GD SwinB weights (~1.3GB)

set -euo pipefail

WITH_GD=0
for arg in "$@"; do
    case "$arg" in
        --with-gd) WITH_GD=1 ;;
        *) echo "Unknown arg: $arg"; exit 2 ;;
    esac
done

OUT="/tmp/coal-belt-bundle.tar.gz"
WORK="/tmp/coal-belt-stage"

echo "[pack] cleaning stage at $WORK"
rm -rf "$WORK"
mkdir -p "$WORK"

# Always-include files
cp -R \
    pyproject.toml uv.lock .gitignore CLAUDE.md LICENSE README.md \
    scripts src docs assets data \
    "$WORK/"

# Dataset
echo "[pack] copying CUMT-BelT (this may take a moment)..."
cp -R CUMT-BelT "$WORK/"

# Weights cache (prefetched)
mkdir -p "$WORK/weights"
if [ -d weights/cache ]; then
    cp -R weights/cache "$WORK/weights/"
else
    echo "WARNING: weights/cache not found. Run scripts/prefetch_weights.py first."
fi

# Optional GD weights
if [ "$WITH_GD" = "1" ]; then
    if [ -d weights/groundingdino ]; then
        echo "[pack] including Grounding DINO SwinB weights (+904MB)..."
        cp -R weights/groundingdino "$WORK/weights/"
    else
        echo "WARNING: --with-gd requested but weights/groundingdino missing."
    fi
fi

# Strip macOS metadata
find "$WORK" -name '._*' -delete 2>/dev/null || true
find "$WORK" -name '.DS_Store' -delete 2>/dev/null || true

# Strip pre-existing pseudo-label outputs and runs (we want a fresh start)
rm -rf "$WORK/data/cumt_belt_yolo" "$WORK/runs" "$WORK/output" 2>/dev/null || true

# Tar it
echo "[pack] writing tarball -> $OUT"
( cd /tmp && tar -czf "$OUT" -C "$WORK" . )

# Report
SIZE=$(du -h "$OUT" | cut -f1)
echo
echo "✅ bundle ready:"
echo "   path: $OUT"
echo "   size: $SIZE"
echo "   gd:   $([ "$WITH_GD" = "1" ] && echo "included" || echo "excluded")"
echo
echo "Upload to AutoDL (after you have the SSH host) with one of:"
echo "   scp $OUT autodl:~/coal-belt-bundle.tar.gz"
echo "   # then on AutoDL:"
echo "   mkdir -p ~/coal-belt && cd ~/coal-belt"
echo "   tar -xzf ~/coal-belt-bundle.tar.gz"
echo "   bash scripts/bootstrap_autodl.sh"

# Cleanup stage
rm -rf "$WORK"
