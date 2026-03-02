#!/usr/bin/env bash
# Patch TriForce sampling.py to handle nan/inf in fp16
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="$SCRIPT_DIR/../vendor/TriForce/utils/sampling.py"

if [ ! -f "$TARGET" ]; then
    echo "[ERROR] sampling.py not found at $TARGET"
    exit 1
fi

# Check if already patched
if grep -q "nan_to_num" "$TARGET"; then
    echo "[INFO] sampling.py already patched"
    exit 0
fi

# Patch: add nan/inf guard before torch.multinomial
sed -i 's|def sample(probs : torch.Tensor, num_samples=1):|def sample(probs : torch.Tensor, num_samples=1):\n    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)\n    probs = torch.clamp(probs, min=0.0)\n    if probs.sum() == 0:\n        probs = torch.ones_like(probs) / probs.shape[-1]|' "$TARGET"

echo "[INFO] Patched sampling.py (nan/inf guard)"
