#!/usr/bin/env bash
# TriForce at 4K (prompt=3800)
# Params proportionally scaled from 122K
set -euo pipefail

export HF_HOME=/workspace/tf/hf_cache
export TRANSFORMERS_CACHE=/workspace/tf/hf_cache

echo "TriForce 4K (draft=128, budget=128)"

cd /workspace/tf/triforce-reproduce

# Need transformers 4.37.2 for TriForce
pip install transformers==4.37.2 -q 2>/dev/null

python test/on_chip.py \
    --prefill 3800 \
    --gen_len 256 \
    --budget 128 \
    --chunk_size 1 \
    --draft_cache_budget 128 \
    --gamma 3 \
    --top_p 0.9 \
    --temp 0.6 \
    --dataset gs

echo "Done!"
