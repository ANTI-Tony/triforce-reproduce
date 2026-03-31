#!/usr/bin/env bash
# TriForce at 16K context for comparison with TinyDraft
set -euo pipefail

export HF_HOME=/workspace/tf/hf_cache
export TRANSFORMERS_CACHE=/workspace/tf/hf_cache
export HF_DATASETS_CACHE=/workspace/tf/hf_cache/datasets

RESULTS_DIR="/workspace/tf/triforce-reproduce/results/triforce_8k_16k"
mkdir -p "$RESULTS_DIR"

DATASETS=("gs")

# TriForce config
BUDGET=4096
DRAFT_CACHE=256
GAMMA=3

echo "========================================="
echo "  TriForce 8K & 16K Eval"
echo "========================================="

for PREFILL in 7936 16128; do
    CONTEXT=$((PREFILL + 256))
    echo ""
    echo "--- Prefill=${PREFILL} (context ~${CONTEXT}) ---"

    for DS in "${DATASETS[@]}"; do
        LOG="$RESULTS_DIR/${DS}_${PREFILL}.log"

        python test/on_chip.py \
            --prefill "$PREFILL" \
            --gen_len 256 \
            --budget "$BUDGET" \
            --chunk_size 8 \
            --draft_cache_budget "$DRAFT_CACHE" \
            --gamma "$GAMMA" \
            --top_p 0.9 \
            --temp 0.6 \
            --dataset "$DS" \
            2>&1 | tee "$LOG"

        echo "[DONE] $DS prefill=$PREFILL"
    done
done

echo ""
echo "========================================="
echo "  All Done!"
echo "========================================="
