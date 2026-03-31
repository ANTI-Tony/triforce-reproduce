#!/usr/bin/env bash
# TriForce at 16K context for comparison with TinyDraft
set -euo pipefail

export HF_HOME=/workspace/tf/hf_cache
export TRANSFORMERS_CACHE=/workspace/tf/hf_cache
export HF_DATASETS_CACHE=/workspace/tf/hf_cache/datasets

RESULTS_DIR="/workspace/tf/triforce-reproduce/results/triforce_16k"
mkdir -p "$RESULTS_DIR"
CSV="$RESULTS_DIR/triforce_16k_results.csv"
rm -f "$CSV"
echo "dataset,budget,baseline_ms,acceptance,triforce_ms,speedup,peak_gpu_mb" > "$CSV"

DATASETS=("gs")

# TriForce config
BUDGET=4096
DRAFT_CACHE=256
GAMMA=3

echo "========================================="
echo "  TriForce 16K (prefill=16128, gen=256)"
echo "========================================="

for DS in "${DATASETS[@]}"; do
    echo ""
    echo "--- Dataset: $DS ---"

    LOG="$RESULTS_DIR/${DS}_16k.log"

    python test/on_chip.py \
        --prefill 16128 \
        --gen_len 256 \
        --budget "$BUDGET" \
        --chunk_size 8 \
        --draft_cache_budget "$DRAFT_CACHE" \
        --gamma "$GAMMA" \
        --top_p 0.9 \
        --temp 0.6 \
        --dataset "$DS" \
        2>&1 | tee "$LOG"

    BASELINE=$(grep -oP 'average latency: \K[0-9.]+' "$LOG" | head -1 || echo "N/A")
    ACCEPTANCE=$(grep -oP 'acceptance rate \(NOT per token\): \K[0-9.]+' "$LOG" || echo "N/A")
    TRIFORCE=$(grep -oP '\[TriForce\] average latency: \K[0-9.]+' "$LOG" || echo "N/A")
    SPEEDUP=$(grep -oP '\[E2E\] average latency: \K[0-9.]+' "$LOG" || echo "N/A")

    echo "$DS,$BUDGET,$BASELINE,$ACCEPTANCE,$TRIFORCE,$SPEEDUP" >> "$CSV"
done

echo ""
echo "========================================="
echo "  Done! Results: $CSV"
echo "========================================="
