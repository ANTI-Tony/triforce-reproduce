#!/usr/bin/env bash
# Run TriForce experiment on all three datasets with multiple budgets
# Context: 3840 tokens, total: 4096 (3840 + 256 gen)
# Budgets: 3840 (full cache = prefill), 2048, 1024, 512, 256
# Usage: cd /workspace/tf/triforce-reproduce/vendor/TriForce && bash ../../run_all_datasets.sh
set -euo pipefail

# Redirect HuggingFace cache to Volume Disk (more space than Container Disk)
export HF_HOME=/workspace/tf/hf_cache
export TRANSFORMERS_CACHE=/workspace/tf/hf_cache
export HF_DATASETS_CACHE=/workspace/tf/hf_cache/datasets

RESULTS_DIR="/workspace/tf/triforce-reproduce/results"
mkdir -p "$RESULTS_DIR"
CSV="$RESULTS_DIR/all_results.csv"

echo "dataset,budget,baseline_ms,acceptance,triforce_ms,speedup" > "$CSV"

DATASETS=("gs" "longbench_packed_qmsum" "lwm")
BUDGETS=(3840 2048 1024 512 256)

for DS in "${DATASETS[@]}"; do
  for BUDGET in "${BUDGETS[@]}"; do
    echo ""
    echo "========================================="
    echo "  Dataset: $DS | Budget: $BUDGET"
    echo "========================================="
    echo ""

    LOG="$RESULTS_DIR/${DS}_budget${BUDGET}.log"

    python test/on_chip.py \
        --prefill 3840 \
        --gen_len 256 \
        --budget "$BUDGET" \
        --chunk_size 8 \
        --draft_cache_budget 256 \
        --gamma 6 \
        --top_p 0.9 \
        --temp 0.6 \
        --dataset "$DS" \
        2>&1 | tee "$LOG"

    # Parse results from log
    BASELINE=$(grep -oP 'average latency: \K[0-9.]+' "$LOG" | head -1 || echo "N/A")
    ACCEPTANCE=$(grep -oP 'acceptance rate \(NOT per token\): \K[0-9.]+' "$LOG" || echo "N/A")
    TRIFORCE=$(grep -oP '\[TriForce\] average latency: \K[0-9.]+' "$LOG" || echo "N/A")
    SPEEDUP=$(grep -oP '\[E2E Speedup\]: \K[0-9.]+' "$LOG" || echo "N/A")

    echo "$DS,$BUDGET,$BASELINE,$ACCEPTANCE,$TRIFORCE,$SPEEDUP" >> "$CSV"

    echo ""
    echo "[DONE] $DS budget=$BUDGET: baseline=${BASELINE}ms, triforce=${TRIFORCE}ms, speedup=${SPEEDUP}x, acceptance=${ACCEPTANCE}"
    echo ""
  done
done

echo ""
echo "========================================="
echo "  All experiments complete!"
echo "  Results: $CSV"
echo "========================================="
echo ""
column -t -s',' "$CSV"
