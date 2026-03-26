#!/usr/bin/env bash
# Run TriForce experiment on all three datasets with multiple budgets
# Context: 124928 tokens (122k), total: 125184 (124928 + 256 gen)
# Budgets: 2048, 1024, 512
# Usage: cd /workspace/tf/triforce-reproduce/vendor/TriForce && bash ../../run_all_datasets.sh
set -euo pipefail

# Redirect HuggingFace cache to Volume Disk (more space than Container Disk)
export HF_HOME=/workspace/tf/hf_cache
export TRANSFORMERS_CACHE=/workspace/tf/hf_cache
export HF_DATASETS_CACHE=/workspace/tf/hf_cache/datasets

RESULTS_DIR="/workspace/tf/triforce-reproduce/results"
mkdir -p "$RESULTS_DIR"
CSV="$RESULTS_DIR/all_results.csv"

echo "dataset,budget,baseline_ms,acceptance,triforce_ms,speedup,peak_gpu_mb" > "$CSV"

DATASETS=("gs" "longbench_packed_qmsum" "lwm")
BUDGETS=(2048 1024 512)

for DS in "${DATASETS[@]}"; do
  for BUDGET in "${BUDGETS[@]}"; do
    echo ""
    echo "========================================="
    echo "  Dataset: $DS | Budget: $BUDGET"
    echo "========================================="
    echo ""

    LOG="$RESULTS_DIR/${DS}_budget${BUDGET}.log"
    MEM_LOG="/tmp/gpu_mem_${DS}_${BUDGET}.log"

    # Start background GPU memory sampling (every 0.5s)
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 1 > "$MEM_LOG" 2>/dev/null &
    MEM_PID=$!

    python test/on_chip.py \
        --prefill 124928 \
        --gen_len 256 \
        --budget "$BUDGET" \
        --chunk_size 8 \
        --draft_cache_budget 256 \
        --gamma 3 \
        --top_p 0.9 \
        --temp 0.6 \
        --dataset "$DS" \
        2>&1 | tee "$LOG"

    # Stop memory monitor and get peak
    kill $MEM_PID 2>/dev/null || true
    sleep 0.5
    PEAK_MEM=$(sort -n "$MEM_LOG" | tail -1 || echo "N/A")

    # Parse results from log
    BASELINE=$(grep -oP 'average latency: \K[0-9.]+' "$LOG" | head -1 || echo "N/A")
    ACCEPTANCE=$(grep -oP 'acceptance rate \(NOT per token\): \K[0-9.]+' "$LOG" || echo "N/A")
    TRIFORCE=$(grep -oP '\[TriForce\] average latency: \K[0-9.]+' "$LOG" || echo "N/A")
    SPEEDUP=$(grep -oP '\[E2E Speedup\]: \K[0-9.]+' "$LOG" || echo "N/A")

    echo "$DS,$BUDGET,$BASELINE,$ACCEPTANCE,$TRIFORCE,$SPEEDUP,$PEAK_MEM" >> "$CSV"

    echo ""
    echo "[DONE] $DS budget=$BUDGET: baseline=${BASELINE}ms, triforce=${TRIFORCE}ms, speedup=${SPEEDUP}x, acceptance=${ACCEPTANCE}, peak_gpu=${PEAK_MEM}MB"
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
