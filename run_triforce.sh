#!/usr/bin/env bash
# TriForce - Long-context only (122K + 256)
# 3 datasets × 1 config = 3 runs (+ AR baseline per dataset)
set -euo pipefail

export HF_HOME=/workspace/tf/hf_cache
export TRANSFORMERS_CACHE=/workspace/tf/hf_cache
export HF_DATASETS_CACHE=/workspace/tf/hf_cache/datasets

RESULTS_DIR="/workspace/tf/triforce-reproduce/results/triforce"
mkdir -p "$RESULTS_DIR"
CSV="$RESULTS_DIR/triforce_results.csv"
rm -f "$CSV"
echo "dataset,budget,baseline_ms,acceptance,triforce_ms,speedup,peak_gpu_mb" > "$CSV"

DATASETS=("gs" "longbench_packed_qmsum" "lwm")

# TriForce config: retrieval cache budget=4096, StreamingLLM draft_cache=256
BUDGET=4096
DRAFT_CACHE=256
GAMMA=3

echo "========================================="
echo "  TriForce Experiments (Long-context)"
echo "  prefill=124928, gen=256, budget=$BUDGET"
echo "========================================="

for DS in "${DATASETS[@]}"; do
    echo ""
    echo "========================================="
    echo "  Dataset: $DS"
    echo "========================================="

    LOG="$RESULTS_DIR/${DS}.log"
    MEM_LOG="/tmp/gpu_mem_triforce_${DS}.log"

    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 1 > "$MEM_LOG" 2>/dev/null &
    MEM_PID=$!

    python test/on_chip.py \
        --prefill 124928 \
        --gen_len 256 \
        --budget "$BUDGET" \
        --chunk_size 8 \
        --draft_cache_budget "$DRAFT_CACHE" \
        --gamma "$GAMMA" \
        --top_p 0.9 \
        --temp 0.6 \
        --dataset "$DS" \
        2>&1 | tee "$LOG"

    kill $MEM_PID 2>/dev/null || true
    sleep 0.5
    PEAK_MEM=$(sort -n "$MEM_LOG" | tail -1 || echo "N/A")

    # Parse results
    BASELINE=$(grep -oP 'average latency: \K[0-9.]+' "$LOG" | head -1 || echo "N/A")
    ACCEPTANCE=$(grep -oP 'acceptance rate \(NOT per token\): \K[0-9.]+' "$LOG" || echo "N/A")
    TRIFORCE=$(grep -oP '\[TriForce\] average latency: \K[0-9.]+' "$LOG" || echo "N/A")
    SPEEDUP=$(grep -oP '\[E2E Speedup\]: \K[0-9.]+' "$LOG" || echo "N/A")

    echo "$DS,$BUDGET,$BASELINE,$ACCEPTANCE,$TRIFORCE,$SPEEDUP,$PEAK_MEM" >> "$CSV"

    echo ""
    echo "[DONE] $DS: baseline=${BASELINE}ms triforce=${TRIFORCE}ms speedup=${SPEEDUP}x accept=${ACCEPTANCE} peak_gpu=${PEAK_MEM}MB"
done

echo ""
echo "========================================="
echo "  TriForce Complete!"
echo "  Results: $CSV"
echo "========================================="
column -t -s',' "$CSV"
