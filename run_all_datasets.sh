#!/usr/bin/env bash
# Run TriForce experiment on all three datasets and collect results
# Usage: cd /workspace/tf/triforce-reproduce/vendor/TriForce && bash ../../run_all_datasets.sh
set -euo pipefail

# Redirect HuggingFace cache to Volume Disk (more space than Container Disk)
export HF_HOME=/workspace/tf/hf_cache
export TRANSFORMERS_CACHE=/workspace/tf/hf_cache
export HF_DATASETS_CACHE=/workspace/tf/hf_cache/datasets

RESULTS_DIR="/workspace/tf/triforce-reproduce/results"
mkdir -p "$RESULTS_DIR"
CSV="$RESULTS_DIR/all_results.csv"

echo "dataset,log_file,baseline_ms,acceptance,triforce_ms,speedup" > "$CSV"

DATASETS=("gs" "longbench_packed_qmsum" "lwm")

for DS in "${DATASETS[@]}"; do
    echo ""
    echo "========================================="
    echo "  Running dataset: $DS"
    echo "========================================="
    echo ""

    LOG="$RESULTS_DIR/${DS}.log"

    python test/on_chip.py \
        --prefill 124928 \
        --gen_len 256 \
        --budget 4096 \
        --chunk_size 8 \
        --draft_cache_budget 256 \
        --gamma 6 \
        --top_p 0.9 \
        --temp 0.6 \
        --dataset "$DS" \
        2>&1 | tee "$LOG"

    # Parse results from log
    BASELINE=$(grep -oP 'average latency: \K[0-9.]+' "$LOG" | head -1)
    ACCEPTANCE=$(grep -oP 'acceptance rate \(NOT per token\): \K[0-9.]+' "$LOG")
    TRIFORCE=$(grep -oP '\[TriForce\] average latency: \K[0-9.]+' "$LOG")
    SPEEDUP=$(grep -oP '\[E2E Speedup\]: \K[0-9.]+' "$LOG")

    echo "$DS,${DS}.log,$BASELINE,$ACCEPTANCE,$TRIFORCE,$SPEEDUP" >> "$CSV"

    echo ""
    echo "[DONE] $DS: baseline=${BASELINE}ms, triforce=${TRIFORCE}ms, speedup=${SPEEDUP}x, acceptance=${ACCEPTANCE}"
    echo ""
done

echo ""
echo "========================================="
echo "  All experiments complete!"
echo "  Results: $CSV"
echo "========================================="
echo ""
column -t -s',' "$CSV"
