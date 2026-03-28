#!/usr/bin/env bash
# SD Long baseline: budget=3800, 3 datasets × 3 gammas
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"

echo "========================================="
echo "  SD Long Baseline: budget=3800"
echo "========================================="

for DS in gs longbench_packed_qmsum lwm; do
  for GAMMA in 3 6 9; do
    echo ""
    echo "--- Dataset: $DS, γ=$GAMMA, budget=3800 ---"
    python3 sd_code/hl/eval_tinydraft.py \
        --target_model NousResearch/Yarn-Llama-2-7b-128k \
        --original_student JackFram/llama-68m \
        --trained_student JackFram/llama-68m \
        --dataset "$DS" \
        --context long \
        --gamma "$GAMMA" \
        --budgets "3800" \
        --max_samples 10 \
        --warmup 1 \
        --output_csv "$RESULTS_DIR/eval_long_${DS}_b3800_g${GAMMA}.csv"
  done
done

echo ""
echo "========================================="
echo "  All Done!"
echo "========================================="
