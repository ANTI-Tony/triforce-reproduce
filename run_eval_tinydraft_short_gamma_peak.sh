#!/usr/bin/env bash
# TinyDraft Short: find gamma peak (γ=12,15,18)
# 3 datasets × 3 gammas
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TRAINED="${1:-/workspace/tf/checkpoints/tinydraft_phase_a_16k/final}"
RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"

echo "========================================="
echo "  TinyDraft Short: γ=12,15,18 (find peak)"
echo "========================================="

for DS in gs longbench_packed_qmsum lwm; do
  for GAMMA in 12 15 18; do
    echo ""
    echo "--- ${DS} short γ=$GAMMA ---"
    python3 sd_code/hl/eval_tinydraft.py \
        --target_model NousResearch/Yarn-Llama-2-7b-128k \
        --original_student JackFram/llama-68m \
        --trained_student "$TRAINED" \
        --dataset "$DS" \
        --context short \
        --gamma "$GAMMA" \
        --budgets "256,512,1024,2048,3800" \
        --max_samples 10 \
        --warmup 1 \
        --output_csv "$RESULTS_DIR/eval_short_${DS}_g${GAMMA}.csv"
  done
done

echo ""
echo "========================================="
echo "  All Done!"
echo "========================================="
