#!/usr/bin/env bash
# Ablation: A-only vs A+C at short context
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"

echo "========================================="
echo "  Ablation: A-only vs A+C"
echo "========================================="

# A-only checkpoint
echo ""
echo "--- A-only (full cache training) ---"
python3 sd_code/hl/eval_tinydraft.py \
    --target_model NousResearch/Yarn-Llama-2-7b-128k \
    --original_student JackFram/llama-68m \
    --trained_student /workspace/tf/checkpoints/tinydraft_a_only/final \
    --dataset gs \
    --context short \
    --gamma 3 \
    --budgets "256,512,1024,2048,3800" \
    --max_samples 10 \
    --warmup 1 \
    --output_csv "$RESULTS_DIR/eval_ablation_a_only.csv"

# A+C checkpoint (existing 16K trained)
echo ""
echo "--- A+C (sparse multi-view training) ---"
python3 sd_code/hl/eval_tinydraft.py \
    --target_model NousResearch/Yarn-Llama-2-7b-128k \
    --original_student JackFram/llama-68m \
    --trained_student /workspace/tf/checkpoints/tinydraft_phase_a_16k/final \
    --dataset gs \
    --context short \
    --gamma 3 \
    --budgets "256,512,1024,2048,3800" \
    --max_samples 10 \
    --warmup 1 \
    --output_csv "$RESULTS_DIR/eval_ablation_ac.csv"

echo ""
echo "========================================="
echo "  All Done!"
echo "========================================="
