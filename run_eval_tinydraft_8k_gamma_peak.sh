#!/usr/bin/env bash
# TinyDraft 8K: find gamma peak (γ=12,15,18)
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TRAINED="${1:-/workspace/tf/checkpoints/tinydraft_phase_a_16k/final}"
RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"

echo "========================================="
echo "  TinyDraft 8K: γ=12,15,18 (find peak)"
echo "========================================="

for GAMMA in 12 15 18; do
    echo ""
    echo "--- gs 8K γ=$GAMMA ---"
    python3 sd_code/hl/eval_tinydraft.py \
        --target_model NousResearch/Yarn-Llama-2-7b-128k \
        --original_student JackFram/llama-68m \
        --trained_student "$TRAINED" \
        --dataset gs \
        --context long \
        --max_length 8192 \
        --gamma "$GAMMA" \
        --budgets "256,512,1024,2048" \
        --max_samples 10 \
        --warmup 1 \
        --output_csv "$RESULTS_DIR/eval_8k_gs_g${GAMMA}.csv"
done

echo ""
echo "========================================="
echo "  All Done!"
echo "========================================="
