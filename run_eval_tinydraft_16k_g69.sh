#!/usr/bin/env bash
# TinyDraft 16K eval: γ=6,9
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TRAINED="${1:-/workspace/tf/checkpoints/tinydraft_phase_a_16k/final}"
RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"

echo "========================================="
echo "  TinyDraft 16K Eval (γ=6,9)"
echo "========================================="

for GAMMA in 6 9; do
    echo ""
    echo "--- Dataset: gs, Context: 16K, γ=$GAMMA ---"
    python3 sd_code/hl/eval_tinydraft.py \
        --target_model NousResearch/Yarn-Llama-2-7b-128k \
        --original_student JackFram/llama-68m \
        --trained_student "$TRAINED" \
        --dataset gs \
        --context long \
        --max_length 16384 \
        --gamma "$GAMMA" \
        --budgets "256,512,1024,2048" \
        --max_samples 15 \
        --warmup 1 \
        --output_csv "$RESULTS_DIR/eval_16k_gs_g${GAMMA}.csv"
done

echo ""
echo "========================================="
echo "  All Done!"
echo "========================================="
