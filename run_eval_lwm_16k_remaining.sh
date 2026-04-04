#!/usr/bin/env bash
# Remaining: lwm 16K, γ=3,6,9
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TRAINED="${1:-/workspace/tf/checkpoints/tinydraft_phase_a_16k/final}"
RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"

echo "========================================="
echo "  lwm 16K (γ=3,6,9) — remaining 3 groups"
echo "========================================="

for GAMMA in 3 6 9; do
    echo ""
    echo "--- lwm 16K γ=$GAMMA ---"
    python3 sd_code/hl/eval_tinydraft.py \
        --target_model NousResearch/Yarn-Llama-2-7b-128k \
        --original_student JackFram/llama-68m \
        --trained_student "$TRAINED" \
        --dataset lwm \
        --context long \
        --max_length 16384 \
        --gamma "$GAMMA" \
        --budgets "256,512,1024,2048,3800" \
        --max_samples 10 \
        --warmup 1 \
        --output_csv "$RESULTS_DIR/eval_16384_lwm_g${GAMMA}.csv"
done

echo ""
echo "========================================="
echo "  All Done!"
echo "========================================="
