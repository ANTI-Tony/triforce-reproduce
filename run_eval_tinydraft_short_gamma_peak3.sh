#!/usr/bin/env bash
# TinyDraft Short: find gamma peak round 3 (γ=30,33,36)
# Only lwm (gs and longbench already plateaued)
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TRAINED="${1:-/workspace/tf/checkpoints/tinydraft_phase_a_16k/final}"
RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"

echo "========================================="
echo "  TinyDraft Short: γ=30,33,36 (lwm only)"
echo "========================================="

for GAMMA in 30 33 36; do
    echo ""
    echo "--- lwm short γ=$GAMMA ---"
    python3 sd_code/hl/eval_tinydraft.py \
        --target_model NousResearch/Yarn-Llama-2-7b-128k \
        --original_student JackFram/llama-68m \
        --trained_student "$TRAINED" \
        --dataset lwm \
        --context short \
        --gamma "$GAMMA" \
        --budgets "256,512,1024,2048,3800" \
        --max_samples 10 \
        --warmup 1 \
        --output_csv "$RESULTS_DIR/eval_short_lwm_g${GAMMA}.csv"
done

echo ""
echo "========================================="
echo "  All Done!"
echo "========================================="
