#!/usr/bin/env bash
# Run TinyDraft short-context eval: 3 datasets × γ=6,9
# (γ=3 already done)
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TRAINED="${1:-/workspace/tf/checkpoints/tinydraft_phase_a_16k/final}"
RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"

echo "========================================="
echo "  TinyDraft Short-Context Eval (γ=6,9)"
echo "========================================="

for DS in gs longbench_packed_qmsum lwm; do
  for GAMMA in 6 9; do
    echo ""
    echo "--- Dataset: $DS, γ=$GAMMA ---"
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
