#!/usr/bin/env bash
# TinyDraft eval at 32K and 64K context
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TRAINED="${1:-/workspace/tf/checkpoints/tinydraft_phase_a_16k/final}"
RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"

echo "========================================="
echo "  TinyDraft Long-Context Eval (32K, 64K)"
echo "========================================="

for LEN in 32768 65536; do
    echo ""
    echo "--- Context: ${LEN}, Dataset: gs, γ=3 ---"
    python3 sd_code/hl/eval_tinydraft.py \
        --target_model NousResearch/Yarn-Llama-2-7b-128k \
        --original_student JackFram/llama-68m \
        --trained_student "$TRAINED" \
        --dataset gs \
        --context long \
        --max_length "$LEN" \
        --gamma 3 \
        --budgets "256,512,1024,2048" \
        --max_samples 10 \
        --warmup 1 \
        --output_csv "$RESULTS_DIR/eval_${LEN}_gs_g3.csv"
done

echo ""
echo "========================================="
echo "  All Done!"
echo "========================================="
