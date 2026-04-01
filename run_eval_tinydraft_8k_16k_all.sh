#!/usr/bin/env bash
# TinyDraft 8K & 16K eval: longbench + lwm, γ=3
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TRAINED="${1:-/workspace/tf/checkpoints/tinydraft_phase_a_16k/final}"
RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"

echo "========================================="
echo "  TinyDraft 8K & 16K (longbench, lwm)"
echo "========================================="

for DS in longbench_packed_qmsum lwm; do
  for LEN in 8192 16384; do
    echo ""
    echo "--- Dataset: $DS, Context: ${LEN}, γ=3 ---"
    python3 sd_code/hl/eval_tinydraft.py \
        --target_model NousResearch/Yarn-Llama-2-7b-128k \
        --original_student JackFram/llama-68m \
        --trained_student "$TRAINED" \
        --dataset "$DS" \
        --context long \
        --max_length "$LEN" \
        --gamma 3 \
        --budgets "256,512,1024,2048" \
        --max_samples 10 \
        --warmup 1 \
        --output_csv "$RESULTS_DIR/eval_${LEN}_${DS}_g3.csv"
  done
done

echo ""
echo "========================================="
echo "  All Done!"
echo "========================================="
