#!/usr/bin/env bash
# SD eval: 8K & 16K, 3 datasets × 3 gammas × 5 budgets
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TRAINED="${1:-/workspace/tf/checkpoints/tinydraft_phase_a_16k/final}"
RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"

echo "========================================="
echo "  SD 8K & 16K Full Eval"
echo "  3 datasets × 3 gammas × 5 budgets"
echo "========================================="

for LEN in 8192 16384; do
  for DS in gs longbench_packed_qmsum lwm; do
    for GAMMA in 3 6 9; do
      echo ""
      echo "--- ${LEN} / ${DS} / γ=${GAMMA} ---"
      python3 sd_code/hl/eval_tinydraft.py \
          --target_model NousResearch/Yarn-Llama-2-7b-128k \
          --original_student JackFram/llama-68m \
          --trained_student "$TRAINED" \
          --dataset "$DS" \
          --context long \
          --max_length "$LEN" \
          --gamma "$GAMMA" \
          --budgets "256,512,1024,2048,3800" \
          --max_samples 10 \
          --warmup 1 \
          --output_csv "$RESULTS_DIR/eval_${LEN}_${DS}_g${GAMMA}.csv"
    done
  done
done

echo ""
echo "========================================="
echo "  All Done!"
echo "========================================="
