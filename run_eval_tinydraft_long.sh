#!/usr/bin/env bash
# Evaluate TinyDraft on long context (122K) - A100 80GB
# Skip budget=0 (full cache at 122K would OOM for drafter)
set -euo pipefail

export HF_HOME=/workspace/tf/hf_cache
export TRANSFORMERS_CACHE=/workspace/tf/hf_cache
export HF_DATASETS_CACHE=/workspace/tf/hf_cache/datasets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TEACHER=$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('NousResearch/Yarn-Llama-2-7b-128k', local_files_only=True))")
ORIGINAL=$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('JackFram/llama-68m', local_files_only=True))")
TRAINED="${1:-/workspace/tf/checkpoints/tinydraft_phase_a_16k/final}"

RESULTS_DIR="results/tinydraft_eval_long"
mkdir -p "$RESULTS_DIR"

echo "========================================="
echo "  TinyDraft Long-Context Evaluation"
echo "  Context: 122K, gamma=3"
echo "========================================="
echo "Trained: $TRAINED"
echo ""

DATASETS=("gs" "longbench_packed_qmsum" "lwm")

for DS in "${DATASETS[@]}"; do
    echo ""
    echo "--- Dataset: $DS (long context, gamma=3) ---"
    python3 sd_code/hl/eval_tinydraft.py \
        --target_model "$TEACHER" \
        --original_student "$ORIGINAL" \
        --trained_student "$TRAINED" \
        --dataset "$DS" \
        --context long \
        --gamma 3 \
        --budgets "256,512,1024,2048,3800" \
        --max_samples 20 \
        --warmup 1 \
        --output_csv "$RESULTS_DIR/eval_${DS}_long_g3.csv" \
        2>&1 | tee "$RESULTS_DIR/eval_${DS}_long_g3.log"
done

echo ""
echo "========================================="
echo "  Long-Context Evaluation Complete!"
echo "========================================="
cat "$RESULTS_DIR"/*.csv
