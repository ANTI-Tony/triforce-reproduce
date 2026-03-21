#!/usr/bin/env bash
# Evaluate trained TinyDraft vs original student
# Run after Phase A training completes
set -euo pipefail

export HF_HOME=/workspace/tf/hf_cache
export TRANSFORMERS_CACHE=/workspace/tf/hf_cache
export HF_DATASETS_CACHE=/workspace/tf/hf_cache/datasets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Paths
TEACHER=$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('NousResearch/Yarn-Llama-2-7b-128k', local_files_only=True))")
ORIGINAL=$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('JackFram/llama-68m', local_files_only=True))")

# Use the final checkpoint (or change to step_XXXX for intermediate)
TRAINED="${1:-/workspace/tf/checkpoints/tinydraft_phase_a_16k/final}"

RESULTS_DIR="results/tinydraft_eval"
mkdir -p "$RESULTS_DIR"

echo "========================================="
echo "  TinyDraft Evaluation"
echo "========================================="
echo "Target:   $TEACHER"
echo "Original: $ORIGINAL"
echo "Trained:  $TRAINED"
echo ""

DATASETS=("gs" "longbench_packed_qmsum" "lwm")

for DS in "${DATASETS[@]}"; do
    echo ""
    echo "--- Dataset: $DS (short context, gamma=3) ---"
    python3 sd_code/hl/eval_tinydraft.py \
        --target_model "$TEACHER" \
        --original_student "$ORIGINAL" \
        --trained_student "$TRAINED" \
        --dataset "$DS" \
        --context short \
        --gamma 3 \
        --budgets "0,256,512,1024,2048,3800" \
        --max_samples 20 \
        --warmup 1 \
        --output_csv "$RESULTS_DIR/eval_${DS}_short_g3.csv" \
        2>&1 | tee "$RESULTS_DIR/eval_${DS}_short_g3.log"
done

echo ""
echo "========================================="
echo "  Evaluation Complete!"
echo "  Results: $RESULTS_DIR/"
echo "========================================="
