#!/usr/bin/env bash
# Evaluate TinyDraft at progressive context lengths: 4K, 8K, 16K, 32K
# Tests position clamping effectiveness as context grows beyond max_pos=2048
# A100 80GB
set -euo pipefail

export HF_HOME=/workspace/tf/hf_cache
export TRANSFORMERS_CACHE=/workspace/tf/hf_cache
export HF_DATASETS_CACHE=/workspace/tf/hf_cache/datasets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TEACHER=$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('NousResearch/Yarn-Llama-2-7b-128k', local_files_only=True))")
ORIGINAL=$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('JackFram/llama-68m', local_files_only=True))")
TRAINED="${1:-/workspace/tf/checkpoints/tinydraft_phase_a_16k/final}"

RESULTS_DIR="results/tinydraft_eval_progressive"
mkdir -p "$RESULTS_DIR"

echo "========================================="
echo "  TinyDraft Progressive Context Evaluation"
echo "  Contexts: 4K, 8K, 16K, 32K"
echo "  Position clamping: auto (pos % 2048)"
echo "========================================="
echo "Trained: $TRAINED"
echo ""

LENGTHS=(4096 8192 16384 32768)
DS="gs"

for LEN in "${LENGTHS[@]}"; do
    echo ""
    echo "--- Context: ${LEN} tokens, Dataset: $DS, gamma=3 ---"
    python3 sd_code/hl/eval_tinydraft.py \
        --target_model "$TEACHER" \
        --original_student "$ORIGINAL" \
        --trained_student "$TRAINED" \
        --dataset "$DS" \
        --context long \
        --max_length "$LEN" \
        --gamma 3 \
        --budgets "256,512,1024,2048" \
        --max_samples 5 \
        --warmup 1 \
        --output_csv "$RESULTS_DIR/eval_${DS}_${LEN}_g3.csv" \
        2>&1 | tee "$RESULTS_DIR/eval_${DS}_${LEN}_g3.log"
done

echo ""
echo "========================================="
echo "  Progressive Evaluation Complete!"
echo "========================================="
echo ""
echo "Summary:"
for LEN in "${LENGTHS[@]}"; do
    echo "--- ${LEN} ---"
    cat "$RESULTS_DIR/eval_${DS}_${LEN}_g3.csv" 2>/dev/null || echo "(no results)"
    echo ""
done
