#!/usr/bin/env bash
# TinyDraft Phase A Training - 80GB A100
# L = L_A + 0.5 * L_C, β=0
# Start with 16K context, scale to 32K/64K/122K later
set -euo pipefail

export HF_HOME=/workspace/tf/hf_cache
export TRANSFORMERS_CACHE=/workspace/tf/hf_cache
export HF_DATASETS_CACHE=/workspace/tf/hf_cache/datasets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

pip install 'transformers>=4.38,<4.45' -q
pip install datasets -q
pip install flash-attn --no-build-isolation -q || true

# Pre-download models
echo "=== Checking models ==="
python3 -c "
from huggingface_hub import snapshot_download
print('Teacher:', snapshot_download('NousResearch/Yarn-Llama-2-7b-128k'))
print('Student:', snapshot_download('JackFram/llama-68m'))
"

TEACHER=$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('NousResearch/Yarn-Llama-2-7b-128k', local_files_only=True))")
STUDENT=$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('JackFram/llama-68m', local_files_only=True))")

OUTPUT_DIR="/workspace/tf/checkpoints/tinydraft_phase_a_16k"
mkdir -p "$OUTPUT_DIR"

echo ""
echo "========================================="
echo "  TinyDraft Phase A Training"
echo "  Context: 16K (prefix=16128, cont=256)"
echo "  Loss: L_A + 0.5 * L_C"
echo "  A100 80GB"
echo "========================================="
echo "Teacher: $TEACHER"
echo "Student: $STUDENT"
echo "Output:  $OUTPUT_DIR"
echo ""

python3 sd_code/hl/train_tinydraft.py \
    --teacher_model "$TEACHER" \
    --student_model "$STUDENT" \
    --seq_len 16384 \
    --cont_len 256 \
    --chunk_size 8 \
    --lam 0.5 \
    --lr 1e-5 \
    --weight_decay 0.01 \
    --warmup_steps 150 \
    --total_steps 5000 \
    --grad_clip 1.0 \
    --log_interval 10 \
    --save_interval 500 \
    --output_dir "$OUTPUT_DIR" \
    --seed 42 \
    2>&1 | tee "$OUTPUT_DIR/train.log"

echo ""
echo "========================================="
echo "  Phase A Training Complete!"
echo "  Checkpoints: $OUTPUT_DIR"
echo "========================================="
