#!/usr/bin/env bash
# TinyDraft Phase A Round 2: 32K context with RoPE scaling
# Resume from 16K checkpoint, train at 32K with RoPE factor=16
# A100 80GB — fits comfortably (~30GB for teacher KV cache)
# Student learns RoPE scaling; can extrapolate to 122K at inference with factor=64
set -euo pipefail

export HF_HOME=/workspace/tf/hf_cache
export TRANSFORMERS_CACHE=/workspace/tf/hf_cache
export HF_DATASETS_CACHE=/workspace/tf/hf_cache/datasets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

pip install 'transformers>=4.38,<4.45' -q
pip install datasets sentencepiece protobuf -q

TEACHER=$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('NousResearch/Yarn-Llama-2-7b-128k', local_files_only=True))")

# Resume from 16K checkpoint
RESUME_FROM="${1:-/workspace/tf/checkpoints/tinydraft_phase_a_16k/final}"
OUTPUT_DIR="/workspace/tf/checkpoints/tinydraft_phase_a_32k"
mkdir -p "$OUTPUT_DIR"

echo "========================================="
echo "  TinyDraft Phase A Round 2"
echo "  Context: 32K (prefix=32512, cont=256)"
echo "  RoPE: linear scaling factor=16"
echo "  Resume from: $RESUME_FROM"
echo "========================================="

python3 sd_code/hl/train_tinydraft.py \
    --teacher_model "$TEACHER" \
    --student_model "JackFram/llama-68m" \
    --resume_from "$RESUME_FROM" \
    --seq_len 32768 \
    --cont_len 256 \
    --chunk_size 8 \
    --lam 0.5 \
    --lr 5e-6 \
    --weight_decay 0.01 \
    --warmup_steps 100 \
    --total_steps 3000 \
    --grad_clip 1.0 \
    --log_interval 10 \
    --save_interval 500 \
    --rope_scale_factor 16.0 \
    --rope_scale_type linear \
    --output_dir "$OUTPUT_DIR" \
    --seed 42 \
    2>&1 | tee "$OUTPUT_DIR/train.log"

echo ""
echo "========================================="
echo "  Phase A Round 2 (32K) Complete!"
echo "========================================="
