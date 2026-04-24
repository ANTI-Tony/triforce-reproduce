#!/usr/bin/env bash
# TinyDraft Phase A: L = L_A + 1.0 * L_C (λ=1.0)
# Compare with previous λ=0.5
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 sd_code/hl/train_tinydraft.py \
    --teacher_model NousResearch/Yarn-Llama-2-7b-128k \
    --student_model JackFram/llama-68m \
    --seq_len 16384 \
    --cont_len 256 \
    --chunk_size 8 \
    --lam 1.0 \
    --lr 1e-5 \
    --weight_decay 0.01 \
    --warmup_steps 150 \
    --total_steps 5000 \
    --grad_clip 1.0 \
    --log_interval 10 \
    --save_interval 500 \
    --output_dir /workspace/tf/checkpoints/tinydraft_phase_a_1c \
    --seed 42
