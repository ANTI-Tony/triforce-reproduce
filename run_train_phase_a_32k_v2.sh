#!/usr/bin/env bash
# TinyDraft Phase A: 32K training (v2)
# - Dynamic RoPE scaling (NTK-aware, preserves short-range)
# - Resume from 16K checkpoint
# - Lower lr to prevent overfitting
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 sd_code/hl/train_tinydraft.py \
    --teacher_model NousResearch/Yarn-Llama-2-7b-128k \
    --student_model JackFram/llama-68m \
    --resume_from /workspace/tf/checkpoints/tinydraft_phase_a_16k/final \
    --seq_len 32768 \
    --cont_len 256 \
    --chunk_size 8 \
    --lam 0.5 \
    --lr 2e-6 \
    --weight_decay 0.01 \
    --warmup_steps 150 \
    --total_steps 3000 \
    --grad_clip 1.0 \
    --log_interval 10 \
    --save_interval 500 \
    --output_dir /workspace/tf/checkpoints/tinydraft_phase_a_32k_v2 \
    --rope_scale_factor 16 \
    --rope_scale_type dynamic \
    --seed 42
