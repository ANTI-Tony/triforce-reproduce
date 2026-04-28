#!/usr/bin/env bash
# TinyDraft: L = L_A + 0.5*L_C + β*L_B (hinge top-k KL)
# Resume from A+0.5C checkpoint
# β warmup: 0 for first 10%, then linear to 0.2
# L_B computed every 2 steps
# Budget: {256,512,1024,2048}, weights {0.4,0.3,0.2,0.1}
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 sd_code/hl/train_tinydraft.py \
    --teacher_model NousResearch/Yarn-Llama-2-7b-128k \
    --student_model JackFram/llama-68m \
    --resume_from /workspace/tf/checkpoints/tinydraft_phase_a_16k/final \
    --seq_len 16384 \
    --cont_len 256 \
    --chunk_size 8 \
    --lam 0.5 \
    --beta 0.2 \
    --lb_every_n 2 \
    --lr 1e-5 \
    --weight_decay 0.01 \
    --warmup_steps 150 \
    --total_steps 5000 \
    --grad_clip 1.0 \
    --log_interval 10 \
    --save_interval 500 \
    --output_dir /workspace/tf/checkpoints/tinydraft_phase_abc_v2 \
    --seed 42
