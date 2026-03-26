#!/usr/bin/env bash
# Run the full TriForce experiment
# Usage: cd /workspace/triforce-reproduce/vendor/TriForce && bash ../../run_experiment.sh
set -euo pipefail

RESULTS_DIR="/workspace/triforce-reproduce/results"
mkdir -p "$RESULTS_DIR"
LOG_FILE="$RESULTS_DIR/triforce_full_run.log"

echo "========================================="
echo "  TriForce Full Experiment"
echo "  prefill=124928, gen_len=256"
echo "  budget=4096, chunk_size=8, gamma=6"
echo "  dataset=128k (all PG-19 samples)"
echo "========================================="
echo ""

python test/on_chip.py \
    --prefill 124928 \
    --gen_len 256 \
    --budget 4096 \
    --chunk_size 8 \
    --draft_cache_budget 256 \
    --gamma 6 \
    --top_p 0.9 \
    --temp 0.6 \
    --dataset 128k \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "========================================="
echo "  Experiment complete!"
echo "  Log saved to: $LOG_FILE"
echo "========================================="
echo ""
echo "Expected results (from paper):"
echo "  Autoregressive:  ~46-48 ms/token"
echo "  TriForce:        ~21-22 ms/token"
echo "  Speedup:         ~2.1-2.2x"
echo "  Acceptance rate:  ~0.85-0.90"
echo ""
nvidia-smi
