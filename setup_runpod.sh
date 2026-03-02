#!/usr/bin/env bash
# One-click RunPod environment setup
# Usage: bash setup_runpod.sh
set -euo pipefail

echo "========================================="
echo " TriForce Experiment - RunPod Setup"
echo "========================================="

# 0. Check GPU
echo ""
echo "[Step 0] Checking GPU ..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# 1. System dependencies
echo "[Step 1] Installing system dependencies ..."
apt-get update -qq && apt-get install -y -qq git wget > /dev/null 2>&1
echo "[INFO] System dependencies OK"

# 2. Python dependencies
echo ""
echo "[Step 2] Installing Python dependencies ..."
pip install -q -r requirements.txt
echo "[INFO] Base dependencies installed"

echo ""
echo "[Step 2b] Installing flash-attn (this may take 5-10 minutes) ..."
pip install flash-attn==2.5.7 --no-build-isolation
echo "[INFO] flash-attn installed"

# 3. Clone TriForce
echo ""
echo "[Step 3] Cloning TriForce ..."
bash scripts/clone_triforce.sh

# 4. Prepare data
echo ""
echo "[Step 4] Preparing PG-19 data ..."
python3 scripts/prepare_data.py

# 5. Download models
echo ""
echo "[Step 5] Downloading models ..."
bash scripts/download_models.sh

echo ""
echo "========================================="
echo " Setup complete! Ready to run experiments"
echo "========================================="
echo ""
echo "Run the experiment with:"
echo "  cd vendor/TriForce"
echo "  python test/on_chip.py --prefill 124928 --budget 4096 \\"
echo "    --chunk_size 8 --draft_cache_budget 256 --gamma 6 \\"
echo "    --top_p 0.9 --temp 0.6 --dataset 128k"
