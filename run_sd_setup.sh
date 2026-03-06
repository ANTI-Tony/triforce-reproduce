#!/usr/bin/env bash
# SD + Sparse KV Cache Experiment - Environment Setup for RunPod
# Usage:
#   git clone https://github.com/ANTI-Tony/triforce-reproduce.git /workspace/triforce-reproduce
#   cd /workspace/triforce-reproduce && bash run_sd_setup.sh
set -euo pipefail

export HF_HOME=/workspace/tf/hf_cache
export TRANSFORMERS_CACHE=/workspace/tf/hf_cache
export HF_DATASETS_CACHE=/workspace/tf/hf_cache/datasets
mkdir -p "$HF_HOME"

echo "========================================="
echo "  SD + Sparse Cache - Setup"
echo "========================================="
echo "[INFO] HF cache: $HF_HOME"

# 1. GPU Check
echo ""
echo "[Step 1/4] GPU Check"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# 2. Install dependencies
echo "[Step 2/4] Installing Python dependencies ..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate datasets tqdm
echo ""

# 3. Download models (Llama-2-7b-hf + llama-68m)
echo "[Step 3/4] Downloading models ..."

# Target: NousResearch/Llama-2-7b-hf (ungated mirror, ~14GB)
echo "[3a] Downloading NousResearch/Llama-2-7b-hf ..."
python3 -c "
from huggingface_hub import snapshot_download
path = snapshot_download('NousResearch/Llama-2-7b-hf')
print(f'[INFO] Llama-2-7b-hf downloaded to: {path}')
"

# Draft: JackFram/llama-68m (~260MB)
echo "[3b] Downloading JackFram/llama-68m ..."
python3 -c "
from huggingface_hub import snapshot_download
path = snapshot_download('JackFram/llama-68m')
print(f'[INFO] llama-68m downloaded to: {path}')
"

# Print model paths
echo ""
echo "[Step 4/4] Model paths:"
python3 -c "
from huggingface_hub import snapshot_download
p1 = snapshot_download('NousResearch/Llama-2-7b-hf', local_files_only=True)
p2 = snapshot_download('JackFram/llama-68m', local_files_only=True)
print(f'  Target (Llama-2-7b): {p1}')
print(f'  Drafter (llama-68m): {p2}')
"

echo ""
echo "========================================="
echo "  Setup complete!"
echo "  Run experiments with:"
echo "    bash run_sd_sparse.sh"
echo "========================================="
