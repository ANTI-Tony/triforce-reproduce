#!/usr/bin/env bash
# Pre-download model weights using huggingface_hub
set -euo pipefail

echo "========================================="
echo " Downloading TriForce model weights"
echo "========================================="

# Target model: Yarn-Llama-2-7b-128k (~14GB)
echo ""
echo "[1/2] Downloading NousResearch/Yarn-Llama-2-7b-128k ..."
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'NousResearch/Yarn-Llama-2-7b-128k',
    ignore_patterns=['*.bin'],  # prefer safetensors
)
print('[INFO] Target model downloaded successfully')
"

# Draft model: llama-68m (~260MB)
echo ""
echo "[2/2] Downloading JackFram/llama-68m ..."
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('JackFram/llama-68m')
print('[INFO] Draft model downloaded successfully')
"

echo ""
echo "========================================="
echo " All models downloaded!"
echo "========================================="
