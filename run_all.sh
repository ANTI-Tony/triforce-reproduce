#!/usr/bin/env bash
# 完整实验流程：AR(6) + SD(81) + TriForce(3) = 90 runs
# Usage: cd /workspace/triforce-reproduce && bash run_all.sh
set -euo pipefail

echo "============================================"
echo "  Full Experiment Pipeline (90 runs)"
echo "============================================"
echo ""

# ==============================
# Step 0: 环境搭建
# ==============================
echo "[Step 0] Setting up environment..."

export HF_HOME=/workspace/tf/hf_cache
export TRANSFORMERS_CACHE=/workspace/tf/hf_cache
export HF_DATASETS_CACHE=/workspace/tf/hf_cache/datasets
mkdir -p /workspace/tf/hf_cache

pip install accelerate datasets sentencepiece protobuf tqdm termcolor huggingface_hub -q
pip install 'transformers>=4.38,<4.45' -q
pip install flash-attn --no-build-isolation -q
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[Step 0] Downloading models..."
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('NousResearch/Yarn-Llama-2-7b-128k')
snapshot_download('JackFram/llama-68m')
print('Models downloaded.')
"

# ==============================
# Step 1: AR Baseline (6 runs)
# transformers>=4.38
# ==============================
echo ""
echo "============================================"
echo "  Step 1: AR Baseline (6 runs)"
echo "============================================"
bash run_ar.sh

# ==============================
# Step 2: SD Experiments (81 runs)
# transformers>=4.38
# ==============================
echo ""
echo "============================================"
echo "  Step 2: SD Experiments (81 runs)"
echo "============================================"
bash run_sd.sh

# ==============================
# Step 3: TriForce (3 runs)
# 需要降级 transformers==4.36.2 + flash-attn
# ==============================
echo ""
echo "============================================"
echo "  Step 3: TriForce (3 runs)"
echo "  Switching to transformers==4.36.2..."
echo "============================================"
pip install transformers==4.36.2 -q
pip install flash-attn --no-build-isolation -q

# Clone vendor/TriForce and apply patches
bash run.sh

cd vendor/TriForce
bash ../../run_triforce.sh
cd ../..

# ==============================
# Step 4: 汇总结果
# ==============================
echo ""
echo "============================================"
echo "  All 90 Experiments Complete!"
echo "============================================"
echo ""
echo "=== AR Baseline ==="
cat results/ar/ar_results.csv
echo ""
echo "=== SD Results ==="
cat results/sd/sd_results.csv
echo ""
echo "=== TriForce Results ==="
cat results/triforce/triforce_results.csv
