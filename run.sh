#!/usr/bin/env bash
# TriForce Reproduction Experiment - Full Pipeline
# Usage on RunPod:
#   git clone https://github.com/ANTI-Tony/triforce-reproduce.git /workspace/triforce-reproduce
#   cd /workspace/triforce-reproduce && bash run.sh
set -euo pipefail

# Redirect HuggingFace cache to Volume Disk (more space than Container Disk)
export HF_HOME=/workspace/tf/hf_cache
export TRANSFORMERS_CACHE=/workspace/tf/hf_cache
export HF_DATASETS_CACHE=/workspace/tf/hf_cache/datasets
mkdir -p "$HF_HOME"

echo "========================================="
echo "  TriForce Reproduction Experiment"
echo "========================================="
echo "[INFO] HF cache: $HF_HOME"

# 1. GPU Check
echo ""
echo "[Step 1/7] GPU Check"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# 2. Install exact PyTorch version matching TriForce
echo "[Step 2/7] Installing PyTorch 2.2.1+cu121 (TriForce requirement) ..."
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121
echo ""
echo "[Step 2b] Installing other Python dependencies ..."
pip install -r requirements.txt
echo ""
echo "[Step 2c] Installing flash-attn (may take 5-10 min to compile) ..."
pip install flash-attn==2.5.7 --no-build-isolation
echo ""

# 3. Verify packages
echo "[Step 3/7] Verifying packages ..."
python3 -c "
import torch, transformers
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
try:
    import flash_attn
    print(f'flash-attn: {flash_attn.__version__}')
except ImportError:
    print('WARN: flash-attn not installed')
"
echo ""

# 4. Clone TriForce + install its deps
echo "[Step 4/7] Cloning TriForce ..."
bash scripts/clone_triforce.sh
pip install -q -r vendor/TriForce/requirements.txt
bash scripts/patch_triforce.sh
echo ""

# 5. Prepare data
echo "[Step 5/7] Preparing PG-19 data ..."
python3 scripts/prepare_data.py
echo ""

# 6. Download models
echo "[Step 6/7] Downloading models ..."
bash scripts/download_models.sh
echo ""

# 7. Sanity check
echo "[Step 7/7] Sanity check (prefill=4096, one-shot) ..."
cd vendor/TriForce
python test/on_chip.py \
    --prefill 4096 \
    --gen_len 32 \
    --budget 4096 \
    --chunk_size 8 \
    --draft_cache_budget 256 \
    --gamma 6 \
    --top_p 0.9 \
    --temp 0.6 \
    --dataset one-shot

echo ""
echo "========================================="
echo "  Setup & sanity check PASSED!"
echo "  Now run the full experiment:"
echo ""
echo "  cd /workspace/triforce-reproduce/vendor/TriForce"
echo "  bash ../../run_experiment.sh"
echo "========================================="
