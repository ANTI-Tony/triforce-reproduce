#!/usr/bin/env bash
# Dolly validation: AR baseline + SD(full/full) γ=3,6,9
set -euo pipefail

export HF_HOME=/workspace/tf/hf_cache
export TRANSFORMERS_CACHE=/workspace/tf/hf_cache
export HF_DATASETS_CACHE=/workspace/tf/hf_cache/datasets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

pip install 'transformers>=4.38,<4.45' -q
pip install flash-attn --no-build-isolation -q

LARGE_MODEL=$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('NousResearch/Yarn-Llama-2-7b-128k', local_files_only=True))")
SMALL_MODEL=$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('JackFram/llama-68m', local_files_only=True))")

# Copy dolly data to data/ dir if not already there
mkdir -p data
if [ ! -f data/validation-00000-of-00001.jsonl ]; then
    cp ~/Desktop/validation-00000-of-00001.jsonl data/ 2>/dev/null || true
fi

RESULTS_DIR="results/dolly"
mkdir -p "$RESULTS_DIR"
AR_CSV="$RESULTS_DIR/dolly_ar.csv"
SD_CSV="$RESULTS_DIR/dolly_sd.csv"

SD_DIR="sd_code/hl"

echo "========================================="
echo "  Dolly Validation: AR + SD(full)"
echo "========================================="
echo "Target: $LARGE_MODEL"
echo "Drafter: $SMALL_MODEL"

# =============================================
# Step 1: AR Baseline
# =============================================
echo ""
echo "--- AR Baseline (dolly) ---"
python3 "$SD_DIR/AR.py" \
    --context short \
    --dataset dolly \
    --model_dir "$LARGE_MODEL" \
    --max_new_tokens 256 \
    --max_samples 20 \
    --warmup 1 \
    --output_csv "$AR_CSV" \
    2>&1 | tee "$RESULTS_DIR/ar_dolly.log"

# =============================================
# Step 2: SD(full/full) γ=3,6,9
# =============================================
for GAMMA in 3 6 9; do
    echo ""
    echo "--- SD(full) dolly gamma=$GAMMA ---"
    python3 "$SD_DIR/SD.py" \
        --dataset dolly \
        --budget 0 \
        --chunk_size 8 \
        --max_length 4056 \
        --max_new_tokens 256 \
        --gamma "$GAMMA" \
        --max_samples 20 \
        --warmup 1 \
        --skip_baseline \
        --small_model "$SMALL_MODEL" \
        --large_model "$LARGE_MODEL" \
        --output_csv "$SD_CSV" \
        2>&1 | tee "$RESULTS_DIR/sd_dolly_g${GAMMA}.log"
done

echo ""
echo "========================================="
echo "  Dolly Experiments Complete!"
echo "========================================="
echo "=== AR ==="
cat "$AR_CSV"
echo ""
echo "=== SD ==="
cat "$SD_CSV"
