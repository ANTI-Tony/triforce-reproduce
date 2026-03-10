#!/usr/bin/env bash
# AR Baseline - Long-context + Short-context
# 3 datasets × 2 contexts = 6 runs
set -euo pipefail

export HF_HOME=/workspace/tf/hf_cache
export TRANSFORMERS_CACHE=/workspace/tf/hf_cache
export HF_DATASETS_CACHE=/workspace/tf/hf_cache/datasets

pip install 'transformers>=4.38,<4.45' -q

MODEL=$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('NousResearch/Yarn-Llama-2-7b-128k', local_files_only=True))")

RESULTS_DIR="results/ar"
mkdir -p "$RESULTS_DIR"
CSV="$RESULTS_DIR/ar_results.csv"
rm -f "$CSV"

SD_DIR="sd_code/hl"
DATASETS=("gs" "longbench_packed_qmsum" "lwm")
CONTEXTS=("long" "short")

echo "========================================="
echo "  AR Baseline Experiments"
echo "========================================="

for CTX in "${CONTEXTS[@]}"; do
    for DS in "${DATASETS[@]}"; do
        echo ""
        echo "--- AR | context=$CTX | dataset=$DS ---"
        LOG="$RESULTS_DIR/ar_${CTX}_${DS}.log"

        MEM_LOG="/tmp/gpu_mem_ar_${CTX}_${DS}.log"
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 1 > "$MEM_LOG" 2>/dev/null &
        MEM_PID=$!

        python3 "$SD_DIR/AR.py" \
            --context "$CTX" \
            --dataset "$DS" \
            --model_dir "$MODEL" \
            --max_new_tokens 256 \
            --max_samples 20 \
            --warmup 1 \
            --output_csv "$CSV" \
            2>&1 | tee "$LOG"

        kill $MEM_PID 2>/dev/null || true
        sleep 0.5
        PEAK_MEM=$(sort -n "$MEM_LOG" | tail -1 || echo "N/A")
        echo "[DONE] AR context=$CTX dataset=$DS peak_gpu=${PEAK_MEM}MB"
    done
done

echo ""
echo "========================================="
echo "  AR Baseline Complete!"
echo "  Results: $CSV"
echo "========================================="
cat "$CSV"
