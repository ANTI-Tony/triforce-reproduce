#!/usr/bin/env bash
# SD Experiments - All combinations
#
# Long-context (122K + 256):
#   SD(sparse/full): 3 datasets × 4 budgets × 3 gammas = 36 runs
#
# Short-context (3800 + 256):
#   SD(full/full):   3 datasets × 3 gammas = 9 runs
#   SD(sparse/full): 3 datasets × 4 budgets × 3 gammas = 36 runs
#
# Total: 36 + 9 + 36 = 81 runs
set -euo pipefail

export HF_HOME=/workspace/tf/hf_cache
export TRANSFORMERS_CACHE=/workspace/tf/hf_cache
export HF_DATASETS_CACHE=/workspace/tf/hf_cache/datasets

pip install 'transformers>=4.38,<4.45' -q

LARGE_MODEL=$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('NousResearch/Yarn-Llama-2-7b-128k', local_files_only=True))")
SMALL_MODEL=$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('JackFram/llama-68m', local_files_only=True))")

RESULTS_DIR="results/sd"
mkdir -p "$RESULTS_DIR"
CSV="$RESULTS_DIR/sd_results.csv"
rm -f "$CSV"

SD_DIR="sd_code/hl"
DATASETS=("gs" "longbench_packed_qmsum" "lwm")
GAMMAS=(3 6 9)
SPARSE_BUDGETS=(256 512 1024 2048)

run_sd() {
    local CTX=$1 DS=$2 BUDGET=$3 GAMMA=$4
    local TAG="${CTX}_${DS}_b${BUDGET}_g${GAMMA}"
    local LOG="$RESULTS_DIR/${TAG}.log"
    local MEM_LOG="/tmp/gpu_mem_sd_${TAG}.log"

    if [ "$BUDGET" -eq 0 ]; then
        local MAX_LEN=4056    # short-context only for full/full
    elif [ "$CTX" = "long" ]; then
        local MAX_LEN=125184
    else
        local MAX_LEN=4056
    fi

    echo ""
    echo "--- SD | ctx=$CTX ds=$DS budget=$BUDGET gamma=$GAMMA ---"

    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 1 > "$MEM_LOG" 2>/dev/null &
    local MEM_PID=$!

    python3 "$SD_DIR/SD.py" \
        --dataset "$DS" \
        --budget "$BUDGET" \
        --chunk_size 8 \
        --max_length "$MAX_LEN" \
        --max_new_tokens 256 \
        --gamma "$GAMMA" \
        --max_samples 20 \
        --warmup 1 \
        --skip_baseline \
        --small_model "$SMALL_MODEL" \
        --large_model "$LARGE_MODEL" \
        --output_csv "$CSV" \
        2>&1 | tee "$LOG"

    kill $MEM_PID 2>/dev/null || true
    sleep 0.5
    local PEAK_MEM
    PEAK_MEM=$(sort -n "$MEM_LOG" | tail -1 || echo "N/A")
    echo "[DONE] $TAG peak_gpu=${PEAK_MEM}MB"
}

echo "========================================="
echo "  SD Experiments"
echo "========================================="
echo "Target: $LARGE_MODEL"
echo "Drafter: $SMALL_MODEL"

# =============================================
# Part 1: Long-context SD(sparse/full) — 36 runs
# =============================================
echo ""
echo "############################################"
echo "# Part 1: Long-context SD(sparse/full)     #"
echo "############################################"

for DS in "${DATASETS[@]}"; do
    for BUDGET in "${SPARSE_BUDGETS[@]}"; do
        for GAMMA in "${GAMMAS[@]}"; do
            run_sd "long" "$DS" "$BUDGET" "$GAMMA"
        done
    done
done

# =============================================
# Part 2: Short-context SD(full/full) — 9 runs
# =============================================
echo ""
echo "############################################"
echo "# Part 2: Short-context SD(full/full)       #"
echo "############################################"

for DS in "${DATASETS[@]}"; do
    for GAMMA in "${GAMMAS[@]}"; do
        run_sd "short" "$DS" 0 "$GAMMA"
    done
done

# =============================================
# Part 3: Short-context SD(sparse/full) — 36 runs
# =============================================
echo ""
echo "############################################"
echo "# Part 3: Short-context SD(sparse/full)     #"
echo "############################################"

for DS in "${DATASETS[@]}"; do
    for BUDGET in "${SPARSE_BUDGETS[@]}"; do
        for GAMMA in "${GAMMAS[@]}"; do
            run_sd "short" "$DS" "$BUDGET" "$GAMMA"
        done
    done
done

echo ""
echo "========================================="
echo "  All SD Experiments Complete!"
echo "  Results: $CSV"
echo "========================================="
cat "$CSV"
