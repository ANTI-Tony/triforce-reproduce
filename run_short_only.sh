#!/usr/bin/env bash
# Short-context only (A100 40GB)
# AR: 3 datasets = 3 runs
# SD(full/full): 3 datasets × 3 gammas = 9 runs
# SD(sparse/full): 3 datasets × 5 budgets × 3 gammas = 45 runs
# Total: 57 runs
set -euo pipefail

export HF_HOME=/workspace/tf/hf_cache
export TRANSFORMERS_CACHE=/workspace/tf/hf_cache
export HF_DATASETS_CACHE=/workspace/tf/hf_cache/datasets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

pip install 'transformers>=4.38,<4.45' -q
pip install flash-attn --no-build-isolation -q || true

LARGE_MODEL=$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('NousResearch/Yarn-Llama-2-7b-128k', local_files_only=True))")
SMALL_MODEL=$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('JackFram/llama-68m', local_files_only=True))")

RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR/ar" "$RESULTS_DIR/sd"
AR_CSV="$RESULTS_DIR/ar/ar_results.csv"
SD_CSV="$RESULTS_DIR/sd/sd_results.csv"
rm -f "$AR_CSV" "$SD_CSV"

SD_DIR="sd_code/hl"
DATASETS=("gs" "longbench_packed_qmsum" "lwm")
GAMMAS=(3 6 9)
SPARSE_BUDGETS=(256 512 1024 2048 3800)

echo "========================================="
echo "  Short-context Experiments (57 runs)"
echo "  A100 40GB"
echo "========================================="
echo "Target: $LARGE_MODEL"
echo "Drafter: $SMALL_MODEL"

# =============================================
# Part 1: AR Baseline (short) — 3 runs
# =============================================
echo ""
echo "############################################"
echo "# Part 1: AR Baseline (short) — 3 runs    #"
echo "############################################"

for DS in "${DATASETS[@]}"; do
    echo ""
    echo "--- AR | short | $DS ---"
    python3 "$SD_DIR/AR.py" \
        --context short \
        --dataset "$DS" \
        --model_dir "$LARGE_MODEL" \
        --max_new_tokens 256 \
        --max_samples 20 \
        --warmup 1 \
        --output_csv "$AR_CSV" \
        2>&1 | tee "$RESULTS_DIR/ar/ar_short_${DS}.log"
done

# =============================================
# Part 2: SD(full/full) — 9 runs
# =============================================
echo ""
echo "############################################"
echo "# Part 2: SD(full/full) short — 9 runs     #"
echo "############################################"

for DS in "${DATASETS[@]}"; do
    for GAMMA in "${GAMMAS[@]}"; do
        echo ""
        echo "--- SD full | $DS | gamma=$GAMMA ---"
        python3 "$SD_DIR/SD.py" \
            --dataset "$DS" \
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
            2>&1 | tee "$RESULTS_DIR/sd/sd_short_${DS}_b0_g${GAMMA}.log"
    done
done

# =============================================
# Part 3: SD(sparse/full) — 45 runs
# =============================================
echo ""
echo "############################################"
echo "# Part 3: SD(sparse/full) short — 45 runs  #"
echo "############################################"

for DS in "${DATASETS[@]}"; do
    for BUDGET in "${SPARSE_BUDGETS[@]}"; do
        for GAMMA in "${GAMMAS[@]}"; do
            echo ""
            echo "--- SD sparse | $DS | budget=$BUDGET | gamma=$GAMMA ---"
            python3 "$SD_DIR/SD.py" \
                --dataset "$DS" \
                --budget "$BUDGET" \
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
                2>&1 | tee "$RESULTS_DIR/sd/sd_short_${DS}_b${BUDGET}_g${GAMMA}.log"
        done
    done
done

echo ""
echo "========================================="
echo "  Short-context Experiments Complete!"
echo "  57 runs done"
echo "========================================="
echo "=== AR ==="
cat "$AR_CSV"
echo ""
echo "=== SD ==="
cat "$SD_CSV"
