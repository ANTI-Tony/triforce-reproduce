#!/usr/bin/env bash
# SD + Sparse KV Cache - Run all ablation experiments
# Usage: cd /workspace/triforce-reproduce && bash run_sd_sparse.sh
set -euo pipefail

export HF_HOME=/workspace/tf/hf_cache
export TRANSFORMERS_CACHE=/workspace/tf/hf_cache
export HF_DATASETS_CACHE=/workspace/tf/hf_cache/datasets

# Yarn-Llama-2-7b-128k needs transformers>=4.38 for yarn rope_scaling support
# (TriForce uses custom modeling_llama.py so it works with 4.36, but SD uses standard HF)
# Cap at <4.45 to stay compatible with PyTorch 2.2.x
pip install 'transformers>=4.38,<4.45' -q

# Auto-detect model paths
LARGE_MODEL=$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('NousResearch/Yarn-Llama-2-7b-128k', local_files_only=True))")
SMALL_MODEL=$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('JackFram/llama-68m', local_files_only=True))")

echo "========================================="
echo "  SD + Sparse KV Cache Experiment"
echo "========================================="
echo "Target: $LARGE_MODEL"
echo "Drafter: $SMALL_MODEL"
echo ""

RESULTS_DIR="results/sd_sparse"
mkdir -p "$RESULTS_DIR"
CSV="$RESULTS_DIR/sd_sparse_results.csv"

SD_DIR="sd_code/hl"

DATASETS=("gs" "longbench_packed_qmsum" "lwm")
# budget=0 = full cache SD (also runs AR baseline); others = sparse
BUDGETS=(0 512 1024 2048)

for DS in "${DATASETS[@]}"; do
    echo ""
    echo "========================================="
    echo "  Dataset: $DS"
    echo "========================================="

    for BUDGET in "${BUDGETS[@]}"; do
        echo ""
        echo "--- $DS | Budget: $BUDGET ---"
        LOG="$RESULTS_DIR/${DS}_budget${BUDGET}.log"

        # Only run AR baseline for budget=0 (full cache), skip for sparse runs
        EXTRA_ARGS=""
        if [ "$BUDGET" -ne 0 ]; then
            EXTRA_ARGS="--skip_baseline"
        fi

        python3 "$SD_DIR/SD.py" \
            --dataset "$DS" \
            --budget "$BUDGET" \
            --chunk_size 8 \
            --max_length 125184 \
            --max_new_tokens 256 \
            --gamma 3 \
            --max_samples 20 \
            --warmup 1 \
            --small_model "$SMALL_MODEL" \
            --large_model "$LARGE_MODEL" \
            --output_csv "$CSV" \
            $EXTRA_ARGS \
            2>&1 | tee "$LOG"

        echo "[DONE] $DS budget=$BUDGET"
    done
done

echo ""
echo "========================================="
echo "  All experiments complete!"
echo "  Results: $CSV"
echo "========================================="
echo ""
cat "$CSV"
