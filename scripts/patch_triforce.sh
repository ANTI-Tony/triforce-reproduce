#!/usr/bin/env bash
# Patch TriForce for numerical stability and short-sample filtering
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRIFORCE_DIR="$SCRIPT_DIR/../vendor/TriForce"

# --- Patch 1: sampling.py nan guard ---
TARGET="$TRIFORCE_DIR/utils/sampling.py"
if [ ! -f "$TARGET" ]; then
    echo "[ERROR] sampling.py not found"
    exit 1
fi
if ! grep -q "nan_to_num" "$TARGET"; then
    sed -i 's|def sample(probs : torch.Tensor, num_samples=1):|def sample(probs : torch.Tensor, num_samples=1):\n    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)\n    probs = torch.clamp(probs, min=0.0)\n    if probs.sum() == 0:\n        probs = torch.ones_like(probs) / probs.shape[-1]|' "$TARGET"
    echo "[INFO] Patched sampling.py (nan/inf guard)"
else
    echo "[INFO] sampling.py already patched"
fi

# --- Patch 2: dataset.py filter short samples ---
DATASET="$TRIFORCE_DIR/data/dataset.py"
if ! grep -q "filter short" "$DATASET"; then
    cat > /tmp/dataset_patch.py << 'PYEOF'
import re, sys

with open(sys.argv[1], 'r') as f:
    content = f.read()

# Replace the '128k' dataset loader to filter short samples
old = """    if dataset_name == '128k':
        datasetparent = "data/pg19/"
        d_files = os.listdir(datasetparent)
        dataset = load_dataset("json", data_files = [datasetparent + name for name in d_files], split = "train")
        tokenized_prompts = []
        for i in tqdm(range(len(dataset))):
            prompt = dataset[i]['text']
            tokenized_prompt = tokenizer.encode(prompt, return_tensors="pt")
            tokenized_prompts.append(tokenized_prompt)

        return tokenized_prompts"""

new = """    if dataset_name == '128k':
        datasetparent = "data/pg19/"
        d_files = os.listdir(datasetparent)
        dataset = load_dataset("json", data_files = [datasetparent + name for name in d_files], split = "train")
        tokenized_prompts = []
        skipped = 0
        for i in tqdm(range(len(dataset))):
            prompt = dataset[i]['text']
            tokenized_prompt = tokenizer.encode(prompt, return_tensors="pt")
            # filter short samples that would leave KV cache partially zero
            if datalen is not None and tokenized_prompt.shape[1] < datalen:
                skipped += 1
                continue
            tokenized_prompts.append(tokenized_prompt)
        if skipped > 0:
            print(f"[INFO] Skipped {skipped} samples shorter than {datalen} tokens")
            print(f"[INFO] Using {len(tokenized_prompts)} samples")

        return tokenized_prompts"""

if old in content:
    content = content.replace(old, new)
    with open(sys.argv[1], 'w') as f:
        f.write(content)
    print(f"[INFO] Patched dataset.py (filter short samples)")
else:
    print(f"[WARN] Could not find exact match in dataset.py, may already be patched")
PYEOF
    python3 /tmp/dataset_patch.py "$DATASET"
else
    echo "[INFO] dataset.py already patched"
fi
