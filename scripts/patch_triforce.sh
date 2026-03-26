#!/usr/bin/env bash
# Patch TriForce for numerical stability, short-sample filtering, and extra datasets
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

# --- Patch 2: dataset.py - rewrite with all fixes ---
DATASET="$TRIFORCE_DIR/data/dataset.py"
if ! grep -q "longbench_packed" "$DATASET"; then
    cp "$DATASET" "$DATASET.bak"
    cat > "$DATASET" << 'PYEOF'
from datasets import load_dataset
from tqdm import tqdm
import secrets
import random
import torch
import json
import os

def build_chat_input_lwm(tokenizer, message, prefill=127*1024):
    book = tokenizer.encode(message)[:prefill-84]
    prompt = "You are a helpful assistant. USER: Please read a part of the book below, and then give me the summary.\n[start of the book]\n" + tokenizer.decode(book, skip_special_tokens=True) + "\n[end of the book]\n\nNow you have read it. Please summarize it for me. First, tell me the title and the author, and then tell the story in 400 words.\n\nASSISTANT: "
    input_tokens = tokenizer.encode(prompt, return_tensors="pt")
    return input_tokens

def get_dataset(dataset_name, tokenizer=None, datalen=None, task=None):
    if dataset_name == '128k':
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
        return tokenized_prompts

    elif dataset_name == 'gs':
        datasetparent = "data/pg19/"
        d_files = os.listdir(datasetparent)
        dataset = load_dataset("json", data_files = [datasetparent + name for name in d_files], split = "train")
        tokenized_prompts = []
        skipped = 0
        for i in tqdm(range(min(20, len(dataset)))):
            prompt = dataset[i]['text']
            tokenized_prompt = tokenizer.encode(prompt, return_tensors="pt")
            if datalen is not None and tokenized_prompt.shape[1] < datalen:
                skipped += 1
                continue
            tokenized_prompts.append(tokenized_prompt)
        if skipped > 0:
            print(f"[INFO] Skipped {skipped} samples shorter than {datalen} tokens")
            print(f"[INFO] Using {len(tokenized_prompts)} samples")
        return tokenized_prompts

    elif dataset_name == 'one-shot':
        datasetparent = "data/pg19/"
        d_files = os.listdir(datasetparent)
        dataset = load_dataset("json", data_files = [datasetparent + name for name in d_files], split = "train")
        tokenized_prompts = []
        for i in tqdm(range(1)):
            prompt = dataset[i]['text']
            tokenized_prompt = tokenizer.encode(prompt, return_tensors="pt")
            tokenized_prompts.append(tokenized_prompt)
        return tokenized_prompts

    elif dataset_name == 'demo':
        dataset = load_dataset("narrativeqa")
        idx = [0, 50, 300, 800, 950, 1100, 2150, 2450, 2550, 2750, 3350, 3400, 3600, 3900, 4000, 4100, 4200, 4400, 4500, 4550]
        tokenized_prompts = []
        tokenized_prompt = build_chat_input_lwm(tokenizer, dataset['train'][idx[2]]['document']['text'][3:1024*500])
        tokenized_prompts.append(tokenized_prompt)
        return tokenized_prompts

    elif dataset_name == 'lwm':
        dataset = load_dataset("narrativeqa")
        idx = [0, 50, 300, 800, 950, 1100, 2150, 2450, 2550, 2750, 3350, 3400, 3600, 3900, 4000, 4100, 4200, 4400, 4500, 4550]
        tokenized_prompts = []
        for i in range(20):
            tokenized_prompt = build_chat_input_lwm(tokenizer, dataset['train'][idx[i]]['document']['text'][3:1024*500])
            if tokenized_prompt.shape[-1] != 127*1024:
                print(i, tokenized_prompt.shape)
                continue
            tokenized_prompts.append(tokenized_prompt)
        return tokenized_prompts

    elif dataset_name == 'longbench_packed_qmsum':
        # Load LongBench QMSum and pack samples to reach target context length
        dataset = load_dataset("THUDM/LongBench", "qmsum", split="test")
        target_len = datalen if datalen else 124928

        # Tokenize all samples
        all_tokens = []
        for item in tqdm(dataset, desc="Tokenizing QMSum"):
            context = item.get('context', '') or item.get('input', '')
            query = item.get('input', '') if 'context' in item else ''
            text = context + "\n" + query if query else context
            tokens = tokenizer.encode(text)
            all_tokens.append(tokens)

        # Pack samples together to reach target length
        tokenized_prompts = []
        current_pack = []
        current_len = 0
        for tokens in all_tokens:
            current_pack.extend(tokens)
            current_len += len(tokens)
            if current_len >= target_len:
                packed = torch.tensor([current_pack[:target_len]])
                tokenized_prompts.append(packed)
                current_pack = []
                current_len = 0

        print(f"[INFO] Packed {len(dataset)} QMSum samples into {len(tokenized_prompts)} prompts of {target_len} tokens")
        return tokenized_prompts

    else:
        raise Exception(f"Dataset not found: {dataset_name}")
PYEOF
    echo "[INFO] Patched dataset.py (filter + longbench_packed_qmsum)"
else
    echo "[INFO] dataset.py already patched"
fi
