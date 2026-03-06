"""
Dataset loader for SD sparse experiments.
Supports: gs (PG-19), longbench_packed_qmsum, lwm (NarrativeQA).
All prompts truncated to max_length tokens.
"""

import json
import os

# Ensure datasets cache goes to Volume Disk if available
_vol_cache = "/workspace/tf/hf_cache/datasets"
if os.path.isdir("/workspace/tf"):
    os.makedirs(_vol_cache, exist_ok=True)
    os.environ.setdefault("HF_DATASETS_CACHE", _vol_cache)

from datasets import load_dataset
from tqdm import tqdm


def load_prompts(dataset_name, tokenizer, max_length=4096, max_samples=20):
    """Load prompts from a dataset, tokenize and truncate to max_length."""
    if dataset_name == 'gs':
        return _load_pg19(tokenizer, max_length, max_samples)
    elif dataset_name == 'longbench_packed_qmsum':
        return _load_longbench_qmsum(tokenizer, max_length, max_samples)
    elif dataset_name == 'lwm':
        return _load_narrativeqa(tokenizer, max_length, max_samples)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def _load_pg19(tokenizer, max_length, max_samples):
    """PG-19 test set (first max_samples books)."""
    local_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'pg19_test.jsonl')
    if os.path.exists(local_path):
        texts = []
        with open(local_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                data = json.loads(line)
                texts.append(data['text'])
    else:
        dataset = load_dataset("pg19", split="test")
        texts = [dataset[i]['text'] for i in range(min(max_samples, len(dataset)))]

    prompts = []
    for text in tqdm(texts, desc="Tokenizing PG-19"):
        tokens = tokenizer.encode(text, truncation=True, max_length=max_length)
        if len(tokens) >= max_length:
            prompts.append({'text': text[:10000], 'tokens': tokens[:max_length]})

    if not prompts:
        for text in texts:
            tokens = tokenizer.encode(text, truncation=True, max_length=max_length)
            prompts.append({'text': text[:10000], 'tokens': tokens})

    print(f"[gs] {len(prompts)} prompts loaded (max_length={max_length})")
    return prompts


def _load_longbench_qmsum(tokenizer, max_length, max_samples):
    """LongBench QMSum - meeting transcripts."""
    try:
        dataset = load_dataset("THUDM/LongBench", "qmsum", split="test")
    except (RuntimeError, ValueError, FileNotFoundError):
        # Newer datasets lib doesn't support loading scripts
        # Download data.zip and extract qmsum.jsonl
        from huggingface_hub import hf_hub_download
        import zipfile
        zip_path = hf_hub_download(repo_id="THUDM/LongBench", filename="data.zip", repo_type="dataset")
        extract_dir = os.path.join(os.path.dirname(zip_path), "longbench_extracted")
        if not os.path.exists(os.path.join(extract_dir, "qmsum.jsonl")):
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(extract_dir)
        # Find qmsum.jsonl (might be in a subdirectory)
        qmsum_path = None
        for root, dirs, files in os.walk(extract_dir):
            for f in files:
                if f == "qmsum.jsonl":
                    qmsum_path = os.path.join(root, f)
                    break
        if qmsum_path is None:
            raise FileNotFoundError(f"qmsum.jsonl not found in {extract_dir}")
        data = []
        with open(qmsum_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        dataset = data

    prompts = []
    for item in tqdm(dataset, desc="Tokenizing QMSum"):
        if len(prompts) >= max_samples:
            break
        context = item.get('context', '') or item.get('input', '')
        query = item.get('input', '') if 'context' in item else ''
        text = context + "\n" + query if query else context
        tokens = tokenizer.encode(text, truncation=True, max_length=max_length)
        if len(tokens) >= max_length // 2:
            prompts.append({'text': text[:10000], 'tokens': tokens[:max_length]})

    print(f"[longbench_packed_qmsum] {len(prompts)} prompts loaded (max_length={max_length})")
    return prompts


def _load_narrativeqa(tokenizer, max_length, max_samples):
    """NarrativeQA - book summarization. Uses streaming to avoid full download."""
    idx_set = {0, 50, 300, 800, 950, 1100, 2150, 2450, 2550, 2750,
               3350, 3400, 3600, 3900, 4000, 4100, 4200, 4400, 4500, 4550}
    max_idx = max(idx_set)

    # Use streaming to avoid downloading the entire dataset
    try:
        stream = load_dataset("deepmind/narrativeqa", split="train", streaming=True)
    except Exception:
        stream = load_dataset("narrativeqa", split="train", streaming=True)

    # Collect items at target indices
    collected = {}
    for i, item in enumerate(tqdm(stream, desc="Streaming NarrativeQA", total=max_idx + 1)):
        if i in idx_set:
            collected[i] = item
        if i > max_idx:
            break

    prompts = []
    for idx in sorted(idx_set):
        if idx not in collected or len(prompts) >= max_samples:
            continue
        item = collected[idx]
        if isinstance(item.get('document'), dict):
            doc_text = item['document'].get('text', '')
        else:
            doc_text = item.get('document', '') or item.get('text', '')

        if not doc_text:
            continue

        book_tokens = tokenizer.encode(doc_text)[:max_length - 100]
        prompt = (
            "You are a helpful assistant. USER: Please read a part of the book below, "
            "and then give me the summary.\n[start of the book]\n"
            + tokenizer.decode(book_tokens, skip_special_tokens=True)
            + "\n[end of the book]\n\nNow you have read it. Please summarize it for me.\n\nASSISTANT: "
        )
        tokens = tokenizer.encode(prompt, truncation=True, max_length=max_length)
        prompts.append({'text': prompt[:10000], 'tokens': tokens[:max_length]})

    print(f"[lwm] {len(prompts)} prompts loaded (max_length={max_length})")
    return prompts
