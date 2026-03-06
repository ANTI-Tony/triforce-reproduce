"""
Dataset loader for SD sparse experiments.
Supports: gs (PG-19), longbench_packed_qmsum, lwm (NarrativeQA).
All prompts truncated to max_length tokens.
"""

import json
import os
from datasets import load_dataset
from tqdm import tqdm


def load_prompts(dataset_name, tokenizer, max_length=4096, max_samples=20):
    """Load prompts from a dataset, tokenize and truncate to max_length.

    Args:
        dataset_name: 'gs', 'longbench_packed_qmsum', or 'lwm'
        tokenizer: HuggingFace tokenizer
        max_length: max token length per prompt
        max_samples: max number of samples to return

    Returns:
        list of dicts: [{'text': str, 'tokens': list[int]}, ...]
    """
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
    # Try local file first
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
        # Fallback: load from HuggingFace
        dataset = load_dataset("pg19", split="test")
        texts = [dataset[i]['text'] for i in range(min(max_samples, len(dataset)))]

    prompts = []
    for text in tqdm(texts, desc=f"Tokenizing PG-19"):
        tokens = tokenizer.encode(text, truncation=True, max_length=max_length)
        if len(tokens) >= max_length:
            prompts.append({'text': text[:10000], 'tokens': tokens[:max_length]})

    if not prompts:
        # If all samples are shorter than max_length, use them anyway
        for text in texts:
            tokens = tokenizer.encode(text, truncation=True, max_length=max_length)
            prompts.append({'text': text[:10000], 'tokens': tokens})

    print(f"[gs] {len(prompts)} prompts loaded (max_length={max_length})")
    return prompts


def _load_longbench_qmsum(tokenizer, max_length, max_samples):
    """LongBench QMSum - meeting transcripts."""
    dataset = load_dataset("THUDM/LongBench", "qmsum", split="test",
                           trust_remote_code=True)

    prompts = []
    for item in tqdm(dataset, desc="Tokenizing QMSum"):
        if len(prompts) >= max_samples:
            break
        context = item.get('context', '') or item.get('input', '')
        query = item.get('input', '') if 'context' in item else ''
        text = context + "\n" + query if query else context
        tokens = tokenizer.encode(text, truncation=True, max_length=max_length)
        if len(tokens) >= max_length // 2:  # at least half the target length
            prompts.append({'text': text[:10000], 'tokens': tokens[:max_length]})

    print(f"[longbench_packed_qmsum] {len(prompts)} prompts loaded (max_length={max_length})")
    return prompts


def _load_narrativeqa(tokenizer, max_length, max_samples):
    """NarrativeQA - book summarization."""
    dataset = load_dataset("narrativeqa")
    idx = [0, 50, 300, 800, 950, 1100, 2150, 2450, 2550, 2750,
           3350, 3400, 3600, 3900, 4000, 4100, 4200, 4400, 4500, 4550]

    prompts = []
    for i in range(min(max_samples, len(idx))):
        doc_text = dataset['train'][idx[i]]['document']['text']
        # Use LWM-style prompt
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
