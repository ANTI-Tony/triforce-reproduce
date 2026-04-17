#!/usr/bin/env bash
# EAGLE-3 evaluation at prompt=2048
# 4 short datasets + 3 long datasets (truncated to 2048)
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RESULTS_DIR="results/eagle3"
mkdir -p "$RESULTS_DIR"

# Clone Spec-Bench if needed
if [ ! -d "/workspace/Spec-Bench" ]; then
    cd /workspace && git clone https://github.com/hemingkx/Spec-Bench.git
fi

# Copy datasets to workspace
mkdir -p /workspace/datasets
cp /workspace/triforce-experiment/data/humaneval_python.jsonl /workspace/datasets/ 2>/dev/null || true
cp /workspace/triforce-experiment/data/GSM8K.jsonl /workspace/datasets/ 2>/dev/null || true
cp /workspace/triforce-experiment/data/dolly.jsonl /workspace/datasets/ 2>/dev/null || true
cp /workspace/Spec-Bench/data/spec_bench/question.jsonl /workspace/datasets/specbench.jsonl 2>/dev/null || true

echo "========================================="
echo "  EAGLE-3 Evaluation (prompt<=2048)"
echo "  Short: HumanEval, GSM8K, Dolly, SpecBench"
echo "  Long (truncated): gs, longbench, lwm"
echo "========================================="

cd /workspace/EAGLE

python3 << 'PYEOF'
import torch
import time
import json
import os
import sys

sys.path.insert(0, '/workspace/triforce-experiment')

from eagle.model.ea_model import EaModel

# Load model once
print("Loading EAGLE-3 model...")
model = EaModel.from_pretrained(
    use_eagle3=True,
    base_model_path='meta-llama/Llama-3.1-8B-Instruct',
    ea_model_path='yuhuili/EAGLE3-LLaMA3.1-Instruct-8B',
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map='auto',
    total_token=60,
    depth=5,
    top_k=10,
)
model.eval()
tokenizer = model.tokenizer
print("Model loaded!")

MAX_PROMPT = 2048
MAX_NEW_TOKENS = 256
MAX_SAMPLES = 20

def load_dataset_prompts(name):
    """Load prompts from various datasets, truncate to MAX_PROMPT tokens."""
    prompts = []

    if name == "humaneval":
        path = "/workspace/datasets/humaneval_python.jsonl"
        if not os.path.exists(path):
            path = "/workspace/triforce-experiment/data/humaneval_python.jsonl"
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                prompts.append(d.get("prompt", ""))
    elif name == "gsm8k":
        path = "/workspace/datasets/GSM8K.jsonl"
        if not os.path.exists(path):
            path = "/workspace/triforce-experiment/data/GSM8K.jsonl"
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                prompts.append(d.get("prompt", d.get("question", "")))
    elif name == "dolly":
        path = "/workspace/datasets/dolly.jsonl"
        if not os.path.exists(path):
            path = "/workspace/triforce-experiment/data/dolly.jsonl"
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                prompts.append(d.get("prompt", ""))
    elif name == "specbench":
        path = "/workspace/datasets/specbench.jsonl"
        if not os.path.exists(path):
            path = "/workspace/Spec-Bench/data/spec_bench/question.jsonl"
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                turns = d.get("turns", [])
                if turns:
                    prompts.append(turns[0])
    elif name in ["gs", "longbench", "lwm"]:
        # Use our data_loader for long datasets
        from sd_code.hl.data_loader import load_prompts as lp
        ds_name = "longbench_packed_qmsum" if name == "longbench" else name
        data = lp(ds_name, tokenizer, max_length=MAX_PROMPT, max_samples=MAX_SAMPLES+1)
        for d in data:
            text = tokenizer.decode(d['tokens'], skip_special_tokens=True)
            prompts.append(text)

    return prompts[:MAX_SAMPLES+1]  # +1 for warmup


def run_eagle3_eval(dataset_name, prompts):
    """Run EAGLE-3 evaluation on a list of prompts."""
    print(f"\n--- EAGLE-3: {dataset_name} (max_prompt={MAX_PROMPT}) ---")

    # Tokenize and truncate
    tokenized = []
    for p in prompts:
        ids = tokenizer.encode(p, truncation=True, max_length=MAX_PROMPT)
        tokenized.append(ids)

    print(f"  {len(tokenized)} prompts, avg length: {sum(len(t) for t in tokenized)/len(tokenized):.0f} tokens")

    # Warmup
    input_ids = torch.tensor([tokenized[0]], device='cuda')
    _ = model.eagenerate(input_ids, temperature=0.0, max_new_tokens=32)

    # Also run AR baseline
    ar_times = []
    eagle_times = []
    results = []

    for i, tokens in enumerate(tokenized[1:]):  # skip warmup
        input_ids = torch.tensor([tokens], device='cuda')
        prompt_len = input_ids.shape[1]

        # AR baseline
        torch.cuda.synchronize()
        t0 = time.time()
        ar_out = model.base_model.generate(input_ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        torch.cuda.synchronize()
        ar_time = time.time() - t0
        ar_gen = ar_out.shape[1] - prompt_len

        # EAGLE-3
        torch.cuda.synchronize()
        t0 = time.time()
        eagle_out = model.eagenerate(input_ids, temperature=0.0, max_new_tokens=MAX_NEW_TOKENS)
        torch.cuda.synchronize()
        eagle_time = time.time() - t0
        eagle_gen = eagle_out.shape[1] - prompt_len

        ar_tps = ar_gen / ar_time if ar_time > 0 else 0
        eagle_tps = eagle_gen / eagle_time if eagle_time > 0 else 0
        speedup = eagle_tps / ar_tps if ar_tps > 0 else 0

        print(f"  sample {i+1}: prompt={prompt_len}, AR={ar_tps:.1f}tok/s, EAGLE={eagle_tps:.1f}tok/s, speedup={speedup:.2f}x")

        results.append({
            'prompt_len': prompt_len,
            'ar_throughput': ar_tps,
            'eagle_throughput': eagle_tps,
            'speedup': speedup,
            'ar_gen': int(ar_gen),
            'eagle_gen': int(eagle_gen),
        })

    # Summary
    avg_ar = sum(r['ar_throughput'] for r in results) / len(results)
    avg_eagle = sum(r['eagle_throughput'] for r in results) / len(results)
    avg_speedup = avg_eagle / avg_ar if avg_ar > 0 else 0

    print(f"\n  [{dataset_name}] AR={avg_ar:.1f}tok/s, EAGLE-3={avg_eagle:.1f}tok/s, Speedup={avg_speedup:.2f}x")

    # Save
    out_path = f"/workspace/triforce-experiment/results/eagle3/eagle3_2k_{dataset_name}.json"
    with open(out_path, 'w') as f:
        json.dump({
            'dataset': dataset_name,
            'max_prompt': MAX_PROMPT,
            'avg_ar_throughput': avg_ar,
            'avg_eagle_throughput': avg_eagle,
            'avg_speedup': avg_speedup,
            'results': results,
        }, f, indent=2)
    print(f"  Saved: {out_path}")


# Run all datasets
datasets = ["humaneval", "gsm8k", "dolly", "specbench", "gs", "longbench", "lwm"]

for ds in datasets:
    try:
        prompts = load_dataset_prompts(ds)
        if prompts:
            run_eagle3_eval(ds, prompts)
        else:
            print(f"\n--- {ds}: No prompts loaded, skipping ---")
    except Exception as e:
        print(f"\n--- {ds}: ERROR: {e} ---")
        import traceback
        traceback.print_exc()

print("\n=========================================")
print("  All Done!")
print("=========================================")
PYEOF
