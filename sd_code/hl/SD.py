"""
SD + TriForce Sparse KV Cache Experiment.

Runs standard Speculative Decoding with optional sparse drafter KV cache.
Measures autoregressive baseline, full-cache SD, and sparse SD.

Usage:
  python SD.py --dataset gs --budget 512
  python SD.py --dataset gs --budget 0        # full cache (no sparse)
  python SD.py --dataset gs --budget 0 --baseline_only  # AR baseline only
"""

import argparse
import csv
import os
import sys
import time

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add parent dirs to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from speculative.speculative_decoding import speculative_generate
from data_loader import load_prompts


def parse_args():
    parser = argparse.ArgumentParser(description="SD + Sparse KV Cache Experiment")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["gs", "longbench_packed_qmsum", "lwm"],
                        help="Dataset name")
    parser.add_argument("--budget", type=int, default=0,
                        help="Sparse budget (0 = full cache)")
    parser.add_argument("--chunk_size", type=int, default=8,
                        help="Chunk size for sparse selection")
    parser.add_argument("--max_length", type=int, default=4096,
                        help="Max prompt length in tokens")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Max new tokens to generate")
    parser.add_argument("--gamma", type=int, default=3,
                        help="Number of draft candidates")
    parser.add_argument("--max_samples", type=int, default=20,
                        help="Max samples from dataset")
    parser.add_argument("--warmup", type=int, default=1,
                        help="Number of warmup samples")
    parser.add_argument("--baseline_only", action="store_true",
                        help="Only run autoregressive baseline")
    parser.add_argument("--skip_baseline", action="store_true",
                        help="Skip autoregressive baseline")
    parser.add_argument("--small_model", type=str,
                        default="/workspace/tf/hf_cache/hub/models--JackFram--llama-68m/snapshots/3f29b1a9c3bbac380abfb81c7063e694e25a4e0f",
                        help="Path to drafter model")
    parser.add_argument("--large_model", type=str,
                        default="/workspace/tf/hf_cache/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9",
                        help="Path to target model")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Path to output CSV (appends)")
    return parser.parse_args()


def autoregressive_generate(input_ids, model, max_new_tokens, eos_token_id, pad_token_id):
    """Simple autoregressive generation for baseline timing."""
    device = model.device
    prompt_len = len(input_ids)
    max_seq = prompt_len + max_new_tokens

    ids = torch.full((1, max_seq), pad_token_id, dtype=torch.long, device=device)
    ids[0, :prompt_len] = torch.tensor(input_ids, dtype=torch.long, device=device)

    past = None
    pos = prompt_len

    # Prefill
    out = model(input_ids=ids[:, :prompt_len], past_key_values=past, use_cache=True)
    past = out.past_key_values
    next_token = torch.argmax(out.logits[:, -1, :], dim=-1)
    ids[0, pos] = next_token
    pos += 1

    if next_token.item() == eos_token_id:
        return ids[0, prompt_len:pos].tolist()

    # Decode
    while pos < max_seq:
        out = model(input_ids=ids[:, pos - 1:pos], past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_token = torch.argmax(out.logits[:, -1, :], dim=-1)
        ids[0, pos] = next_token
        pos += 1
        if next_token.item() == eos_token_id:
            break

    return ids[0, prompt_len:pos].tolist()


def run_baseline(prompts, model, tokenizer, max_new_tokens, warmup=1):
    """Run autoregressive baseline and return average ms/token."""
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    total_tokens = 0
    total_time = 0.0

    for i, prompt in enumerate(tqdm(prompts, desc="AR Baseline")):
        torch.cuda.synchronize()
        t0 = time.time()

        with torch.no_grad():
            tokens = autoregressive_generate(
                prompt['tokens'], model, max_new_tokens, eos_id, pad_id
            )

        torch.cuda.synchronize()
        t1 = time.time()

        if i < warmup:
            continue  # skip warmup samples

        n = len(tokens)
        total_tokens += n
        total_time += (t1 - t0)

    if total_tokens == 0:
        return 0.0, 0
    ms_per_token = (total_time / total_tokens) * 1000
    return ms_per_token, total_tokens


def run_sd(prompts, drafter, target, tokenizer, max_new_tokens, gamma,
           sparse_budget, chunk_size, warmup=1):
    """Run speculative decoding and return average ms/token + acceptance rate."""
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    total_tokens = 0
    total_time = 0.0
    total_accepted = 0
    total_speculated = 0

    budget = sparse_budget if sparse_budget > 0 else None

    for i, prompt in enumerate(tqdm(prompts, desc=f"SD (budget={sparse_budget})")):
        torch.cuda.synchronize()
        t0 = time.time()

        with torch.no_grad():
            output_tokens, accepted, speculated = speculative_generate(
                inputs=prompt['tokens'],
                drafter=drafter,
                target=target,
                tokenizer=tokenizer,
                gamma=gamma,
                max_gen_len=max_new_tokens,
                eos_tokens_id=eos_id,
                pad_token_id=pad_id,
                use_cache=True,
                sparse_budget=budget,
                chunk_size=chunk_size,
            )

        torch.cuda.synchronize()
        t1 = time.time()

        if i < warmup:
            continue

        n = len(output_tokens)
        total_tokens += n
        total_time += (t1 - t0)
        total_accepted += accepted
        total_speculated += speculated

    if total_tokens == 0:
        return 0.0, 0, 0.0
    ms_per_token = (total_time / total_tokens) * 1000
    acc_rate = total_accepted / total_speculated if total_speculated > 0 else 0.0
    return ms_per_token, total_tokens, acc_rate


def main():
    args = parse_args()

    print(f"=== SD Sparse Experiment ===")
    print(f"Dataset: {args.dataset}")
    print(f"Budget: {args.budget} ({'full cache' if args.budget == 0 else 'sparse'})")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Max length: {args.max_length}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Gamma: {args.gamma}")
    print()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.small_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    # Load prompts
    print(f"Loading dataset: {args.dataset}...")
    prompts = load_prompts(args.dataset, tokenizer, args.max_length, args.max_samples)
    print(f"Loaded {len(prompts)} prompts")
    print()

    # Load models
    print("Loading target model (Llama-2-7b, fp16)...")
    target = AutoModelForCausalLM.from_pretrained(
        args.large_model, torch_dtype=torch.float16, device_map='cuda:0'
    ).eval()

    print("Loading drafter model (llama-68m, fp32)...")
    drafter = AutoModelForCausalLM.from_pretrained(
        args.small_model, torch_dtype=torch.float32, device_map='cuda:0'
    ).eval()

    results = {}

    # Autoregressive baseline
    if not args.skip_baseline:
        print("\n=== Autoregressive Baseline ===")
        baseline_ms, baseline_tokens = run_baseline(
            prompts, target, tokenizer, args.max_new_tokens, warmup=args.warmup
        )
        print(f"Baseline: {baseline_ms:.2f} ms/token ({baseline_tokens} tokens)")
        results['baseline_ms'] = baseline_ms
    else:
        results['baseline_ms'] = 0.0

    if args.baseline_only:
        print("Done (baseline only).")
        return

    # Speculative decoding
    print(f"\n=== Speculative Decoding (budget={args.budget}) ===")
    sd_ms, sd_tokens, acc_rate = run_sd(
        prompts, drafter, target, tokenizer,
        args.max_new_tokens, args.gamma,
        args.budget, args.chunk_size, warmup=args.warmup
    )
    print(f"SD: {sd_ms:.2f} ms/token ({sd_tokens} tokens)")
    print(f"Acceptance rate: {acc_rate:.3f}")

    if results['baseline_ms'] > 0:
        speedup = results['baseline_ms'] / sd_ms if sd_ms > 0 else 0
        print(f"Speedup: {speedup:.3f}x")
    else:
        speedup = 0.0

    results['sd_ms'] = sd_ms
    results['acc_rate'] = acc_rate
    results['speedup'] = speedup

    # Print summary
    print(f"\n{'='*50}")
    print(f"Dataset: {args.dataset}")
    print(f"Budget: {args.budget}")
    print(f"Baseline: {results['baseline_ms']:.2f} ms/token")
    print(f"SD:       {sd_ms:.2f} ms/token")
    print(f"Speedup:  {speedup:.3f}x")
    print(f"Accept:   {acc_rate:.3f}")
    print(f"{'='*50}")

    # Append to CSV
    if args.output_csv:
        file_exists = os.path.exists(args.output_csv)
        with open(args.output_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['dataset', 'budget', 'baseline_ms', 'sd_ms', 'speedup', 'acceptance_rate'])
            writer.writerow([
                args.dataset, args.budget,
                f"{results['baseline_ms']:.2f}",
                f"{sd_ms:.2f}",
                f"{speedup:.3f}",
                f"{acc_rate:.3f}"
            ])
        print(f"Results appended to {args.output_csv}")


if __name__ == "__main__":
    main()
