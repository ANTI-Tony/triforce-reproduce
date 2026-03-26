#!/usr/bin/env python3
"""
Evaluate TinyDraft: run SD with trained student vs original student.
Compares acceptance rate & throughput across budgets and datasets.
"""

import torch
import argparse
import csv
import os
import sys
import time

from transformers import AutoModelForCausalLM, AutoTokenizer

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, "speculative"))
from data_loader import load_prompts
from speculative_decoding import speculative_generate


def run_sd_eval(target, drafter, tokenizer, prompts, gamma, budget, chunk_size,
                max_new_tokens, warmup, device, drafter_max_pos=None):
    """Run SD and return (throughput, acceptance_rate, total_tokens, total_time)."""
    eos_id = tokenizer.eos_token_id or 2
    total_tokens = 0
    total_time = 0.0
    total_accepted = 0.0
    total_speculated = 0.0

    n_prompts = len(prompts)
    for i, prompt in enumerate(prompts):
        tokens = prompt["tokens"]
        print(f"      sample {i+1}/{n_prompts} (prompt={len(tokens)} tokens)...", end="", flush=True)
        torch.cuda.synchronize()
        t0 = time.time()

        output, accepted, speculated = speculative_generate(
            inputs=tokens,
            drafter=drafter,
            target=target,
            tokenizer=tokenizer,
            gamma=gamma,
            max_gen_len=max_new_tokens,
            eos_tokens_id=eos_id,
            pad_token_id=tokenizer.pad_token_id or 0,
            use_cache=True,
            use_greedy_sampler=True,
            sparse_budget=budget if budget > 0 else None,
            chunk_size=chunk_size,
            drafter_max_pos=drafter_max_pos,
        )

        torch.cuda.synchronize()
        t1 = time.time()
        elapsed = t1 - t0
        acc = accepted / speculated if speculated > 0 else 0

        if i >= warmup:
            total_tokens += len(output)
            total_time += t1 - t0
            total_accepted += accepted
            total_speculated += speculated

        print(f" {elapsed:.1f}s  accept={acc:.3f}  gen={len(output)} tokens", flush=True)
        torch.cuda.empty_cache()

    throughput = total_tokens / total_time if total_time > 0 else 0
    accept_rate = total_accepted / total_speculated if total_speculated > 0 else 0
    return throughput, accept_rate, total_tokens, total_time


def main():
    parser = argparse.ArgumentParser(description="Evaluate TinyDraft")
    parser.add_argument("--target_model", type=str,
                        default="NousResearch/Yarn-Llama-2-7b-128k")
    parser.add_argument("--original_student", type=str,
                        default="JackFram/llama-68m",
                        help="Original (untrained) student for comparison")
    parser.add_argument("--trained_student", type=str, required=True,
                        help="Path to trained TinyDraft checkpoint")
    parser.add_argument("--dataset", type=str, default="gs",
                        choices=["gs", "longbench_packed_qmsum", "lwm", "dolly"])
    parser.add_argument("--context", type=str, default="short",
                        choices=["short", "long"])
    parser.add_argument("--gamma", type=int, default=3)
    parser.add_argument("--budgets", type=str, default="0,256,512,1024,2048,3800",
                        help="Comma-separated budgets (0=full cache)")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_samples", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--rope_scale_factor", type=float, default=None,
                        help="RoPE scaling factor for trained student (e.g., 64)")
    parser.add_argument("--rope_scale_type", type=str, default="linear",
                        choices=["linear", "dynamic"])
    parser.add_argument("--max_length", type=int, default=None,
                        help="Override max prompt length (default: 3800 for short, 124928 for long)")
    parser.add_argument("--output_csv", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda")
    if args.max_length is not None:
        max_length = args.max_length
    else:
        max_length = 124928 if args.context == "long" else 3800
    budgets = [int(b) for b in args.budgets.split(",")]

    print("=" * 60)
    print("  TinyDraft Evaluation")
    print("=" * 60)
    print(f"Target:           {args.target_model}")
    print(f"Original student: {args.original_student}")
    print(f"Trained student:  {args.trained_student}")
    print(f"Dataset:          {args.dataset} ({args.context})")
    print(f"Gamma:            {args.gamma}")
    print(f"Budgets:          {budgets}")
    print("=" * 60)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.original_student)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load target model
    print("\nLoading target model...")
    target = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        torch_dtype=torch.float16,
        device_map=device,
        attn_implementation="sdpa",
    ).eval()

    # Load prompts
    prompts = load_prompts(args.dataset, tokenizer, max_length, args.max_samples)
    print(f"Loaded {len(prompts)} prompts\n")

    results = []

    for student_label, student_path in [
        ("original", args.original_student),
        ("tinydraft", args.trained_student),
    ]:
        print(f"\n{'─'*40}")
        print(f"  Student: {student_label}")
        print(f"{'─'*40}")

        # Apply RoPE scaling for trained student at long context
        drafter_kwargs = dict(torch_dtype=torch.float32, device_map=device)
        if student_label == "tinydraft" and args.rope_scale_factor is not None:
            from transformers import AutoConfig
            drafter_config = AutoConfig.from_pretrained(student_path)
            original_max_pos = drafter_config.max_position_embeddings
            drafter_config.rope_scaling = {
                "type": args.rope_scale_type,
                "factor": args.rope_scale_factor,
            }
            drafter_config.max_position_embeddings = int(original_max_pos * args.rope_scale_factor)
            drafter_kwargs["config"] = drafter_config
            print(f"  RoPE scaling: factor={args.rope_scale_factor} max_pos={drafter_config.max_position_embeddings}")
        drafter = AutoModelForCausalLM.from_pretrained(
            student_path,
            **drafter_kwargs,
        ).eval()

        # Auto-detect drafter max_position_embeddings for position clamping
        # Apply when prompt exceeds drafter's native position range
        drafter_native_max_pos = getattr(drafter.config, 'max_position_embeddings', None)
        # If RoPE scaling was applied, native range is already extended
        if args.rope_scale_factor is not None and student_label == "tinydraft":
            drafter_native_max_pos = None  # Already extended, no clamping needed
        use_pos_clamp = (drafter_native_max_pos is not None and max_length > drafter_native_max_pos)
        if use_pos_clamp:
            print(f"  Position clamping: pos % {drafter_native_max_pos} (prompt {max_length} > max_pos {drafter_native_max_pos})")

        for budget in budgets:
            budget_label = "full" if budget == 0 else f"b{budget}"
            print(f"\n  [{student_label}] budget={budget_label} gamma={args.gamma}...")

            torch.cuda.reset_peak_memory_stats()
            throughput, accept_rate, total_tok, total_t = run_sd_eval(
                target, drafter, tokenizer, prompts,
                args.gamma, budget, args.chunk_size,
                args.max_new_tokens, args.warmup, device,
                drafter_max_pos=drafter_native_max_pos if use_pos_clamp else None,
            )
            peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

            print(f"    throughput={throughput:.2f} tok/s  "
                  f"accept={accept_rate:.3f}  "
                  f"peak={peak_mb:.0f}MB")

            results.append({
                "student": student_label,
                "dataset": args.dataset,
                "context": args.context,
                "budget": budget_label,
                "gamma": args.gamma,
                "throughput": f"{throughput:.2f}",
                "accept_rate": f"{accept_rate:.4f}",
                "total_tokens": total_tok,
                "total_time": f"{total_t:.2f}",
                "peak_gpu_mb": f"{peak_mb:.0f}",
            })

        del drafter
        torch.cuda.empty_cache()

    # Print summary
    print(f"\n{'='*60}")
    print("  Results Summary")
    print(f"{'='*60}")
    print(f"{'Student':<12} {'Budget':<8} {'Throughput':>12} {'Accept':>10}")
    print("-" * 44)
    for r in results:
        print(f"{r['student']:<12} {r['budget']:<8} {r['throughput']:>10} tok/s {r['accept_rate']:>10}")

    # Save CSV
    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {args.output_csv}")


if __name__ == "__main__":
    main()
