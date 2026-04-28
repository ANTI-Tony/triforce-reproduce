#!/usr/bin/env python3
"""
TinyDraft Phase A Training
L = L_A + λ * L_C  (β=0, L_B disabled)

L_A: CE(student_full_cache_logits, teacher_argmax) over continuation positions
L_C: CE(student_sparse_cache_logits, teacher_argmax) over continuation positions
     with random budget sampled from {256,512,1024,2048,3800}

Training flow per step:
1. Sample text from PG-19, tokenize to seq_len
2. Split: prefix (seq_len - cont_len) + continuation (cont_len)
3. Teacher: chunked forward → argmax at continuation positions
4. Student: prefill prefix → shared cache
   - L_A: forward continuation with full prefix cache
   - L_C: sparsify cache → forward continuation with sparse cache
5. L = L_A + λ * L_C, backward, optimizer step
"""

import torch
import torch.nn.functional as F
import argparse
import os
import sys
import time
import random
import json
import math

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from datasets import load_dataset

# Import sparse cache
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, "speculative"))
from sparse_cache import apply_triforce_sparse

# ── Budget sampling config ──
BUDGETS = [256, 512, 1024, 2048]
BUDGET_WEIGHTS = [0.4, 0.3, 0.2, 0.1]


def cache_to_tuples(cache):
    """Extract (key, value) tuples from DynamicCache (preserves grad)."""
    if hasattr(cache, "key_cache"):
        return tuple(
            (cache.key_cache[i], cache.value_cache[i])
            for i in range(len(cache.key_cache))
        )
    return cache


def build_dynamic_cache(kv_tuples):
    """Build DynamicCache from (key, value) tuples (preserves grad)."""
    cache = DynamicCache()
    for i, (k, v) in enumerate(kv_tuples):
        cache.update(k, v, i)
    return cache


def sample_budget():
    return random.choices(BUDGETS, weights=BUDGET_WEIGHTS, k=1)[0]


# ── Data ──

def data_iterator(tokenizer, seq_len):
    """Yield tokenized chunks of seq_len from PG-19 train split (streaming)."""
    dataset = load_dataset("pg19", split="train", streaming=True, trust_remote_code=True)
    buffer = []
    for item in dataset:
        tokens = tokenizer.encode(item["text"], add_special_tokens=False)
        buffer.extend(tokens)
        while len(buffer) >= seq_len:
            yield torch.tensor(buffer[:seq_len], dtype=torch.long)
            buffer = buffer[seq_len:]


# ── Teacher targets ──

@torch.no_grad()
def get_teacher_targets(teacher, input_ids, prefix_len, cont_len, prefill_chunk=2048):
    """
    Get teacher's argmax predictions at continuation positions.
    Uses chunked prefill + continuation forward for memory efficiency.

    Returns: [1, cont_len - 1] tensor of target token IDs.
    """
    device = input_ids.device

    # Chunked prefill for prefix
    cache = None
    for start in range(0, prefix_len, prefill_chunk):
        end = min(start + prefill_chunk, prefix_len)
        out = teacher(input_ids[:, start:end], past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        del out

    # Continuation forward
    cont_ids = input_ids[:, prefix_len:]
    cont_out = teacher(cont_ids, past_key_values=cache, use_cache=True)
    targets = cont_out.logits[:, :-1, :].argmax(-1)  # [1, cont_len - 1]

    del cont_out, cache
    torch.cuda.empty_cache()
    return targets


@torch.no_grad()
def get_teacher_topk_logits(teacher, input_ids, prefix_len, cont_len, prefill_chunk=2048, topk=32):
    """
    Get teacher's top-k token indices and logits at continuation positions (for L_B).
    Returns: targets [1, cont_len-1], topk_indices [1, cont_len-1, k], topk_logits [1, cont_len-1, k],
             prefix_cache_tuples (for sparse reuse).
    """
    device = input_ids.device

    # Chunked prefill for prefix
    cache = None
    for start in range(0, prefix_len, prefill_chunk):
        end = min(start + prefill_chunk, prefix_len)
        out = teacher(input_ids[:, start:end], past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        del out

    # Save prefix cache for sparse teacher forward
    prefix_cache_tuples = cache_to_tuples(cache)

    # Continuation forward
    cont_ids = input_ids[:, prefix_len:]
    cont_out = teacher(cont_ids, past_key_values=cache, use_cache=True)
    full_logits = cont_out.logits[:, :-1, :]  # [1, cont_len-1, V]
    targets = full_logits.argmax(-1)  # [1, cont_len-1]

    # Get top-k indices and logits
    topk_logits, topk_indices = torch.topk(full_logits, topk, dim=-1)  # [1, cont_len-1, k]

    del cont_out, full_logits, cache
    torch.cuda.empty_cache()
    return targets, topk_indices, topk_logits, prefix_cache_tuples


@torch.no_grad()
def get_teacher_sparse_topk_logits(teacher, input_ids, prefix_cache_tuples, prefix_len, cont_len, budget, chunk_size, device, topk_indices):
    """
    Get teacher's logits under sparse KV cache at the same top-k positions (for L_B).
    Returns: [1, cont_len-1, k] tensor of logits at topk positions.
    """
    sparse_tuples, _ = apply_triforce_sparse(prefix_cache_tuples, budget, chunk_size)
    sparse_cache = build_dynamic_cache(sparse_tuples)

    cont_ids = input_ids[:, prefix_len:]
    cont_pos = torch.arange(prefix_len, prefix_len + cont_len, device=device).unsqueeze(0)
    cont_out = teacher(cont_ids, past_key_values=sparse_cache, position_ids=cont_pos)
    sparse_logits = cont_out.logits[:, :-1, :]  # [1, cont_len-1, V]

    # Gather logits at top-k positions
    sparse_topk_logits = torch.gather(sparse_logits, -1, topk_indices)  # [1, cont_len-1, k]

    del cont_out, sparse_cache, sparse_logits
    torch.cuda.empty_cache()
    return sparse_topk_logits


# ── Training step ──

def train_step(student, teacher, input_ids, prefix_len, cont_len, budget, chunk_size, lam, device, verbose=False, beta=0.0, topk=32, step=0, lb_every_n=2):
    """
    One training step.

    L = L_A + lam * L_C + beta * L_B
    L_A: CE(student, teacher_argmax) — top-1 acceptance
    L_C: same as L_A but with sparse cache — multi-view
    L_B: hinge top-k KL = max(0, KL_full - KL_sparse) on teacher's top-k tokens

    L_B only computed every lb_every_n steps to save cost.
    """
    V = student.config.vocab_size
    use_sparse = budget < prefix_len
    # L_B: only when beta>0, sparse, and on schedule (every N steps)
    use_lb = beta > 0 and use_sparse and (step % lb_every_n == 0)

    # 1. Teacher forward (no grad)
    if verbose:
        print(f"    [1/4] Teacher forward (L_B={'on' if use_lb else 'off'})...", flush=True)

    if use_lb:
        # Get top-1 targets + top-k logits + prefix cache for sparse reuse
        teacher_targets, topk_indices, teacher_full_topk, teacher_prefix_cache = get_teacher_topk_logits(
            teacher, input_ids, prefix_len, cont_len, prefill_chunk=1024, topk=topk
        )
        # Teacher sparse forward at same top-k positions
        teacher_sparse_topk = get_teacher_sparse_topk_logits(
            teacher, input_ids, teacher_prefix_cache, prefix_len, cont_len, budget, chunk_size, device, topk_indices
        )
        del teacher_prefix_cache
    else:
        teacher_targets = get_teacher_targets(teacher, input_ids, prefix_len, cont_len, prefill_chunk=1024)

    # 2. Student prefill prefix (no grad, chunked)
    if verbose:
        print("    [2/4] Student prefill (no grad)...", flush=True)
    student.eval()
    with torch.no_grad():
        prefix_cache = None
        for start in range(0, prefix_len, 2048):
            end = min(start + 2048, prefix_len)
            chunk_out = student(
                input_ids[:, start:end],
                past_key_values=prefix_cache,
                use_cache=True,
            )
            prefix_cache = chunk_out.past_key_values
            if isinstance(prefix_cache, tuple):
                prefix_cache = build_dynamic_cache(prefix_cache)
            del chunk_out
    student.train()

    # 3. Sparsify cache if needed, then continuation forward (WITH grad)
    cont_ids = input_ids[:, prefix_len:]

    if use_sparse:
        if verbose:
            print(f"    [3/4] Sparse forward (budget={budget})...", flush=True)
        kv_tuples = cache_to_tuples(prefix_cache)
        sparse_tuples, _ = apply_triforce_sparse(kv_tuples, budget, chunk_size)
        cont_cache = build_dynamic_cache(sparse_tuples)
        cont_pos = torch.arange(prefix_len, prefix_len + cont_len, device=device).unsqueeze(0)
        cont_out = student(cont_ids, past_key_values=cont_cache, position_ids=cont_pos)
        del prefix_cache
    else:
        if verbose:
            print("    [3/4] Full forward...", flush=True)
        cont_out = student(cont_ids, past_key_values=prefix_cache)

    # 4. Compute loss
    if verbose:
        print("    [4/4] Loss computation...", flush=True)

    student_logits = cont_out.logits[:, :-1, :]  # [1, cont_len-1, V]

    # L_A + L_C (CE with teacher argmax)
    loss_ce = F.cross_entropy(student_logits.reshape(-1, V), teacher_targets.reshape(-1))

    if use_lb:
        # Top-k KL: gather student logits at teacher's top-k positions
        student_topk_logits = torch.gather(student_logits, -1, topk_indices)  # [1, cont_len-1, k]

        # Normalize over k tokens (log_softmax)
        student_topk_lp = F.log_softmax(student_topk_logits.float(), dim=-1)
        teacher_full_topk_lp = F.log_softmax(teacher_full_topk.float(), dim=-1)
        teacher_sparse_topk_lp = F.log_softmax(teacher_sparse_topk.float(), dim=-1)

        # KL on k tokens only — much smaller scale
        kl_full = F.kl_div(student_topk_lp, teacher_full_topk_lp, reduction='batchmean', log_target=True)
        kl_sparse = F.kl_div(student_topk_lp, teacher_sparse_topk_lp, reduction='batchmean', log_target=True)

        # Hinge: max(0, KL_full - KL_sparse + m), m=0
        loss_b = torch.clamp(kl_full - kl_sparse, min=0.0)
        loss = loss_ce + beta * loss_b

        if verbose:
            print(f"    L_CE={loss_ce.item():.4f}, L_B={loss_b.item():.4f} (KL_full={kl_full.item():.4f}, KL_sparse={kl_sparse.item():.4f})", flush=True)

        del teacher_full_topk, teacher_sparse_topk, topk_indices
        del student_topk_logits, student_topk_lp, teacher_full_topk_lp, teacher_sparse_topk_lp
    else:
        loss = loss_ce

    if verbose:
        print(f"    loss={loss.item():.4f} (sparse={use_sparse})", flush=True)

    del cont_out, student_logits, teacher_targets
    return loss.item(), loss


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="TinyDraft Phase A Training")
    parser.add_argument("--teacher_model", type=str,
                        default="NousResearch/Yarn-Llama-2-7b-128k")
    parser.add_argument("--student_model", type=str,
                        default="JackFram/llama-68m")
    parser.add_argument("--seq_len", type=int, default=16384,
                        help="Total sequence length (prefix + continuation)")
    parser.add_argument("--cont_len", type=int, default=256,
                        help="Continuation length for loss computation")
    parser.add_argument("--chunk_size", type=int, default=8,
                        help="Chunk size for sparse cache selection")
    parser.add_argument("--lam", type=float, default=0.5,
                        help="Weight for L_C (sparse loss)")
    parser.add_argument("--beta", type=float, default=0.0,
                        help="Weight for L_B (shift-aware alignment, max after warmup)")
    parser.add_argument("--lb_every_n", type=int, default=2,
                        help="Compute L_B every N steps (saves ~50%% teacher sparse cost)")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=150,
                        help="Linear warmup steps (~3%% of 5000)")
    parser.add_argument("--total_steps", type=int, default=5000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--output_dir", type=str,
                        default="checkpoints/tinydraft_phase_a")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing for student")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Resume student from checkpoint directory")
    parser.add_argument("--rope_scale_factor", type=float, default=None,
                        help="RoPE scaling factor for long context (e.g., 64 for 128K)")
    parser.add_argument("--rope_scale_type", type=str, default="linear",
                        choices=["linear", "dynamic"],
                        help="RoPE scaling type")
    parser.add_argument("--full_cache_only", action="store_true",
                        help="Disable sparse training (L_A only, no L_C)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ── Setup ──
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prefix_len = args.seq_len - args.cont_len

    print("=" * 60)
    print("  TinyDraft Phase A Training")
    print("=" * 60)
    print(f"Teacher:    {args.teacher_model}")
    print(f"Student:    {args.student_model}")
    print(f"seq_len:    {args.seq_len}  (prefix={prefix_len}, cont={args.cont_len})")
    if args.full_cache_only:
        print(f"Loss:       L = L_A only (full cache, no sparse training)")
        print(f"Budgets:    full cache (budget={prefix_len})")
    else:
        if args.beta > 0:
            print(f"Loss:       L = L_A + {args.lam} * L_C + {args.beta} * L_B  (hinge top-k, every {args.lb_every_n} steps)")
        else:
            print(f"Loss:       L = L_A + {args.lam} * L_C  (β=0, L_B off)")
        print(f"Budgets:    {BUDGETS}  weights={BUDGET_WEIGHTS}")
    print(f"Optimizer:  AdamW lr={args.lr} wd={args.weight_decay}")
    print(f"Schedule:   warmup={args.warmup_steps} → cosine → 0")
    print(f"Steps:      {args.total_steps}")
    print(f"Output:     {args.output_dir}")
    print("=" * 60)
    print()

    # ── Load tokenizer ──
    tokenizer = AutoTokenizer.from_pretrained(args.student_model)

    # ── Load teacher (frozen, fp16) ──
    print("Loading teacher model (frozen, fp16)...")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        torch_dtype=torch.float16,
        device_map=device,
        attn_implementation="sdpa",
    ).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"  Teacher loaded: {sum(p.numel() for p in teacher.parameters())/1e6:.0f}M params")

    # ── Load student (trainable, fp32) ──
    student_path = args.resume_from if args.resume_from else args.student_model
    print(f"Loading student model from {student_path} (trainable, fp32)...")

    # Apply RoPE scaling for long context support
    student_kwargs = dict(
        torch_dtype=torch.float32,
        device_map=device,
        attn_implementation="sdpa",
    )
    if args.rope_scale_factor is not None:
        from transformers import AutoConfig
        student_config = AutoConfig.from_pretrained(student_path)
        original_max_pos = student_config.max_position_embeddings  # 2048
        student_config.rope_scaling = {
            "type": args.rope_scale_type,
            "factor": args.rope_scale_factor,
        }
        student_config.max_position_embeddings = int(original_max_pos * args.rope_scale_factor)
        student_kwargs["config"] = student_config
        print(f"  RoPE scaling: type={args.rope_scale_type} factor={args.rope_scale_factor}")
        print(f"  max_position_embeddings: {original_max_pos} → {student_config.max_position_embeddings}")

    student = AutoModelForCausalLM.from_pretrained(
        student_path,
        **student_kwargs,
    )
    student.config.use_cache = True  # Ensure cache is returned in training mode
    student.train()
    if args.gradient_checkpointing:
        student.gradient_checkpointing_enable()
        print("  Gradient checkpointing enabled")
        print("  WARNING: gradient checkpointing disables use_cache, L_C will use workaround")
    print(f"  Student loaded: {sum(p.numel() for p in student.parameters())/1e6:.1f}M params")

    # ── Optimizer + scheduler ──
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, args.total_steps - args.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Output dir + log ──
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "training_log.jsonl")
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"  Config saved: {config_path}")

    # ── Data iterator ──
    print("\nStarting PG-19 streaming data iterator...")
    data_iter = data_iterator(tokenizer, args.seq_len)

    # ── Training loop ──
    print(f"\n{'='*60}")
    print("  Training started")
    print(f"{'='*60}\n")

    running_total = 0.0
    start_time = time.time()

    for step in range(1, args.total_steps + 1):
        # Get next sample
        if step == 1:
            print("  Loading first PG-19 sample (may download data)...", flush=True)
        try:
            tokens = next(data_iter)
        except StopIteration:
            print("  [Data] Restarting PG-19 iterator")
            data_iter = data_iterator(tokenizer, args.seq_len)
            tokens = next(data_iter)
        if step == 1:
            print(f"  Got sample: {tokens.shape[0]} tokens", flush=True)

        input_ids = tokens.unsqueeze(0).to(device)  # [1, seq_len]
        budget = prefix_len if args.full_cache_only else sample_budget()

        # Forward + loss
        optimizer.zero_grad(set_to_none=True)

        try:
            is_verbose = (step <= 3)  # Verbose for first 3 steps
            # β warmup: first 10% steps β=0, then linear to beta_max over 90%
            warmup_frac = 0.1
            if step <= int(args.total_steps * warmup_frac):
                beta_now = 0.0
            else:
                progress = (step - int(args.total_steps * warmup_frac)) / max(1, args.total_steps * (1 - warmup_frac))
                beta_now = args.beta * progress

            loss_val, loss_tensor = train_step(
                student, teacher, input_ids,
                prefix_len, args.cont_len,
                budget, args.chunk_size, args.lam, device,
                verbose=is_verbose,
                beta=beta_now, topk=32, step=step, lb_every_n=args.lb_every_n,
            )
        except torch.cuda.OutOfMemoryError:
            print(f"  [Step {step}] OOM with budget={budget}, skipping", flush=True)
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            continue

        # Check for NaN
        if torch.isnan(loss_tensor) or torch.isinf(loss_tensor):
            print(f"  [Step {step}] NaN/Inf loss, skipping", flush=True)
            optimizer.zero_grad(set_to_none=True)
            del loss_tensor
            torch.cuda.empty_cache()
            continue

        # Backward
        if is_verbose:
            print("    backward...", flush=True)
        loss_tensor.backward()
        if is_verbose:
            print("    optimizer step...", flush=True)
        torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        # Accumulate for logging
        running_total += loss_val

        del input_ids, loss_tensor
        torch.cuda.empty_cache()

        # ── Log ──
        # Print every step for first 20, then every log_interval
        if step <= 20 or step % args.log_interval == 0:
            n = 1 if step <= 20 else args.log_interval
            avg_loss = running_total / n
            elapsed = time.time() - start_time
            lr_now = scheduler.get_last_lr()[0]
            peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

            print(
                f"[{step:5d}/{args.total_steps}] "
                f"loss={avg_loss:.4f}  "
                f"b={budget:4d}  lr={lr_now:.2e}  "
                f"peak={peak_mb:.0f}MB  {elapsed/step:.1f}s/step",
                flush=True,
            )

            log_entry = {
                "step": step,
                "loss": round(avg_loss, 6),
                "budget": budget,
                "lr": lr_now,
                "peak_gpu_mb": round(peak_mb, 0),
                "elapsed_s": round(elapsed, 1),
                "s_per_step": round(elapsed / step, 2),
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            running_total = 0.0

        # ── Checkpoint ──
        if step % args.save_interval == 0:
            ckpt_dir = os.path.join(args.output_dir, f"step_{step}")
            student.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            print(f"  → Checkpoint: {ckpt_dir}")

    # ── Final save ──
    final_dir = os.path.join(args.output_dir, "final")
    student.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  {args.total_steps} steps in {total_time/3600:.1f}h ({total_time/args.total_steps:.1f}s/step)")
    print(f"  Final model: {final_dir}")
    print(f"  Log: {log_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
