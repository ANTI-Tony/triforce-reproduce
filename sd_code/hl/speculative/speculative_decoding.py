import sys
import os
import time
# 获取当前文件的路径，并添加到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 添加上一级目录，确保 speculative 可以被导入
sys.path.append(os.path.join(current_dir, "speculative"))

import torch
from torch.nn import Module
from logits_processor import LogitsProcessor, GreedyProcessor, MultinomialProcessor
from transformers.cache_utils import DynamicCache, StaticCache
from sparse_cache import apply_triforce_sparse
import printing as printing
from typing import List, Tuple, Optional


def max_fn(x: torch.Tensor) -> torch.Tensor:
    x_max = torch.where(x > 0, x, torch.zeros_like(x))
    x_max_sum = torch.sum(x_max, dim=-1, keepdim=True)
    return x_max / x_max_sum


# ── Cache helpers (support transformers 4.x and 5.x DynamicCache) ──

def _get_layer_kv(cache, layer_idx):
    """Get (key, value) tensors for a layer. Works with both transformers versions."""
    if hasattr(cache, 'layers'):
        # transformers 5.x
        return cache.layers[layer_idx].keys, cache.layers[layer_idx].values
    elif hasattr(cache, 'key_cache'):
        # transformers 4.x
        return cache.key_cache[layer_idx], cache.value_cache[layer_idx]
    else:
        # tuple cache
        return cache[layer_idx][0], cache[layer_idx][1]


def _set_layer_kv(cache, layer_idx, key, value):
    """Set (key, value) tensors for a layer."""
    if hasattr(cache, 'layers'):
        cache.layers[layer_idx].keys = key
        cache.layers[layer_idx].values = value
    elif hasattr(cache, 'key_cache'):
        cache.key_cache[layer_idx] = key
        cache.value_cache[layer_idx] = value


def _num_layers(cache):
    if hasattr(cache, 'layers'):
        return len(cache.layers)
    elif hasattr(cache, 'key_cache'):
        return len(cache.key_cache)
    return len(cache)


def _cache_seq_len(cache):
    """Get actual cache sequence length from tensor shape (not _seen_tokens)."""
    k, _ = _get_layer_kv(cache, 0)
    return k.shape[-2]


def prune_cache(cache, num_tokens_to_discard):
    """Prune last N tokens from cache in-place."""
    if cache is None or num_tokens_to_discard == 0:
        return cache
    n_layers = _num_layers(cache)
    for i in range(n_layers):
        k, v = _get_layer_kv(cache, i)
        _set_layer_kv(cache, i, k[:, :, :-num_tokens_to_discard, :], v[:, :, :-num_tokens_to_discard, :])
    # Keep _seen_tokens in sync so model computes correct position_ids
    if hasattr(cache, '_seen_tokens'):
        cache._seen_tokens -= num_tokens_to_discard
    return cache


def cache_to_tuples(cache):
    """Extract (key, value) tuples from cache for sparse selection."""
    n = _num_layers(cache)
    return tuple((_get_layer_kv(cache, i)) for i in range(n))


def build_dynamic_cache(kv_tuples):
    """Build a new DynamicCache from list of (key, value) tuples."""
    cache = DynamicCache()
    for i, (k, v) in enumerate(kv_tuples):
        cache.update(k, v, i)
    return cache


@torch.no_grad()
def speculative_generate(
    inputs: List[int],
    drafter: Module,
    target: Module,
    tokenizer = None,
    gamma: int = 3,
    logits_processor: LogitsProcessor = MultinomialProcessor(0.7),
    max_gen_len: int = 40,
    eos_tokens_id: int | List[int] = 1,
    pad_token_id: int = 0,
    use_cache: bool = False,
    skip_sample_adjustment: bool = False,
    first_target: bool = True,
    debug: bool = False,
    use_greedy_sampler: bool = False,
    sparse_budget: Optional[int] = None,
    chunk_size: int = 8,
    drafter_max_pos: Optional[int] = None,
) -> Tuple[List[int], float]:
    """
    Speculative Decoding with optional sparse drafter KV cache.
    """

    drafter_cache, target_cache = None, None

    drafts_accepted, drafts_speculated = .0, .0
    vocabulary_size = target.config.vocab_size
    stop_tokens = eos_tokens_id

    # prepare input tensor
    prompt_len = len(inputs)
    total_len = prompt_len + max_gen_len
    input_ids = torch.full((1, total_len), pad_token_id, dtype=torch.long, device=target.device)
    input_ids[0, :prompt_len] = torch.tensor(inputs, dtype=torch.long, device=target.device)

    tot_token_nums = prompt_len

    # Whether we are using sparse drafter cache
    use_sparse = sparse_budget is not None and sparse_budget < prompt_len
    drafter_pos_offset = 0

    # Use StaticCache for target when prompt is long to avoid OOM
    use_static_target = prompt_len > 4096
    prefill_chunk = 1024

    if first_target:
        if use_static_target:
            # StaticCache + chunked prefill for long context
            target_cache = StaticCache(
                config=target.config,
                max_batch_size=1,
                max_cache_len=total_len + gamma + 2,
                device=target.device,
                dtype=torch.float16,
            )
            last_logits = None
            for start_pos in range(0, prompt_len, prefill_chunk):
                end_pos = min(start_pos + prefill_chunk, prompt_len)
                cache_positions = torch.arange(start_pos, end_pos, device=target.device)
                out = target(
                    input_ids=input_ids[:, start_pos:end_pos],
                    past_key_values=target_cache,
                    use_cache=True,
                    cache_position=cache_positions,
                )
                last_logits = out.logits[:, -1:, :]
                del out

            # Cast to float32 and sanitize NaN/Inf from fp16 overflow at long context
            prefill_logits = torch.nan_to_num(last_logits.squeeze(1).float(), nan=0.0, posinf=1e4, neginf=-1e4)
            if use_greedy_sampler:
                t = torch.argmax(prefill_logits, dim=-1)
            else:
                p_p = logits_processor(prefill_logits)
                t = logits_processor.sample(p_p)
            del last_logits, prefill_logits
        else:
            # DynamicCache + single-shot prefill for short context
            Mp = target(
                input_ids=input_ids[..., :tot_token_nums],
                past_key_values=target_cache,
                use_cache=use_cache,
            )
            target_cache = Mp.past_key_values

            if use_greedy_sampler:
                t = torch.argmax(Mp.logits[..., -1, :], dim=-1)
            else:
                p_p = logits_processor(Mp.logits[..., -1, :])
                t = logits_processor.sample(p_p)

        input_ids[0, tot_token_nums] = t
        tot_token_nums += 1

        if t == stop_tokens:
            if debug:
                printing.end_token_found(0)
            return input_ids[0, prompt_len:tot_token_nums].tolist(), 0, 0

        if debug:
            printing.initial_step(t, tokenizer)

        # Drafter prefill (chunked for long context to avoid activation OOM)
        # No position clamping here - let model use natural positions during prefill.
        # After sparsification, cache length = budget, so draft positions auto-start
        # from budget (within safe range).
        if use_static_target:
            for start_pos in range(0, prompt_len, prefill_chunk):
                end_pos = min(start_pos + prefill_chunk, prompt_len)
                Mq = drafter(
                    input_ids=input_ids[:, start_pos:end_pos],
                    past_key_values=drafter_cache,
                    use_cache=use_cache,
                )
                drafter_cache = Mq.past_key_values
                del Mq
        else:
            Mq = drafter(
                    input_ids=input_ids[..., :prompt_len],
                    past_key_values=drafter_cache,
                    use_cache=use_cache,
                )
            drafter_cache = Mq.past_key_values

        # Apply sparse selection to drafter cache
        if use_sparse:
            kv_tuples = cache_to_tuples(drafter_cache)
            sparse_tuples, original_len = apply_triforce_sparse(
                kv_tuples, sparse_budget, chunk_size
            )
            drafter_cache = build_dynamic_cache(sparse_tuples)
            sparse_cache_len = _cache_seq_len(drafter_cache)
            drafter_pos_offset = original_len - sparse_cache_len

    step = 0
    while tot_token_nums < total_len:
        step += 1
        corrected_gamma = min(gamma, total_len - tot_token_nums - 1)
        if not use_greedy_sampler:
            q = torch.zeros((corrected_gamma, vocabulary_size), device=target.device)

        # generate gamma drafts
        hl = []
        current_position = tot_token_nums - 1

        for k in range(corrected_gamma):
            # No explicit position_ids — model auto-computes from cache length.
            # After sparsification, cache length = budget, so positions stay in safe range.
            Mq = drafter(
                input_ids=input_ids[..., current_position + k: current_position + k + 1],
                past_key_values=drafter_cache,
                use_cache=use_cache,
            )
            drafter_cache = Mq.past_key_values

            if use_greedy_sampler:
                xi = t = torch.argmax(Mq.logits, dim = -1)
            else:
                draft_probs = logits_processor(Mq.logits)[0]
                xi = logits_processor.sample(draft_probs)
                q[k] = draft_probs

            input_ids[0, current_position + 1 + k] = xi
            hl.append(xi)

        drafts_speculated += corrected_gamma

        # run target model on drafts
        if use_static_target:
            cache_positions = torch.arange(
                current_position, current_position + corrected_gamma + 1,
                device=target.device
            )
            Mp = target(
                input_ids=input_ids[..., current_position: current_position + corrected_gamma + 1],
                past_key_values=target_cache,
                use_cache=True,
                cache_position=cache_positions,
            )
        else:
            # DynamicCache: auto-fix cache alignment
            cache_len = _cache_seq_len(target_cache)
            expected = tot_token_nums - 1
            if cache_len > expected:
                target_cache = prune_cache(target_cache, cache_len - expected)
            Mp = target(
                input_ids=input_ids[..., current_position: current_position + corrected_gamma + 1],
                past_key_values=target_cache,
                use_cache=use_cache,
            )
            target_cache = Mp.past_key_values
        target_logits = Mp.logits
        if use_static_target:
            target_logits = torch.nan_to_num(target_logits.float(), nan=0.0, posinf=1e4, neginf=-1e4)

        # rejection sampling
        n = corrected_gamma
        if use_greedy_sampler:
            target_token_ids = torch.argmax(target_logits, dim=-1)
            for i in range(corrected_gamma):
                if target_token_ids[0, i] != hl[i]:
                    n = i
                    break
                drafts_accepted += 1
        else:
            p = logits_processor(target_logits[0])
            r = torch.rand(corrected_gamma, device=target.device)
            p = p.float()
            fractions = p[:corrected_gamma, ...] / q
            for i in range(corrected_gamma):
                j = current_position + i + 1
                if r[i] > fractions[i, input_ids[0, j]]:
                    n = i
                    break
                drafts_accepted += 1
                if input_ids[0, j] == stop_tokens:
                    return input_ids[0, prompt_len:j + 1].tolist(), drafts_accepted, drafts_speculated

        # adjust distribution and sample replacement token
        if use_greedy_sampler:
            x = target_token_ids[0, n]
        else:
            if n == corrected_gamma:
                p_p = p[-1:, ...]
            else:
                if not skip_sample_adjustment:
                    p_p = max_fn(p[n:n+1, :] - q[n:n+1, :])
                else:
                    p_p = p[n, :]
            x = logits_processor.sample(p_p)

        if n != corrected_gamma:
            # prune the cache
            if use_cache:
                drafter_cache = prune_cache(drafter_cache, corrected_gamma - n - 1)
                if not use_static_target:
                    target_cache = prune_cache(target_cache, corrected_gamma - n)
        else:
            # all drafts accepted: supplement drafter cache
            Mq = drafter(
                input_ids=input_ids[..., current_position + corrected_gamma: current_position + corrected_gamma + 1],
                past_key_values=drafter_cache,
                use_cache=use_cache,
            )
            drafter_cache = Mq.past_key_values

        if debug:
            generated = input_ids.clone().detach()

        input_ids[0, current_position + 1 + n : current_position + 1 + corrected_gamma] = pad_token_id
        input_ids[0, current_position + 1 + n] = x

        if debug:
            printing.speculative_step(tokenizer, generated, input_ids, n, prompt_len, tot_token_nums, corrected_gamma)

        tot_token_nums += n + 1

        if x == stop_tokens:
            if debug:
                printing.end_token_found(n)
            return input_ids[0, prompt_len:tot_token_nums].tolist(), drafts_accepted, drafts_speculated

    return input_ids[0, prompt_len:].tolist(), drafts_accepted, drafts_speculated
