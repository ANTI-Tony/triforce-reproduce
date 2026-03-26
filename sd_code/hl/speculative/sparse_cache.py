"""
TriForce-style sparse KV cache selection for HuggingFace models.

Implements attention-based chunk selection from TriForce (arXiv:2404.11912)
to sparsify a drafter model's KV cache in standard Speculative Decoding.

Uses the last key vector as query proxy for chunk scoring (avoids
extra forward passes or model-internal hooks).

NOTE: All caches are expected as tuple format: tuple of (key, value) pairs.
"""

import torch
from typing import Tuple


def _select_chunks(key, value, query, budget, chunk_size):
    """Per-head attention-based chunk selection for one layer.

    Args:
        key:    [bsz, heads, seq_len, head_dim]
        value:  [bsz, heads, seq_len, head_dim]
        query:  [bsz, heads, 1, head_dim]  (last-token key as proxy)
        budget: tokens to keep (rounded down to chunk_size multiple)
        chunk_size: tokens per chunk

    Returns:
        (sparse_key, sparse_value) each [bsz, heads, budget, head_dim]
    """
    bsz, num_heads, seq_len, head_dim = key.shape
    num_chunks = seq_len // chunk_size
    select_sets = min(budget // chunk_size, num_chunks)
    actual_budget = select_sets * chunk_size

    if select_sets <= 0:
        select_sets = 1
        actual_budget = chunk_size

    usable_len = num_chunks * chunk_size

    # [bsz, heads, chunks, chunk_size, dim]
    k_chunked = key[:, :, :usable_len].reshape(bsz, num_heads, num_chunks, chunk_size, head_dim)
    v_chunked = value[:, :, :usable_len].reshape(bsz, num_heads, num_chunks, chunk_size, head_dim)

    # Chunk averages: [bsz, heads, chunks, dim]
    chunk_avg = k_chunked.mean(dim=3)

    # Attention scores: [bsz, heads, chunks]
    scores = torch.matmul(query, chunk_avg.transpose(-1, -2)).squeeze(2)

    # Always keep chunk 0; select rest by score
    if select_sets > 1 and num_chunks > 1:
        _, topk_rest = torch.topk(scores[:, :, 1:], k=select_sets - 1, dim=-1)
        topk_rest += 1
        chunk0 = torch.zeros(bsz, num_heads, 1, device=key.device, dtype=torch.long)
        topk = torch.cat([chunk0, topk_rest], dim=-1)
    else:
        topk = torch.zeros(bsz, num_heads, select_sets, device=key.device, dtype=torch.long)

    # Gather: expand indices to [bsz, heads, select_sets, chunk_size, dim]
    idx = topk.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, chunk_size, head_dim)

    sparse_k = torch.gather(k_chunked, 2, idx).reshape(bsz, num_heads, actual_budget, head_dim)
    sparse_v = torch.gather(v_chunked, 2, idx).reshape(bsz, num_heads, actual_budget, head_dim)

    return sparse_k, sparse_v


def apply_triforce_sparse(
    full_cache: tuple,
    budget: int,
    chunk_size: int = 8,
) -> Tuple[tuple, int]:
    """Apply TriForce sparse selection to a full KV cache (tuple format).

    Uses the last key vector per layer as query proxy for chunk scoring.

    Args:
        full_cache: tuple of (key, value) pairs from drafter prefill
        budget: number of KV pairs to keep
        chunk_size: chunk granularity

    Returns:
        (sparse_cache, original_seq_len)
    """
    num_layers = len(full_cache)
    k0 = full_cache[0][0]
    original_seq_len = k0.shape[2]

    if budget >= original_seq_len:
        return full_cache, original_seq_len

    # Align budget to chunk_size
    eff_budget = (budget // chunk_size) * chunk_size
    if eff_budget == 0:
        eff_budget = chunk_size

    layers = []
    for i in range(num_layers):
        k, v = full_cache[i][0], full_cache[i][1]
        q_proxy = k[:, :, -1:, :]
        sk, sv = _select_chunks(k, v, q_proxy, eff_budget, chunk_size)
        layers.append((sk, sv))

    return tuple(layers), original_seq_len
