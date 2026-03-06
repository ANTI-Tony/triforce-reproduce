from typing import Tuple
from torch import Tensor


def prune_cache(cache: Tuple[Tuple[Tensor, Tensor]], num_tokens_to_discard: int):
    """
    Prune the cache by removing the specified number of tokens from the end.
    Cache is always tuple of (key, value) pairs.
    """
    if cache is None:
        return None
    if num_tokens_to_discard == 0:
        return cache

    new_cache = []
    for layer_cache in cache:
        if layer_cache is None:
            new_cache.append(None)
            continue
        layer = []
        for tensor in layer_cache:
            layer.append(tensor[:, :, :-num_tokens_to_discard, :])
        new_cache.append(tuple(layer))

    return tuple(new_cache)
