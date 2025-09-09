import numpy as np
from ..core.registry import register

@register("channel_shuffle")
def channel_shuffle(arr, ctx):
    """
    Losowo permutuje kanały R/G/B obrazu.
    """
    out = arr.copy()
    perm = np.random.permutation(3)
    out = out[..., perm]
    return out
