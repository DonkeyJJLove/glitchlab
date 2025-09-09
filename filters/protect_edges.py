import numpy as np
from ..core.registry import register
from ..core.utils import Ctx

@register("protect_edges")
def protect_edges(arr: np.ndarray, ctx: Ctx, keep: float=0.93) -> np.ndarray:
    """Blend back original image on detected edge/text areas."""
    if "original" not in ctx.meta:
        return arr
    orig = ctx.meta["original"]
    mask = ctx.masks.get("edges", None)
    if mask is None:
        return arr
    m = mask[...,None]
    return np.where(m==1, (keep*orig + (1-keep)*arr).astype(np.int16), arr)
