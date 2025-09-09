import numpy as np
from ..core.registry import register

@register("protect_edges")
def protect_edges(arr, ctx, keep=0.95):
    m = ctx.masks.get("edges")
    if m is None:
        return arr

    orig = ctx.meta.get("original")
    if orig is None:
        return arr

    # Upewnij się, że oryginał i arr mają tyle samo kanałów
    if orig.shape[-1] == 4 and arr.shape[-1] == 3:
        orig = orig[..., :3]  # RGBA -> RGB
    elif orig.shape[-1] == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]    # RGBA -> RGB

    return np.where(
        m[..., None] == 1,
        (keep * orig + (1 - keep) * arr).astype(np.int16),
        arr
    )
