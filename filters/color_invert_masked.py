import numpy as np
from ..core.registry import register

@register("color_invert_masked")
def color_invert_masked(arr, ctx, mask_key=None):
    """
    Odwraca kolory (inwersja) tylko w obszarach określonych przez maskę.
    Jeśli maska nie istnieje → odwraca cały obraz.
    """
    if mask_key is None or mask_key not in ctx.masks:
        return 255 - arr  # pełna inwersja

    mask = ctx.masks[mask_key][..., None]  # [h, w, 1]
    out = arr.copy()
    out[mask > 0.5] = 255 - out[mask > 0.5]
    return out
