import numpy as np
from ..core.registry import register
from ..core.utils import Ctx

@register("rgb_offset")
def rgb_offset(arr: np.ndarray, ctx: Ctx, r=(1,0), g=(-1,0), b=(2,0)) -> np.ndarray:
    out = arr.copy()
    for c,(dx,dy) in enumerate([r,g,b]):
        ch = out[...,c]
        ch = np.roll(ch, dy, axis=0)
        ch = np.roll(ch, dx, axis=1)
        out[...,c] = ch
    return out
