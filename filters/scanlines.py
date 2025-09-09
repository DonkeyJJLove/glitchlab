import numpy as np
from ..core.registry import register
from ..core.utils import Ctx

@register("scanlines")
def scanlines(arr: np.ndarray, ctx: Ctx, period: int=2, strength: float=0.15) -> np.ndarray:
    out = arr.copy()
    out[::period, :, :3] = (out[::period, :, :3] * (1.0 - strength)).astype(np.int16)
    return out
