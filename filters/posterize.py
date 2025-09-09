import numpy as np
from ..core.registry import register
from ..core.utils import Ctx

@register("posterize")
def posterize(arr: np.ndarray, ctx: Ctx, bits: int=5) -> np.ndarray:
    step = 2**(8-bits)
    out = arr.copy()
    out[...,:3] = (out[...,:3] // step) * step
    return out
