import numpy as np
from ..core.registry import register
from ..core.utils import Ctx

@register("noise")
def noise(arr: np.ndarray, ctx: Ctx, amount: int=8) -> np.ndarray:
    n = ctx.rng.integers(-amount, amount+1, size=arr[...,:3].shape)
    out = arr.copy()
    out[...,:3] = out[...,:3] + n
    return out
