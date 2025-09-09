import numpy as np
from ..core.registry import register
from ..core.utils import Ctx


@register("amp_mul")
def amp_mul(arr: np.ndarray, ctx: Ctx, factor: float = 1.0):
    if ctx.amplitude is None: return arr
    ctx.amplitude = ctx.amplitude * factor;
    return arr


@register("amp_mask_mul")
def amp_mask_mul(arr: np.ndarray, ctx: Ctx, mask_key: str, factor: float = 1.5):
    m = ctx.masks.get(mask_key, None)
    if m is None: return arr
    if ctx.amplitude is None:
        ctx.amplitude = m.astype(np.float32)
    else:
        ctx.amplitude = ctx.amplitude * (1.0 + (factor - 1.0) * m.astype(np.float32))
    return arr
