import numpy as np
from ..core.registry import register
from ..core.utils import Ctx

@register("block_mosh")
def block_mosh(arr: np.ndarray, ctx: Ctx, size: int=24, p: float=0.33, max_shift: int=8) -> np.ndarray:
    out = arr.copy()
    H,W,_ = arr.shape
    rng = ctx.rng
    for by in range(0, H, size):
        for bx in range(0, W, size):
            if rng.random() < p:
                dy = int(rng.integers(-max_shift, max_shift+1))
                dx = int(rng.integers(-max_shift, max_shift+1))
                y2 = min(H, by+size)
                x2 = min(W, bx+size)
                # scale by amplitude field patch-average
                amp = float(ctx.amplitude[by:y2, bx:x2].mean()) if ctx.amplitude is not None else 1.0
                dy = int(dy * amp); dx = int(dx * amp)
                out[by:y2, bx:x2, :] = np.roll(np.roll(out[by:y2, bx:x2, :], dy, axis=0), dx, axis=1)
    return out
