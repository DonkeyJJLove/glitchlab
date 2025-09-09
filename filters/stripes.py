import numpy as np
from ..core.registry import register
from ..core.utils import Ctx

@register("stripes")
def stripes(arr: np.ndarray, ctx: Ctx, bands: int=12, max_h: int=24, max_dx: int=60) -> np.ndarray:
    out = arr.copy()
    H,W,_ = arr.shape
    rng = ctx.rng
    for _ in range(bands):
        y = int(rng.integers(0, H-1))
        h = int(np.clip(np.abs(rng.normal(max_h*0.6, max_h*0.25)), 6, max_h))
        y2 = min(H, y+h)
        # scale shift by amplitude field average in band
        amp = float(ctx.amplitude[y:y2, :].mean()) if ctx.amplitude is not None else 1.0
        dx = int(rng.integers(-max_dx, max_dx+1) * amp)
        out[y:y2,:,:] = np.roll(out[y:y2,:,:], dx, axis=1)
    return out
