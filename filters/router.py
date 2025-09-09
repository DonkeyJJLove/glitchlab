
import numpy as np
from typing import List, Dict, Any
from ..core.registry import register, get
from ..core.utils import Ctx
def _run_steps(arr: np.ndarray, ctx: Ctx, steps: List[Dict[str,Any]]) -> np.ndarray:
    out = arr
    for step in steps:
        fn = get(step["name"]); params = step.get("params", {})
        out = fn(out, ctx, **params)
    return out
@register("with_mask")
def with_mask(arr: np.ndarray, ctx: Ctx, mask_key: str, steps: List[Dict[str,Any]], keep_outside: float=1.0) -> np.ndarray:
    m = ctx.masks.get(mask_key, None)
    if m is None: return arr
    m3 = m[...,None]
    sub = _run_steps(arr.copy(), ctx, steps)
    out = np.where(m3==1, sub, (keep_outside*arr + (1.0-keep_outside)*sub).astype(np.int16))
    return out
