from __future__ import annotations
import numpy as np
from typing import Dict, Any
try:
    from glitchlab.core.registry import register
except Exception:
    from glitchlab.core.registry import register

DOC = "Soft glow: lifts highlights and adds mild saturation; mask/amp aware."
DEFAULTS: Dict[str, Any] = {
    "lift": 0.15,         # 0..1 - ile podnieść jasne partie
    "sat": 0.2,           # -1..1 - zmiana nasycenia
    "mask_key": None,
    "use_amp": 1.0,
    "clamp": True,
}

def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    # rgb, hsv in [0,1]
    r,g,b = rgb[...,0], rgb[...,1], rgb[...,2]
    mx = np.max(rgb, axis=-1)
    mn = np.min(rgb, axis=-1)
    d = mx - mn
    h = np.zeros_like(mx)
    mask = d != 0
    idx = (mx == r) & mask
    h[idx] = ((g[idx]-b[idx])/d[idx]) % 6.0
    idx = (mx == g) & mask
    h[idx] = (b[idx]-r[idx])/d[idx] + 2.0
    idx = (mx == b) & mask
    h[idx] = (r[idx]-g[idx])/d[idx] + 4.0
    h = (h/6.0) % 1.0
    s = np.where(mx == 0, 0, d/np.maximum(mx, 1e-6))
    v = mx
    return np.stack([h,s,v], axis=-1)

def _hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    h,s,v = hsv[...,0], hsv[...,1], hsv[...,2]
    h6 = (h*6.0) % 6.0
    i = np.floor(h6).astype(np.int32)
    f = h6 - i
    p = v*(1.0 - s)
    q = v*(1.0 - s*f)
    t = v*(1.0 - s*(1.0-f))
    out = np.zeros_like(hsv)
    cases = [
        (i==0, np.stack([v,t,p],-1)),
        (i==1, np.stack([q,v,p],-1)),
        (i==2, np.stack([p,v,t],-1)),
        (i==3, np.stack([p,q,v],-1)),
        (i==4, np.stack([t,p,v],-1)),
        (i==5, np.stack([v,p,q],-1)),
    ]
    for m,val in cases:
        out[m] = val[m]
    return out

@register("rgb_glow", defaults=DEFAULTS, doc=DOC)
def rgb_glow(img_u8: np.ndarray, ctx, **p) -> np.ndarray:
        lift = float(p.get("lift", 0.15))
        sat_delta = float(p.get("sat", 0.2))
        mask_key = p.get("mask_key", None)
        use_amp = p.get("use_amp", 1.0)
        clamp = bool(p.get("clamp", True))

        x = img_u8.astype(np.float32) / 255.0
        H, W = x.shape[:2]
        amp = getattr(ctx, "amplitude", None)
        if amp is None:
            amp_map = np.ones((H,W), np.float32)
        else:
            amp_map = np.clip(amp.astype(np.float32), 0.0, 1.0)
            if isinstance(use_amp, (int,float)):
                amp_map = np.clip(amp_map * float(use_amp), 0.0, 1.0)
            elif not use_amp:
                amp_map[:] = 1.0

        m = None
        if isinstance(mask_key, str):
            mm = ctx.masks.get(mask_key)
            if isinstance(mm, np.ndarray) and mm.shape[:2] == (H,W):
                m = np.clip(mm.astype(np.float32), 0.0, 1.0)
        w = amp_map if m is None else (amp_map * m)

        # lift highlights
        y = np.clip(x + lift * (x**2), 0.0, 1.0)
        # saturation tweak in HSV
        hsv = _rgb_to_hsv(y)
        hsv[...,1] = np.clip(hsv[...,1] + sat_delta, 0.0, 1.0)
        y2 = np.clip(_hsv_to_rgb(hsv), 0.0, 1.0)

        out = x*(1.0 - w[...,None]) + y2*(w[...,None])
        out = np.clip(out, 0.0, 1.0)
        out_u8 = (out*255.0 + 0.5).astype(np.uint8) if clamp else (out*255.0).astype(np.uint8)

        # diag
        ctx.cache["diag/rgb_glow/weight"] = (np.stack([w]*3, -1)*255+0.5).astype(np.uint8)
        return out_u8
