# glitchlab/filters/rgb_offset.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from typing import Any, Dict

try:
    from glitchlab.core.registry import register
except Exception:  # pragma: no cover
    from core.registry import register  # type: ignore

DOC = (
    "RGB channel offset/dispersion with bilinear subpixel shift. "
    "Per-channel dx,dy; optional global dx,dy addend; mask/amplitude/mix; wrap or edge clamp. "
    "Emits diag alpha."
)

DEFAULTS: Dict[str, Any] = {
    # per-channel (float; px)
    "dx_r": 2.0,  "dy_r": 0.0,
    "dx_g": 0.0,  "dy_g": 0.0,
    "dx_b": -2.0, "dy_b": 0.0,
    # optional global addends (float; px) — applied to all channels
    "dx": 0.0, "dy": 0.0,
    # mixing & edges
    "mix": 1.0,              # 0..1 – global mix toward shifted (modulated by mask*amp)
    "wrap": False,           # False: sample with edge clamp; True: wrap modulo
    # context maps
    "mask_key": None,
    "use_amp": 1.0,          # float|bool
    "clamp": True,           # final clip-to-u8
}

def _to_f32(img_u8: np.ndarray) -> np.ndarray:
    return img_u8.astype(np.float32) / 255.0

def _to_u8(img_f32: np.ndarray) -> np.ndarray:
    x = np.clip(img_f32, 0.0, 1.0)
    return (x * 255.0 + 0.5).astype(np.uint8)

def _fit_hw(m: np.ndarray, H: int, W: int) -> np.ndarray:
    """Nearest-like dopasowanie bez zależności: rozszerza krawędzią."""
    m = np.asarray(m, dtype=np.float32)
    mh, mw = m.shape[:2]
    out = np.zeros((H, W), dtype=np.float32)
    h = min(H, mh); w = min(W, mw)
    out[:h, :w] = m[:h, :w]
    if h > 0 and h < H: out[h:, :w] = out[h-1:h, :w]
    if w > 0 and w < W: out[:H, w:] = out[:H, w-1:w]
    if h < H and w < W: out[h:, w:] = out[h-1, w-1]
    return np.clip(out, 0.0, 1.0)

def _resolve_mask(ctx, mask_key: str | None, H: int, W: int) -> np.ndarray:
    if not mask_key:
        return np.ones((H, W), dtype=np.float32)
    m = None
    if getattr(ctx, "masks", None):
        m = ctx.masks.get(mask_key)
    if m is None:
        return np.ones((H, W), dtype=np.float32)
    m = np.asarray(m, dtype=np.float32)
    if m.shape[:2] != (H, W):
        m = _fit_hw(m, H, W)
    return np.clip(m, 0.0, 1.0)

def _amplitude_weight(ctx, H: int, W: int, use_amp) -> np.ndarray:
    if not hasattr(ctx, "amplitude") or ctx.amplitude is None:
        return np.ones((H, W), dtype=np.float32)
    amp = np.asarray(ctx.amplitude, dtype=np.float32)
    if amp.shape[:2] != (H, W):
        amp = _fit_hw(amp, H, W)
    # normalize and bias to avoid zeroing effect
    mn, mx = float(amp.min()), float(amp.max())
    if mx > mn:
        amp = (amp - mn) / (mx - mn)
    base = 0.25 + 0.75 * amp
    if isinstance(use_amp, bool):
        return base if use_amp else np.ones((H, W), dtype=np.float32)
    return base * float(max(0.0, use_amp))

def _bilinear_sample(ch: np.ndarray, x: np.ndarray, y: np.ndarray, wrap: bool) -> np.ndarray:
    """Sample ch[y,x] with float coords and bilinear; handles wrap or edge clamp."""
    H, W = ch.shape
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    fx = (x - x0).astype(np.float32)
    fy = (y - y0).astype(np.float32)

    if wrap:
        x0 %= W; x1 %= W
        y0 %= H; y1 %= H
    else:
        x0 = np.clip(x0, 0, W-1); x1 = np.clip(x1, 0, W-1)
        y0 = np.clip(y0, 0, H-1); y1 = np.clip(y1, 0, H-1)

    Ia = ch[y0, x0]; Ib = ch[y0, x1]
    Ic = ch[y1, x0]; Id = ch[y1, x1]
    return (Ia*(1-fx)*(1-fy) + Ib*fx*(1-fy) + Ic*(1-fx)*fy + Id*fx*fy).astype(np.float32)

def _shift_channel(ch: np.ndarray, dx: float, dy: float, wrap: bool) -> np.ndarray:
    H, W = ch.shape
    yy, xx = np.meshgrid(np.arange(H, dtype=np.float32),
                         np.arange(W, dtype=np.float32), indexing="ij")
    # sample source at (x-dx, y-dy)
    return _bilinear_sample(ch, xx - float(dx), yy - float(dy), wrap)

@register("rgb_offset", defaults=DEFAULTS, doc=DOC)
def rgb_offset(img: np.ndarray, ctx, **p) -> np.ndarray:
    """
    Per-channel subpixel offset (bilinear), plus optional global dx,dy addends.
    Alpha = mix * mask * amp; edge handling via wrap=True/False.
    """
    a = np.asarray(img)
    if a.ndim != 3 or a.shape[2] < 3:
        raise ValueError("rgb_offset: expected RGB-like image (H,W,C>=3)")
    H, W, _ = a.shape
    x = _to_f32(a)

    # params
    dx_add = float(p.get("dx", DEFAULTS["dx"]))
    dy_add = float(p.get("dy", DEFAULTS["dy"]))
    dx_r = float(p.get("dx_r", DEFAULTS["dx_r"])) + dx_add
    dy_r = float(p.get("dy_r", DEFAULTS["dy_r"])) + dy_add
    dx_g = float(p.get("dx_g", DEFAULTS["dx_g"])) + dx_add
    dy_g = float(p.get("dy_g", DEFAULTS["dy_g"])) + dy_add
    dx_b = float(p.get("dx_b", DEFAULTS["dx_b"])) + dx_add
    dy_b = float(p.get("dy_b", DEFAULTS["dy_b"])) + dy_add

    wrap  = bool(p.get("wrap", DEFAULTS["wrap"]))
    mix   = float(p.get("mix", DEFAULTS["mix"]))
    mkey  = p.get("mask_key", DEFAULTS["mask_key"])
    use_amp = p.get("use_amp", DEFAULTS["use_amp"])
    clamp = bool(p.get("clamp", DEFAULTS["clamp"]))

    # alpha
    m   = _resolve_mask(ctx, mkey, H, W)
    amp = _amplitude_weight(ctx, H, W, use_amp)
    alpha = np.clip(mix, 0.0, 1.0) * m * amp  # [H,W]

    # shift per channel
    r = _shift_channel(x[..., 0], dx_r, dy_r, wrap)
    g = _shift_channel(x[..., 1], dx_g, dy_g, wrap)
    b = _shift_channel(x[..., 2], dx_b, dy_b, wrap)
    shifted = np.stack([r, g, b], axis=-1)

    eff = x * (1.0 - alpha[..., None]) + shifted * (alpha[..., None])
    out = _to_u8(eff) if clamp else (np.clip(eff, 0.0, 1.0) * 255.0).astype(np.uint8)

    # diagnostics
    try:
        a3 = np.stack([alpha, alpha, alpha], axis=-1)
        ctx.cache["diag/rgb_offset/alpha"] = _to_u8(a3)
    except Exception:
        pass
    return out
