# glitchlab/filters/phase_glitch.py
# -*- coding: utf-8 -*-
"""
phase_glitch — zabawa fazą w domenie częstotliwości (z zachowaniem amplitudy).

Parametry:
    low        : float [0..1]    (dolna granica promienia pasma)
    high       : float [0..1]
    strength   : float [0..1]    (0=bez zmian, 1=losowa faza)
    preserve_dc: bool            (zachowaj obszar bardzo niskich częstotliwości)
    blend      : float [0..1]    (miks z oryginałem)
    mask_key   : str|None        (miks w dziedzinie obrazu wg maski)

Diagnostyka:
    ctx.cache["phase_glitch/noise"] — zastosowana mapa szumu fazy (downsample).
"""

from __future__ import annotations
import numpy as np
from glitchlab.core.registry import register
from glitchlab.core.utils import resize_mask_to

def _fft2c(x):
    return np.fft.fftshift(np.fft.fft2(x))

def _ifft2c(X):
    return np.fft.ifft2(np.fft.ifftshift(X))

@register("phase_glitch", schema={
    "low": {"type": "float", "min": 0.0, "max": 1.0, "step": 0.01},
    "high": {"type": "float", "min": 0.0, "max": 1.0, "step": 0.01},
    "strength": {"type": "float", "min": 0.0, "max": 1.0, "step": 0.01},
    "preserve_dc": {"type": "bool"},
    "blend": {"type": "float", "min": 0.0, "max": 1.0, "step": 0.05},
})
def phase_glitch(
    img, ctx,
    low: float = 0.18,
    high: float = 0.6,
    strength: float = 0.6,
    preserve_dc: bool = True,
    blend: float = 0.0,
    mask_key: str | None = None,
):
    a = img.astype(np.uint8, copy=False)
    h, w, _ = a.shape
    out = np.empty_like(a)

    yy, xx = np.meshgrid(np.arange(h) - h/2.0, np.arange(w) - w/2.0, indexing="ij")
    r = np.sqrt(xx*xx + yy*yy)
    rmax = 0.5 * min(h, w)
    rn = r / (rmax + 1e-8)

    rng = getattr(ctx, "rng", np.random.default_rng(7))

    noise_map = None
    for ch in range(3):
        X = _fft2c(a[..., ch].astype(np.float32))
        mag = np.abs(X)
        ph = np.angle(X)

        band = ((rn >= float(low)) & (rn <= float(high))).astype(np.float32)
        if preserve_dc:
            band[rn < 0.03] = 0.0

        # losowy szum fazy w paśmie
        noise = (rng.random(size=band.shape).astype(np.float32) - 0.5) * 2.0 * np.pi
        dphi = strength * noise * band
        ph2 = ph + dphi

        Y = mag * np.exp(1j * ph2)
        y = np.real(_ifft2c(Y))
        out[..., ch] = np.clip(y, 0, 255)

        if noise_map is None:
            noise_map = dphi

    # miks globalny
    if 0.0 < float(blend) < 1.0:
        out = np.clip(out.astype(np.float32) * (1.0 - blend) + a.astype(np.float32) * blend, 0, 255).astype(np.uint8)

    # miks maską przestrzenną
    if isinstance(mask_key, str) and hasattr(ctx, "masks") and mask_key in ctx.masks:
        m = ctx.masks[mask_key]
        if m.shape != (h, w):
            m = resize_mask_to(m, (h, w))
        m3 = m[..., None]
        out = (out.astype(np.float32) * m3 + a.astype(np.float32) * (1.0 - m3)).astype(np.uint8)

    # diagnostyka
    try:
        H, W = noise_map.shape
        s0 = max(1, H // 256); s1 = max(1, W // 256)
        vis = (np.mod(noise_map + np.pi, 2*np.pi) / (2*np.pi) * 255).astype(np.uint8)
        ctx.cache["phase_glitch/noise"] = vis[::s0, ::s1]
    except Exception:
        pass

    return out.astype(np.uint8)
