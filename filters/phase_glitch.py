# glitchlab/filters/phase_glitch.py
# -*- coding: utf-8 -*-
"""
phase_glitch — zabawa fazą w domenie częstotliwości (z zachowaniem amplitudy).

Parametry:
    low        : float [0..1]    (dolna granica promienia pasma)
    high       : float [0..1]    (górna granica)
    strength   : float [0..1]    (0=bez zmian, 1=losowa faza w paśmie)
    preserve_dc: bool            (wytnij bardzo niskie częstotliwości z pasma)
    blend      : float [0..1]    (miks z oryginałem w przestrzeni obrazu)
    mask_key   : str|None        (miks maską w przestrzeni obrazu)

Diag (HUD):
    ctx.cache["phase_glitch/noise"] — znormalizowana mapa zakłócenia fazy (downsample).
"""

from __future__ import annotations
import numpy as np
from typing import Any, Dict, Optional

try:
    from glitchlab.core.registry import register
except Exception:  # pragma: no cover
    from core.registry import register  # type: ignore

DOC = "Randomizacja fazy w paśmie promieniowym częstotliwości; opcjonalny miks i maska."
DEFAULTS: Dict[str, Any] = {
    "low": 0.18,
    "high": 0.60,
    "strength": 0.60,
    "preserve_dc": True,
    "blend": 0.00,
    "mask_key": None,
}

def _fft2c(x: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft2(x))

def _ifft2c(X: np.ndarray) -> np.ndarray:
    return np.fft.ifft2(np.fft.ifftshift(X))

def _fit_hw(m: np.ndarray, H: int, W: int) -> np.ndarray:
    """Prosty nearest 'rozciągacz' bez zależności od PIL/Skimage (bez antyaliasingu)."""
    m = np.asarray(m, dtype=np.float32)
    mh, mw = m.shape[:2]
    if (mh, mw) == (H, W):
        return np.clip(m, 0.0, 1.0)
    out = np.zeros((H, W), dtype=np.float32)
    ys = (np.linspace(0, mh - 1, H)).astype(np.int32)
    xs = (np.linspace(0, mw - 1, W)).astype(np.int32)
    out[:] = m[ys][:, xs]
    return np.clip(out, 0.0, 1.0)

@register("phase_glitch", defaults=DEFAULTS, doc=DOC)
def phase_glitch(
    img: np.ndarray, ctx,
    **p: Any
) -> np.ndarray:
    a = np.asarray(img)
    if a.ndim != 3 or a.shape[2] < 3:
        raise ValueError("phase_glitch: expected RGB-like image (H,W,C>=3)")
    H, W, _ = a.shape

    low        = float(np.clip(p.get("low",        DEFAULTS["low"]),  0.0, 1.0))
    high       = float(np.clip(p.get("high",       DEFAULTS["high"]), 0.0, 1.0))
    strength   = float(np.clip(p.get("strength",   DEFAULTS["strength"]), 0.0, 1.0))
    preserve_dc= bool(p.get("preserve_dc", DEFAULTS["preserve_dc"]))
    blend      = float(np.clip(p.get("blend",      DEFAULTS["blend"]), 0.0, 1.0))
    mask_key   = p.get("mask_key", DEFAULTS["mask_key"])

    # porządek pasma
    if high < low:
        low, high = high, low

    # siatka promieni
    yy, xx = np.meshgrid(np.arange(H) - H/2.0, np.arange(W) - W/2.0, indexing="ij")
    r = np.sqrt(xx*xx + yy*yy)
    rmax = 0.5 * float(min(H, W))
    rn = r / (rmax + 1e-8)

    rng = getattr(ctx, "rng", np.random.default_rng(int(getattr(ctx, "seed", 7))))
    out = np.empty_like(a)
    noise_map: Optional[np.ndarray] = None

    band = ((rn >= low) & (rn <= high)).astype(np.float32)
    if preserve_dc:
        band[rn < 0.03] = 0.0

    # dla powtarzalności – jedna mapa szumu fazy
    base_noise = (rng.random(size=band.shape).astype(np.float32) - 0.5) * (2.0 * np.pi)
    dphi_base = strength * base_noise * band

    for ch in range(3):
        X = _fft2c(a[..., ch].astype(np.float32))
        mag = np.abs(X)
        ph  = np.angle(X)

        dphi = dphi_base  # ten sam rozkład na kanały (spójność)
        Y = mag * np.exp(1j * (ph + dphi))
        y = np.real(_ifft2c(Y))
        out[..., ch] = np.clip(y, 0, 255)

        if noise_map is None:
            noise_map = dphi

    # miks globalny (image-space)
    if 0.0 < blend < 1.0:
        out = np.clip(out.astype(np.float32) * (1.0 - blend) + a.astype(np.float32) * blend, 0, 255).astype(np.uint8)

    # miks maską przestrzenną
    if isinstance(mask_key, str) and getattr(ctx, "masks", None) and (mask_key in ctx.masks):
        m = ctx.masks[mask_key]
        if m.shape != (H, W):
            m = _fit_hw(m, H, W)
        m = np.clip(m, 0.0, 1.0).astype(np.float32)[..., None]
        out = (out.astype(np.float32) * m + a.astype(np.float32) * (1.0 - m)).astype(np.uint8)

    # diagnostyka HUD
    try:
        if noise_map is not None:
            vis = (np.mod(noise_map + np.pi, 2*np.pi) / (2*np.pi))  # 0..1
            s0 = max(1, H // 256); s1 = max(1, W // 256)
            u8 = (np.clip(vis[::s0, ::s1], 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
            ctx.cache["phase_glitch/noise"] = u8  # grayscale
    except Exception:
        pass

    return out.astype(np.uint8)
