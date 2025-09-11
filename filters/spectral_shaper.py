# glitchlab/filters/spectral_shaper.py
# -*- coding: utf-8 -*-
"""
spectral_shaper — kształtowanie widma (FFT) przez maski radialne/kierunkowe.

Tworzy kontrolowany "błąd widmowy": podbija/tłumi pasma (ring, band, directional).
Zapis diagnostyczny (uint8, downsample):
  - ctx.cache['spectral_shaper/mag']   : log|F| (≈256×256)
  - ctx.cache['spectral_shaper/mask']  : użyta maska częstotliwościowa

Parametry:
    mode        : 'bandpass' | 'bandstop' | 'ring' | 'direction'
    low         : float [0..1]      (dolna granica radialna)
    high        : float [0..1]      (górna granica radialna)
    angle_deg   : float             (dla 'direction': kąt osi pasma, 0=poziomo)
    ang_width   : float [0..180]    (szerokość kątowa pasma, w stopniach)
    boost       : float             (współczynnik modyfikacji, np. -1.0..+3.0)
    soft        : float             (piórko krawędzi maski, 0..1)
    blend       : float [0..1]      (miks z oryginałem w dziedzinie obrazu)
    mask_key    : str|None          (jeśli podany, miksuje wynik wg maski w obrazie)
"""

from __future__ import annotations
import numpy as np
from typing import Any, Dict

try:
    from glitchlab.core.registry import register
except Exception:  # pragma: no cover
    from core.registry import register  # type: ignore

# opcjonalny utils; zapewniamy fallback, żeby nie „skipowało” modułu
try:
    from glitchlab.core.utils import resize_mask_to as _resize_mask_to  # type: ignore
except Exception:
    _resize_mask_to = None  # fallback poniżej

DOC = "Kształtowanie widma (FFT): ring/band/direction z miękkimi krawędziami; miks i maskowanie przestrzenne."
DEFAULTS: Dict[str, Any] = {
    "mode": "ring",
    "low": 0.15,
    "high": 0.45,
    "angle_deg": 0.0,
    "ang_width": 20.0,
    "boost": 0.8,
    "soft": 0.08,
    "blend": 0.0,
    "mask_key": None,
}

def _fft2c(x: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft2(x))

def _ifft2c(X: np.ndarray) -> np.ndarray:
    return np.fft.ifft2(np.fft.ifftshift(X))

def _radial_angular_grids(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.meshgrid(np.arange(h) - h/2.0, np.arange(w) - w/2.0, indexing="ij")
    r = np.sqrt(xx*xx + yy*yy).astype(np.float32)
    phi = np.arctan2(yy, xx).astype(np.float32)  # [-pi,pi]
    rmax = 0.5 * min(h, w)
    rn = (r / (rmax + 1e-8)).astype(np.float32)  # ~0..1.4
    return rn, phi

def _soft_step(x: np.ndarray, edge: float, feather: float) -> np.ndarray:
    if feather <= 0:
        return (x >= edge).astype(np.float32)
    k = 1.0 / max(1e-6, feather)
    return (1.0 / (1.0 + np.exp(-(x - float(edge)) * k))).astype(np.float32)

def _build_mask(mode: str, rn: np.ndarray, phi: np.ndarray,
                low: float, high: float, angle_deg: float, ang_width: float, soft: float) -> np.ndarray:
    low  = np.clip(float(low),  0.0, 1.5)
    high = np.clip(float(high), 0.0, 1.5)
    soft = float(soft)
    if mode == "direction":
        ang = np.deg2rad(float(angle_deg))
        dphi = np.abs((phi - ang + np.pi) % (2*np.pi) - np.pi)
        width = np.deg2rad(max(1e-3, float(ang_width)))
        Mdir = 1.0 - _soft_step(dphi, width, soft)
        inner = _soft_step(rn, low, soft)
        outer = 1.0 - _soft_step(rn, high, soft)
        return (Mdir * inner * outer).astype(np.float32)
    else:
        inner = _soft_step(rn, low, soft)
        outer = 1.0 - _soft_step(rn, high, soft)
        return (inner * outer).astype(np.float32)

def _downsample_approx(x: np.ndarray, target: int = 256) -> np.ndarray:
    H, W = x.shape[:2]
    s0 = max(1, H // target); s1 = max(1, W // target)
    return x[::s0, ::s1]

def _fit_mask_nn(m: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """Prosty fallback NN bez zależności: szybki i wystarczający do miksu."""
    H, W = hw
    m = np.asarray(m, dtype=np.float32)
    if m.shape[:2] == (H, W):
        return np.clip(m, 0.0, 1.0)
    mh, mw = m.shape[:2]
    ys = np.linspace(0, mh - 1, H).astype(np.int32)
    xs = np.linspace(0, mw - 1, W).astype(np.int32)
    out = m[ys][:, xs]
    return np.clip(out, 0.0, 1.0)

@register("spectral_shaper", defaults=DEFAULTS, doc=DOC)
def spectral_shaper(img: np.ndarray, ctx, **p: Any) -> np.ndarray:
    a = img.astype(np.uint8, copy=False)
    H, W, C = a.shape
    if C < 3:
        raise ValueError("spectral_shaper: expected RGB-like image (H,W,C>=3)")

    mode      = str(p.get("mode", DEFAULTS["mode"])).lower()
    low       = float(p.get("low", DEFAULTS["low"]))
    high      = float(p.get("high", DEFAULTS["high"]))
    angle_deg = float(p.get("angle_deg", DEFAULTS["angle_deg"]))
    ang_width = float(p.get("ang_width", DEFAULTS["ang_width"]))
    boost     = float(p.get("boost", DEFAULTS["boost"]))
    soft      = float(p.get("soft", DEFAULTS["soft"]))
    blend     = float(np.clip(p.get("blend", DEFAULTS["blend"]), 0.0, 1.0))
    mask_key  = p.get("mask_key", DEFAULTS["mask_key"])

    rn, phi = _radial_angular_grids(H, W)
    M = _build_mask(mode, rn, phi, low, high, angle_deg, ang_width, soft)  # 0..1

    out = np.empty_like(a)
    last_mag = None

    for ch in range(3):
        X = _fft2c(a[..., ch].astype(np.float32))
        mag   = np.abs(X)
        phase = np.angle(X)

        if last_mag is None:
            last_mag = np.log1p(mag)

        if mode == "bandstop":
            # clamp boost do [-1,1] by nie odwracać widma
            new_mag = mag * (1.0 - np.clip(boost, -1.0, 1.0) * M)
        else:
            new_mag = mag * (1.0 + boost * M)

        Y = new_mag * np.exp(1j * phase)
        y = np.real(_ifft2c(Y))
        out[..., ch] = np.clip(y, 0, 255)

    # miks globalny w domenie obrazu
    if 0.0 < blend < 1.0:
        out = (out.astype(np.float32) * (1.0 - blend) + a.astype(np.float32) * blend).clip(0, 255).astype(np.uint8)

    # miks maską przestrzenną (fallback bez resize_mask_to)
    if isinstance(mask_key, str) and getattr(ctx, "masks", None) and mask_key in ctx.masks:
        m = ctx.masks[mask_key]
        if m.shape[:2] != (H, W):
            m = _resize_mask_to(m, (H, W)) if _resize_mask_to is not None else _fit_mask_nn(m, (H, W))
        m = np.clip(m.astype(np.float32), 0.0, 1.0)
        out = (out.astype(np.float32) * m[..., None] + a.astype(np.float32) * (1.0 - m[..., None])).clip(0, 255).astype(np.uint8)

    # diagnostyka do HUD
    try:
        mag_vis  = _downsample_approx(last_mag)
        mag_vis  = (255.0 * (mag_vis / (mag_vis.max() + 1e-8))).astype(np.uint8)
        mask_vis = (255.0 * _downsample_approx(M)).astype(np.uint8)
        ctx.cache["spectral_shaper/mag"]  = mag_vis
        ctx.cache["spectral_shaper/mask"] = mask_vis
    except Exception:
        pass

    return out.astype(np.uint8)
