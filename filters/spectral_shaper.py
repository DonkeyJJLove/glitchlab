# glitchlab/filters/spectral_shaper.py
# -*- coding: utf-8 -*-
"""
spectral_shaper — kształtowanie widma (FFT) przez maski radialne/kierunkowe.

Tworzy kontrolowany "błąd widmowy": podbija/tłumi pasma (ring, band, directional).
Dodatkowo zapisuje do ctx.cache:
  - 'spectral_shaper/mag'  : ostatnia mapa |F| (downsample 256x256, uint8)
  - 'spectral_shaper/mask' : użyta maska w dziedzinie częstotliwości (uint8)

Parametry:
    mode        : 'bandpass' | 'bandstop' | 'ring' | 'direction'
    low         : float [0..1]    (dolna granica radialna, jako frakcja Nyquista)
    high        : float [0..1]    (górna granica radialna)
    angle_deg   : float           (dla 'direction': kąt osi pasma, 0=poziomo)
    ang_width   : float [0..180]  (szerokość kątowa pasma w stopniach)
    boost       : float           (współczynnik modyfikacji, np. -0.8..+3.0)
    soft        : float           (piórko krawędzi maski, 0..1)
    blend       : float [0..1]    (miks z oryginałem w dziedzinie obrazu)
    mask_key    : str|None        (jeśli podany, miksuje wynik w przestrzeni obrazu)

Uwaga:
 - 'low'/'high' jako frakcja promienia Nyquista: 1.0 ≈ min(H,W)/2.
 - 'boost' > 0 podbija, < 0 tłumi. Efektywnie: |F|' = |F| * (1 + boost*M).
"""

from __future__ import annotations
import numpy as np
from glitchlab.core.registry import register
from glitchlab.core.utils import resize_mask_to

def _fft2c(x):
    return np.fft.fftshift(np.fft.fft2(x))

def _ifft2c(X):
    return np.fft.ifft2(np.fft.ifftshift(X))

def _radial_angular_grids(h, w):
    yy, xx = np.meshgrid(np.arange(h) - h/2.0, np.arange(w) - w/2.0, indexing="ij")
    r = np.sqrt(xx*xx + yy*yy)
    phi = np.arctan2(yy, xx)  # [-pi,pi]
    rmax = 0.5 * min(h, w)
    rn = (r / (rmax + 1e-8)).astype(np.float32)  # 0..~1.4
    return rn, phi

def _soft_step(x, edge, feather):
    # płynne przejście 0..1; feather w [0..1] → grubość przejścia
    if feather <= 0:
        return (x >= edge).astype(np.float32)
    k = 1.0 / max(1e-6, feather)
    return (1.0 / (1.0 + np.exp(-(x - edge) * k))).astype(np.float32)

def _build_mask(mode, rn, phi, low, high, angle_deg, ang_width, soft):
    low = np.clip(float(low), 0.0, 1.5)
    high = np.clip(float(high), 0.0, 1.5)
    soft = float(soft)

    if mode == "ring":
        # wieniec wokół (low..high) z miękkimi krawędziami
        inner = _soft_step(rn, low, soft)
        outer = 1.0 - _soft_step(rn, high, soft)
        M = inner * outer

    elif mode in ("bandpass", "bandstop"):
        inner = _soft_step(rn, low, soft)
        outer = 1.0 - _soft_step(rn, high, soft)
        ring = inner * outer
        M = ring

    elif mode == "direction":
        # pas kątowy ± ang_width wokół angle_deg
        ang = np.deg2rad(float(angle_deg))
        # minimalna odległość kątowa
        dphi = np.abs((phi - ang + np.pi) % (2*np.pi) - np.pi)
        width = np.deg2rad(max(1e-3, float(ang_width)))
        M = 1.0 - _soft_step(dphi, width, soft)
        # (opcjonalnie można też ograniczyć radialnie przez [low,high])
        inner = _soft_step(rn, low, soft)
        outer = 1.0 - _soft_step(rn, high, soft)
        M = M * inner * outer

    else:
        # domyślnie zwykły ring band
        inner = _soft_step(rn, low, soft)
        outer = 1.0 - _soft_step(rn, high, soft)
        M = inner * outer

    return M.astype(np.float32)

@register("spectral_shaper")
def spectral_shaper(
    img, ctx,
    mode: str = "ring",
    low: float = 0.15,
    high: float = 0.45,
    angle_deg: float = 0.0,
    ang_width: float = 20.0,
    boost: float = 0.8,
    soft: float = 0.08,
    blend: float = 0.0,
    mask_key: str | None = None,
):
    """
    Kształtowanie widma przez maskę M (0..1):  |F|' = |F| * (1 + boost*M).
    """
    a = img.astype(np.uint8, copy=False)
    h, w, c = a.shape
    out = np.empty_like(a)

    rn, phi = _radial_angular_grids(h, w)
    M = _build_mask(mode, rn, phi, low, high, angle_deg, ang_width, soft)  # 0..1

    # Diagnostyka (downsample do 256x256)
    def _down8(x):
        H, W = x.shape
        s0 = max(1, H // 256); s1 = max(1, W // 256)
        return x[::s0, ::s1]

    last_mag = None
    for ch in range(3):
        X = _fft2c(a[..., ch].astype(np.float32))
        mag = np.abs(X)
        phase = np.angle(X)

        if last_mag is None:
            last_mag = np.log1p(mag)

        if mode == "bandstop":
            new_mag = mag * (1.0 - np.clip(boost, -1.0, 1.0) * M)
        else:
            new_mag = mag * (1.0 + float(boost) * M)

        Y = new_mag * np.exp(1j * phase)
        y = np.real(_ifft2c(Y))
        out[..., ch] = np.clip(y, 0, 255)

    # opcjonalny miks w przestrzeni obrazu
    if 0.0 < float(blend) < 1.0:
        out = np.clip(out.astype(np.float32) * (1.0 - blend) + a.astype(np.float32) * blend, 0, 255).astype(np.uint8)

    # opcjonalne mieszanie maską przestrzenną
    if isinstance(mask_key, str) and hasattr(ctx, "masks") and mask_key in ctx.masks:
        m = ctx.masks[mask_key]
        if m.shape != (h, w):
            m = resize_mask_to(m, (h, w))
        m3 = m[..., None]
        out = (out.astype(np.float32) * m3 + a.astype(np.float32) * (1.0 - m3)).astype(np.uint8)

    # zapisz diagnostykę
    try:
        ctx.cache["spectral_shaper/mag"] = (255 * (_down8(last_mag) / (last_mag.max() + 1e-8))).astype(np.uint8)
        ctx.cache["spectral_shaper/mask"] = (255 * _down8(M)).astype(np.uint8)
    except Exception:
        pass

    return out.astype(np.uint8)
