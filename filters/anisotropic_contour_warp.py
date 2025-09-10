# -*- coding: utf-8 -*-
"""
Anisotropic Contour Warp
========================
Popycha piksele *wzdłuż konturów* (tangencjalnie do gradientu),
dając wrażenie falowania/anizotropowego rozciągania. Wspiera maski i amplitude.

Parametry:
  strength   : float  – piksele na iterację
  ksize      : int    – rozmiar rdzenia Sobela (3 lub 5)
  iters      : int    – liczba kroków propagacji
  smooth     : float  – sigma Gaussa przed detekcją konturów (0 = brak)
  edge_bias  : float  – >0 wzmacnia krawędzie, <0 faworyzuje gładkie obszary
  mask_key   : str|None – ograniczenie działania do maski (0..1)
  use_amp    : float [0..1] – wpływ ctx.amplitude
  clamp      : bool   – przycięcie przemieszczeń do krawędzi (True) lub wrap
"""

from __future__ import annotations
import numpy as np

from glitchlab.core.registry import register  # <-- ważne
# brak innych zależności od rejestru

@register("anisotropic_contour_warp")
def anisotropic_contour_warp(
    img: np.ndarray,
    ctx,
    strength: float = 1.5,
    ksize: int = 3,
    iters: int = 1,
    smooth: float = 0.0,
    edge_bias: float = 0.0,
    mask_key: str | None = None,
    use_amp: float = 1.0,
    clamp: bool = True,
):
    a = np.asarray(img)
    if a.ndim != 3 or a.shape[2] < 3:
        raise ValueError("anisotropic_contour_warp: expected RGB-like image (H,W,C>=3)")
    H, W, C = a.shape
    work = a[..., :3].astype(np.float32) / 255.0

    # luminancja + opcjonalne wygładzenie
    gray = work.mean(axis=2)
    if smooth and smooth > 0:
        gray = _gauss_blur(gray, sigma=float(smooth))

    # gradienty Sobela
    kx, ky = _sobel_kernels(ksize)
    gx = _conv2(gray, kx)
    gy = _conv2(gray, ky)
    mag = np.sqrt(gx * gx + gy * gy) + 1e-12

    # styczna do konturu (rotacja ∇I o +90°)
    vx, vy = gx / mag, gy / mag
    tx, ty = -vy, vx

    # wagi: maska * bias krawędzi * amplitude
    w_mask = _resolve_mask(ctx, mask_key, H, W)
    w_edge = _edge_weight(mag, edge_bias)
    w_amp  = _amplitude_weight(ctx, H, W, use_amp)
    weight = w_mask * w_edge * w_amp
    if weight.max() <= 0:
        return a

    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    out = a.copy()
    step = float(max(0.0, strength))

    for _ in range(max(1, int(iters))):
        dx = (tx * step * weight).round().astype(np.int32)
        dy = (ty * step * weight).round().astype(np.int32)

        if clamp:
            x2 = np.clip(xx + dx, 0, W - 1)
            y2 = np.clip(yy + dy, 0, H - 1)
        else:
            x2 = (xx + dx) % W
            y2 = (yy + dy) % H

        for ch in range(3):
            out[..., ch] = out[y2, x2, ch]

    # diagnostyka do cache
    if getattr(ctx, "cache", None) is not None:
        ctx.cache["acw_mag"] = (mag / (mag.max() + 1e-12)).astype(np.float32)
        ctx.cache["acw_tx"]  = ((tx + 1.0) * 0.5).astype(np.float32)
        ctx.cache["acw_ty"]  = ((ty + 1.0) * 0.5).astype(np.float32)

    return out


# -------------------------- Pomocnicze --------------------------

def _sobel_kernels(ksize: int) -> tuple[np.ndarray, np.ndarray]:
    ksize = 5 if int(ksize) == 5 else 3
    if ksize == 3:
        kx = np.array([[ 1, 0,-1],
                       [ 2, 0,-2],
                       [ 1, 0,-1]], dtype=np.float32)
        ky = np.array([[ 1, 2, 1],
                       [ 0, 0, 0],
                       [-1,-2,-1]], dtype=np.float32)
    else:
        kx = np.array([[ 2, 1, 0,-1,-2],
                       [ 3, 2, 0,-2,-3],
                       [ 4, 3, 0,-3,-4],
                       [ 3, 2, 0,-2,-3],
                       [ 2, 1, 0,-1,-2]], dtype=np.float32)
        ky = kx.T
    return kx, ky


def _conv2(img: np.ndarray, k: np.ndarray) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)
    kh, kw = k.shape
    ph, pw = kh // 2, kw // 2
    pad = np.pad(img, ((ph, ph), (pw, pw)), mode="reflect")
    out = np.empty_like(img, dtype=np.float32)
    # prosto (czytelnie); można przyspieszyć im2col jeśli będzie potrzebne
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.sum(pad[i:i + kh, j:j + kw] * k)
    return out


def _gauss_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img
    radius = max(1, int(round(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    g = np.exp(-0.5 * (x / float(sigma)) ** 2); g /= g.sum()

    def _conv1(u: np.ndarray, kern: np.ndarray, axis: int) -> np.ndarray:
        pad = [(0, 0)] * u.ndim
        pad[axis] = (radius, radius)
        up = np.pad(u, pad, mode="reflect")
        out = np.empty_like(u, dtype=np.float32)
        if axis == 0:
            for i in range(u.shape[0]):
                out[i, :] = np.tensordot(up[i:i + 2 * radius + 1, :], kern, axes=(0, 0))
        else:
            for j in range(u.shape[1]):
                out[:, j] = np.tensordot(up[:, j:j + 2 * radius + 1], kern, axes=(1, 0))
        return out

    tmp = _conv1(img, g, axis=1)
    return _conv1(tmp, g, axis=0)


def _resolve_mask(ctx, mask_key: str | None, H: int, W: int) -> np.ndarray:
    m = None
    if mask_key and getattr(ctx, "masks", None):
        m = ctx.masks.get(mask_key)
    if m is None and getattr(ctx, "masks", None):
        m = ctx.masks.get("edge")
    if m is None:
        return np.ones((H, W), dtype=np.float32)
    m = np.asarray(m).astype(np.float32)
    if m.shape != (H, W):
        m = _fit_mask_hw(m, H, W)
    return np.clip(m, 0, 1)


def _fit_mask_hw(m: np.ndarray, H: int, W: int) -> np.ndarray:
    mh, mw = m.shape[:2]
    out = np.zeros((H, W), dtype=np.float32)
    h = min(H, mh); w = min(W, mw)
    out[:h, :w] = m[:h, :w].astype(np.float32)
    if h < H:
        out[h:, :w] = out[h - 1:h, :w]
    if w < W:
        out[:H, w:] = out[:H, w - 1:w]
    return out


def _edge_weight(mag: np.ndarray, edge_bias: float) -> np.ndarray:
    m = mag / (mag.max() + 1e-12)
    if abs(edge_bias) < 1e-12:
        return m.copy()
    if edge_bias > 0:
        return np.power(m, 1.0 / (1.0 + edge_bias))
    else:
        return np.power(1.0 - m, 1.0 / (1.0 + (-edge_bias)))


def _amplitude_weight(ctx, H: int, W: int, use_amp: float) -> np.ndarray:
    if use_amp <= 0 or not hasattr(ctx, "amplitude") or ctx.amplitude is None:
        return np.ones((H, W), dtype=np.float32)
    amp = np.asarray(ctx.amplitude).astype(np.float32)
    if amp.shape != (H, W):
        amp = _fit_mask_hw(amp, H, W)
    amp = amp - amp.min()
    amp = amp / (amp.max() + 1e-12)
    return (0.25 + 0.75 * amp) * float(np.clip(use_amp, 0.0, 1.0))
