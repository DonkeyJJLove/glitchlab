# glitchlab/filters/anisotropic_contour_warp.py
# -*- coding: utf-8 -*-
"""
Anisotropic Contour Warp
========================
Filtr „pcha” piksele *wzdłuż konturów* (tangencjalnie do gradientu),
dając wrażenie falowania/anizotropowego rozciągania. Wspiera maski i amplitude.

Wejście:
  - img : (H,W,C) uint8/float
  - ctx : kontekst glitchlab (masks, amplitude, cache, seed...)

Parametry:
  strength   : float, piksele na iterację (domyślnie 1.5)
  ksize      : int, rozmiar rdzenia Sobela (3 lub 5)
  iters      : int, liczba kroków propagacji
  smooth     : float, sigma Gaussa przed detekcją konturów (0 = brak)
  edge_bias  : float, [opc] >0 wzmacnia obszary o dużym gradiencie,
                         <0 faworyzuje „gładkie” fragmenty
  mask_key   : str|None, jeżeli podane — ogranicza działanie do tej maski (0..1)
  use_amp    : float in [0..1], w jakim stopniu uwzględniać ctx.amplitude (mnożnik)
  clamp      : bool, czy przyciąć mapę przemieszczeń do krawędzi obrazu

Diagnostyka (zapisywana do ctx.cache):
  - acw_mag : znormalizowana mapa |∇I|
  - acw_tx  : składowa X wektora stycznego
  - acw_ty  : składowa Y wektora stycznego
"""

from __future__ import annotations
import numpy as np
from glitchlab.core.registry import register


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
    # --- sanity & typy ---
    a = np.asarray(img)
    if a.ndim != 3 or a.shape[2] < 3:
        raise ValueError("anisotropic_contour_warp: expected RGB-like image (H,W,C>=3)")
    H, W, C = a.shape
    work = a[..., :3].astype(np.float32) / 255.0

    # --- przygotowanie luminancji / smoothing ---
    gray = work.mean(axis=2)
    if smooth and smooth > 0:
        gray = _gauss_blur(gray, sigma=float(smooth))

    # --- gradienty (Sobel) ---
    kx, ky = _sobel_kernels(ksize)
    gx = _conv2(gray, kx)
    gy = _conv2(gray, ky)

    mag = np.sqrt(gx * gx + gy * gy) + 1e-12  # unikaj dzielenia przez 0
    # wektor styczny do konturu (rotacja +90°): t = (-vy, vx), gdzie v = ∇I/|∇I|
    vx, vy = gx / mag, gy / mag
    tx, ty = -vy, vx  # komponenty styczne

    # --- wagi: maska + bias krawędzi + amplitude ---
    w_mask = _resolve_mask(ctx, mask_key, H, W)
    w_edge = _edge_weight(mag, edge_bias)
    w_amp  = _amplitude_weight(ctx, H, W, use_amp)

    weight = w_mask * w_edge * w_amp  # finalna mapa [0..1]
    if weight.max() <= 0:
        return a  # nic do zrobienia

    # --- pętle przesunięć ---
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
            # „zawijanie” zamiast clamp
            x2 = (xx + dx) % W
            y2 = (yy + dy) % H

        for ch in range(3):
            out[..., ch] = out[y2, x2, ch]

    # --- diagnostyka do cache ---
    if getattr(ctx, "cache", None) is not None:
        ctx.cache["acw_mag"] = (mag / (mag.max() + 1e-12)).astype(np.float32)
        ctx.cache["acw_tx"]  = ((tx + 1.0) * 0.5).astype(np.float32)
        ctx.cache["acw_ty"]  = ((ty + 1.0) * 0.5).astype(np.float32)

    return out


# -------------------------- Pomocnicze funkcje --------------------------

def _sobel_kernels(ksize: int) -> tuple[np.ndarray, np.ndarray]:
    ksize = 5 if int(ksize) == 5 else 3
    if ksize == 3:
        kx = np.array([[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]], dtype=np.float32)
        ky = np.array([[1, 2, 1],
                       [0, 0, 0],
                       [-1, -2, -1]], dtype=np.float32)
    else:  # 5x5 Sobel (łagodniejszy)
        kx = np.array([[2, 1, 0, -1, -2],
                       [3, 2, 0, -2, -3],
                       [4, 3, 0, -3, -4],
                       [3, 2, 0, -2, -3],
                       [2, 1, 0, -1, -2]], dtype=np.float32)
        ky = kx.T
    return kx, ky


def _conv2(img: np.ndarray, k: np.ndarray) -> np.ndarray:
    """2D konwolucja z odbiciem brzegów (numpy only)."""
    img = np.asarray(img, dtype=np.float32)
    kh, kw = k.shape
    ph, pw = kh // 2, kw // 2
    pad = np.pad(img, ((ph, ph), (pw, pw)), mode="reflect")
    out = np.empty_like(img, dtype=np.float32)
    # macierzowy sposób: im2col/podglądy byłby szybszy, tu prosty loop dla czytelności
    for i in range(out.shape[0]):
        pp = pad[i:i + kh, :]
        # użyj 1D konwolucji w poziomie i pionie? Zostajemy przy pełnym 2D:
        for j in range(out.shape[1]):
            out[i, j] = np.sum(pp[:, j:j + kw] * k)
    return out


def _gauss_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Prosty Gauss separowalny (numpy only)."""
    if sigma <= 0:
        return img
    radius = max(1, int(round(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    g = np.exp(-0.5 * (x / float(sigma)) ** 2)
    g /= g.sum()

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

    img = np.asarray(img, dtype=np.float32)
    tmp = _conv1(img, g, axis=1)
    out = _conv1(tmp, g, axis=0)
    return out


def _resolve_mask(ctx, mask_key: str | None, H: int, W: int) -> np.ndarray:
    """Zwróć maskę [0..1] (z ctx.masks[mask_key] / edge / 1)."""
    m = None
    if mask_key and getattr(ctx, "masks", None):
        m = ctx.masks.get(mask_key)
    if m is None and getattr(ctx, "masks", None):
        m = ctx.masks.get("edge")
    if m is None:
        return np.ones((H, W), dtype=np.float32)
    m = np.asarray(m).astype(np.float32)
    if m.shape != (H, W):
        # spróbuj prostej korekty wymiarów przez crop/resize (tu: crop lub pad)
        m = _fit_mask_hw(m, H, W)
    m = np.clip(m, 0, 1)
    return m


def _fit_mask_hw(m: np.ndarray, H: int, W: int) -> np.ndarray:
    """Dopasuj 2D maskę do (H,W) przez proste wycięcie/padding (bez interpolacji)."""
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
    """Waga na bazie modułu gradientu (0..1), z biasem."""
    m = mag / (mag.max() + 1e-12)
    if abs(edge_bias) < 1e-12:
        return m * 1.0 + 0.0  # kopia
    if edge_bias > 0:
        return np.power(m, 1.0 / (1.0 + edge_bias))  # rozjaśnij krawędzie
    else:
        return np.power(1.0 - m, 1.0 / (1.0 + (-edge_bias)))  # rozjaśnij obszary gładkie


def _amplitude_weight(ctx, H: int, W: int, use_amp: float) -> np.ndarray:
    """Waga z ctx.amplitude (0..1); use_amp=0 wyłącza, 1 — pełny wpływ."""
    if use_amp <= 0 or not hasattr(ctx, "amplitude") or ctx.amplitude is None:
        return np.ones((H, W), dtype=np.float32)
    amp = np.asarray(ctx.amplitude).astype(np.float32)
    if amp.shape != (H, W):
        amp = _fit_mask_hw(amp, H, W)
    # normalizacja i lekkie podbicie, by 0 nie zamrażał całkiem
    amp = amp - amp.min()
    d = amp.max() + 1e-12
    amp = amp / d
    return (0.25 + 0.75 * amp) * float(np.clip(use_amp, 0.0, 1.0))
