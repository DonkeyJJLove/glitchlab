# -*- coding: utf-8 -*-
"""
Anisotropic Contour Warp (ACW)
------------------------------
Popycha piksele wzdłuż konturów (tangencjalnie do ∇I), dając wrażenie falowania.
Wspiera maski ROI i amplitude. Zapisuje diagnostyki do HUD: diag/acw/mag, /tx, /ty.

Parametry:
  strength   : float  – piksele na iterację (>=0)
  ksize      : int    – rdzeń Sobela: 3 lub 5
  iters      : int    – liczba iteracji przemieszczenia (>=1)
  smooth     : float  – sigma Gaussa przed Sobelem (0 = brak)
  edge_bias  : float  – >0 wzmacnia krawędzie, <0 faworyzuje gładkie obszary
  mask_key   : str|None – ograniczenie działania (0..1)
  use_amp    : float|bool – wpływ ctx.amplitude (0..2 typowo; bool => 1.0/0.0)
  clamp      : bool   – True: przycinanie do granic, False: wrap (modulo)
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any

try:
    from glitchlab.core.registry import register
except Exception:  # pragma: no cover
    from core.registry import register  # type: ignore

DOC = "Anisotropic contour warp (tangent push). Tangencjalne przesunięcia względem ∇I; maski i amplitude."
DEFAULTS: Dict[str, Any] = {
    "strength": 1.5,
    "ksize": 3,  # 3|5
    "iters": 1,
    "smooth": 0.0,
    "edge_bias": 0.0,  # >0: krawędzie, <0: obszary gładkie
    "mask_key": None,
    "use_amp": 1.0,  # float|bool
    "clamp": True,
}


@register("anisotropic_contour_warp", defaults=DEFAULTS, doc=DOC)
def anisotropic_contour_warp(
        img: np.ndarray,
        ctx,
        **p,
) -> np.ndarray:
    a = np.asarray(img)
    if a.ndim != 3 or a.shape[2] < 3:
        raise ValueError("anisotropic_contour_warp: expected RGB-like image (H,W,C>=3)")
    H, W, _ = a.shape

    # ---- paramy ----
    strength = float(p.get("strength", DEFAULTS["strength"]))
    ksize = int(p.get("ksize", DEFAULTS["ksize"]))
    iters = max(1, int(p.get("iters", DEFAULTS["iters"])))
    smooth = float(p.get("smooth", DEFAULTS["smooth"]))
    edge_bias = float(p.get("edge_bias", DEFAULTS["edge_bias"]))
    mask_key = p.get("mask_key", DEFAULTS["mask_key"])
    use_amp = p.get("use_amp", DEFAULTS["use_amp"])
    clamp = bool(p.get("clamp", DEFAULTS["clamp"]))

    # ---- luminancja + opcjonalne wygładzenie ----
    work = a[..., :3].astype(np.float32) / 255.0
    gray = (0.299 * work[..., 0] + 0.587 * work[..., 1] + 0.114 * work[..., 2]).astype(np.float32)
    if smooth > 0.0:
        gray = _gauss_blur(gray, sigma=smooth)

    # ---- gradienty Sobela -> styczna (tangent) ----
    kx, ky = _sobel_kernels(ksize)
    gx = _conv2(gray, kx)
    gy = _conv2(gray, ky)
    mag = np.sqrt(gx * gx + gy * gy) + 1e-12

    # jednostkowy gradient, styczna (rotacja o +90°)
    vx, vy = gx / mag, gy / mag
    tx, ty = -vy, vx

    # ---- wagi: maska * edge_bias * amplitude ----
    w_mask = _resolve_mask(ctx, mask_key, H, W)
    w_edge = _edge_weight(mag, edge_bias)
    w_amp = _amplitude_weight(ctx, H, W, use_amp)
    weight = w_mask * w_edge * w_amp

    if weight.max() <= 0 or strength <= 0:
        # diagnostyki i szybki powrót
        _emit_diag(ctx, mag, tx, ty)
        return a

    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    out = a.copy()

    step = float(max(0.0, strength))
    for _ in range(iters):
        dx = (tx * step * weight).round().astype(np.int32)
        dy = (ty * step * weight).round().astype(np.int32)
        if clamp:
            x2 = np.clip(xx + dx, 0, W - 1)
            y2 = np.clip(yy + dy, 0, H - 1)
        else:
            x2 = (xx + dx) % W
            y2 = (yy + dy) % H
        # per-channel kopiujemy przesunięte piksele (uint8)
        for ch in range(3):
            out[..., ch] = out[y2, x2, ch]

    _emit_diag(ctx, mag, tx, ty)
    return out


# -------------------------- diagnostyka HUD --------------------------

def _emit_diag(ctx, mag: np.ndarray, tx: np.ndarray, ty: np.ndarray) -> None:
    try:
        m = (mag / (mag.max() + 1e-12)).astype(np.float32)
        txx = ((tx + 1.0) * 0.5).astype(np.float32)
        tyy = ((ty + 1.0) * 0.5).astype(np.float32)

        def _u8(g: np.ndarray) -> np.ndarray:
            g = np.clip(g, 0.0, 1.0)
            u = (g * 255.0 + 0.5).astype(np.uint8)
            return np.stack([u, u, u], axis=-1)

        ctx.cache["diag/acw/mag"] = _u8(m)
        ctx.cache["diag/acw/tx"] = _u8(txx)
        ctx.cache["diag/acw/ty"] = _u8(tyy)
    except Exception:
        pass


# -------------------------- Pomocnicze --------------------------

def _sobel_kernels(ksize: int) -> tuple[np.ndarray, np.ndarray]:
    ksize = 5 if int(ksize) == 5 else 3
    if ksize == 3:
        kx = np.array([[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]], dtype=np.float32)
        ky = np.array([[1, 2, 1],
                       [0, 0, 0],
                       [-1, -2, -1]], dtype=np.float32)
    else:
        kx = np.array([[2, 1, 0, -1, -2],
                       [3, 2, 0, -2, -3],
                       [4, 3, 0, -3, -4],
                       [3, 2, 0, -2, -3],
                       [2, 1, 0, -1, -2]], dtype=np.float32)
        ky = kx.T
    return kx, ky


def _conv2(img: np.ndarray, k: np.ndarray) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)
    kh, kw = k.shape
    ph, pw = kh // 2, kw // 2
    pad = np.pad(img, ((ph, ph), (pw, pw)), mode="reflect")
    out = np.empty_like(img, dtype=np.float32)
    # prosta koewolucja (czytelna); w razie potrzeby można później przyspieszyć
    for i in range(out.shape[0]):
        row = pad[i:i + kh, :]
        for j in range(out.shape[1]):
            out[i, j] = np.sum(row[:, j:j + kw] * k)
    return out


def _gauss_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img
    radius = max(1, int(round(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    g = np.exp(-0.5 * (x / float(sigma)) ** 2);
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
    h = min(H, mh);
    w = min(W, mw)
    out[:h, :w] = m[:h, :w].astype(np.float32)
    if h < H:
        out[h:, :w] = out[h - 1:h, :w]
    if w < W:
        out[:H, w:] = out[:H, w - 1:w]
    return out


def _edge_weight(mag: np.ndarray, edge_bias: float) -> np.ndarray:
    m = mag / (mag.max() + 1e-12)
    if abs(edge_bias) < 1e-12:
        return m
    if edge_bias > 0:
        # preferuj krawędzie (podniesienie wartości)
        return np.power(m, 1.0 / (1.0 + edge_bias))
    else:
        # preferuj gładkie obszary
        return np.power(1.0 - m, 1.0 / (1.0 + (-edge_bias)))


def _amplitude_weight(ctx, H: int, W: int, use_amp: float | bool) -> np.ndarray:
    if not hasattr(ctx, "amplitude") or ctx.amplitude is None:
        return np.ones((H, W), dtype=np.float32)
    amp = np.asarray(ctx.amplitude).astype(np.float32)
    if amp.shape != (H, W):
        amp = _fit_mask_hw(amp, H, W)
    # normalizacja i baza 0.25 dla stabilności
    amp -= amp.min()
    amp /= (amp.max() + 1e-12)
    base = 0.25 + 0.75 * amp
    if isinstance(use_amp, bool):
        return base if use_amp else np.ones((H, W), dtype=np.float32)
    return base * float(max(0.0, use_amp))
