# glitchlab/core/metrics/compare.py
"""
---
version: 2
kind: module
id: "core-metrics-compare"
created_at: "2025-09-11"
name: "glitchlab.core.metrics.compare"
author: "GlitchLab v2"
role: "Image Comparison Metrics (PSNR & SSIM-box)"
description: >
  Lekkie metryki porównawcze obrazu bez SciPy/OpenCV. Normalizuje wejścia do Gray
  float32 [0,1], liczy PSNR (dB) oraz przybliżone SSIM z oknem pudełkowym (integral image).
  Zaprojektowane do szybkich porównań w pipeline/HUD i testach antykruchości.
inputs:
  a: {dtype: "uint8|float32", shape: "(H,W)|(H,W,3)", colorspace: "Gray|RGB"}
  b: {dtype: "uint8|float32", shape: "(H,W)|(H,W,3)", colorspace: "Gray|RGB"}
  params:
    win: {type: int, default: 7, desc: "rozmiar okna SSIM (nieparzysty)"}
    K1:  {type: float, default: 0.01}
    K2:  {type: float, default: 0.03}
outputs:
  psnr: {type: float, units: "dB", desc: "∞ dla MSE≈0"}
  ssim: {type: float, range: "[0,1]"}
interfaces:
  exports: ["to_gray_f32","psnr","ssim_box"]
  depends_on: ["numpy"]
  used_by: ["glitchlab.core.pipeline","glitchlab.analysis.exporters","glitchlab.app"]
policy:
  deterministic: true
  side_effects: false
constraints:
  - "no SciPy/OpenCV"
  - "operacje w float32 [0,1]"
telemetry:
  metrics: ["psnr","ssim_box"]
hud:
  channels: {}  # moduł nie zapisuje do cache bezpośrednio
license: "Proprietary"
---
"""
from __future__ import annotations

import numpy as np
from typing import Tuple

__all__ = ["to_gray_f32", "psnr", "ssim_box"]


# ------------------------------
# Helpers: dtype/shape handling
# ------------------------------

def to_gray_f32(arr: np.ndarray) -> np.ndarray:
    """
    Konwersja obrazu do gray float32 [0,1].
    Wejście: uint8 (H,W[,3]) lub float (H,W[,3]) w [0,1] (inne typy rzutowane).
    """
    if arr.ndim not in (2, 3):
        raise ValueError("to_gray_f32: expected 2D gray or 3D RGB array")
    if arr.dtype == np.uint8:
        a = arr.astype(np.float32) / 255.0
    else:
        a = arr.astype(np.float32, copy=False)
    if a.ndim == 3:
        if a.shape[-1] != 3:
            raise ValueError("to_gray_f32: for 3D inputs, last dim must be 3 (RGB)")
        y = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
        return np.clip(y, 0.0, 1.0, out=y)
    return np.clip(a, 0.0, 1.0, out=a)


# ------------------------------
# PSNR
# ------------------------------

def psnr(a: np.ndarray, b: np.ndarray, *, eps: float = 1e-12) -> float:
    """
    PSNR na Gray [0,1]. Zwraca dB (float, może być inf przy MSE=0).
    """
    x = to_gray_f32(a)
    y = to_gray_f32(b)
    if x.shape != y.shape:
        raise ValueError("psnr: shapes must match")
    diff = x - y
    mse = float(np.mean(diff * diff))
    if mse <= eps:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


# ------------------------------
# SSIM (box window, bez SciPy)
# ------------------------------

def _pad_reflect(a: np.ndarray, pad: Tuple[int, int]) -> np.ndarray:
    """Odbiciowe obramowanie 2D: pad = (pad_y, pad_x)."""
    py, px = pad
    return np.pad(a, ((py, py), (px, px)), mode="reflect")


def _box_mean(a: np.ndarray, win: int) -> np.ndarray:
    """
    Średnia w oknie win×win przez integral image (O(1) na piksel).
    Używa odbiciowego obramowania na brzegach.
    """
    if win < 1 or win % 2 == 0:
        raise ValueError("win must be odd and >=1")
    r = win // 2
    a_pad = _pad_reflect(a, (r, r))  # (H+2r, W+2r)

    # Integral image z wiodącym wierszem/kolumną zer — ułatwia indeksowanie:
    # S[y, x] = suma a_pad[0:y, 0:x]
    S = np.pad(a_pad, ((1, 0), (1, 0)), mode="constant", constant_values=0).cumsum(0).cumsum(1)

    H, W = a.shape
    y0 = np.arange(H)          # top-left w a_pad: [y0, y0+win)
    y1 = y0 + win
    x0 = np.arange(W)
    x1 = x0 + win

    Y0, X0 = np.meshgrid(y0, x0, indexing="ij")
    Y1, X1 = np.meshgrid(y1, x1, indexing="ij")

    # Suma w prostokącie [y0,y1) × [x0,x1)
    total = S[Y1, X1] - S[Y0, X1] - S[Y1, X0] + S[Y0, X0]
    return total / float(win * win)


def ssim_box(a: np.ndarray, b: np.ndarray, *, win: int = 7, K1: float = 0.01, K2: float = 0.03) -> float:
    """
    Aproksymacja SSIM z oknem pudełkowym (uniform). Zwraca średnie SSIM w [0,1].
    Operuje na Gray [0,1], bez SciPy/OpenCV.
    """
    x = to_gray_f32(a)
    y = to_gray_f32(b)
    if x.shape != y.shape:
        raise ValueError("ssim_box: shapes must match")
    if win % 2 == 0 or win < 1:
        raise ValueError("ssim_box: win must be odd and >=1")

    # Lokalne średnie
    mu_x = _box_mean(x, win)
    mu_y = _box_mean(y, win)

    # Lokalne momenty drugie
    x2 = _box_mean(x * x, win)
    y2 = _box_mean(y * y, win)
    xy = _box_mean(x * y, win)

    sigma_x2 = np.maximum(0.0, x2 - mu_x * mu_x)
    sigma_y2 = np.maximum(0.0, y2 - mu_y * mu_y)
    sigma_xy = xy - mu_x * mu_y

    L = 1.0
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    num = (2.0 * mu_x * mu_y + C1) * (2.0 * sigma_xy + C2)
    den = (mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x2 + sigma_y2 + C2)

    den = np.where(den <= 1e-12, 1e-12, den)
    ssim_map = np.clip(num / den, 0.0, 1.0)
    return float(np.mean(ssim_map))
