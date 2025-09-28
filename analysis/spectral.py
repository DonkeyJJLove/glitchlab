"""
---
version: 2
kind: module
id: "analysis-spectral"
created_at: "2025-09-11"
name: "glitchlab.analysis.spectral"
author: "GlitchLab v2"
role: "Spectral & Histogram Analysis"
description: >
  Log-magnitude FFT (fft2+shift), energia w pierścieniach i sektorach kątowych
  oraz histogram jasności. Minimalne zależności (NumPy+Pillow), bez SciPy/OpenCV.
inputs:
  arr: {dtype: "uint8|float32", shape: "(H,W)|(H,W,3)", colorspace: "RGB/Gray"}
  max_side: {type: int, default: 1024}
  bins: {type: int, default: 256}
  ring: {low: float, high: float}
  sector: {angle_deg: float, width_deg: float}
outputs:
  fft_mag: {dtype: "float32", shape: "(H,W)", range: "[0,1]"}
  band_ring: {type: float, range: "[0,1]"}
  band_sector: {type: float, range: "[0,1]"}
  hist: {bins: "np.ndarray(float32)", counts: "np.ndarray(float32, sum=1.0)"}
interfaces:
  exports: ["to_gray_f32","fft_mag","band_ring","band_sector","hist"]
  depends_on: ["numpy","Pillow"]
  used_by: ["glitchlab.core.pipeline","glitchlab.analysis.exporters","glitchlab.gui"]
policy:
  deterministic: true
  side_effects: false
constraints:
  - "no SciPy/OpenCV"
  - "operacje w float32 [0,1]"
telemetry:
  snapshots: ["stage/{i}/fft_mag","stage/{i}/hist"]  # zwykle zapisywane przez pipeline/GUI
hud:
  channels: {}
license: "Proprietary"
---
"""
# glitchlab/analysis/spectral.py
from __future__ import annotations

from typing import Tuple
import numpy as np
from PIL import Image

__all__ = ["to_gray_f32", "fft_mag", "band_ring", "band_sector", "hist"]


# -----------------------------
# Konwersje i pomocnicze
# -----------------------------

def to_gray_f32(arr: np.ndarray) -> np.ndarray:
    """
    uint8 (H,W[,3]) lub float (H,W[,3]) -> gray float32 [0,1]
    """
    if arr.ndim not in (2, 3):
        raise ValueError("to_gray_f32: expected 2D gray or 3D RGB")
    if arr.dtype == np.uint8:
        a = arr.astype(np.float32) / 255.0
    else:
        a = arr.astype(np.float32, copy=False)

    if a.ndim == 3:
        if a.shape[-1] != 3:
            raise ValueError("to_gray_f32: for 3D inputs, last dim must be 3 (RGB)")
        g = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
        return np.clip(g, 0.0, 1.0, out=g)
    return np.clip(a, 0.0, 1.0, out=a)


def _downsample_max_side_gray(g: np.ndarray, max_side: int = 1024) -> np.ndarray:
    H, W = g.shape
    m = max(H, W)
    if m <= max_side:
        return g
    scale = max_side / float(m)
    new_size = (max(1, int(round(W * scale))), max(1, int(round(H * scale))))
    im = Image.fromarray((np.clip(g, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), mode="L")
    im = im.resize(new_size, resample=Image.BICUBIC)
    return np.asarray(im, dtype=np.float32) / 255.0


# -----------------------------
# FFT & pasma
# -----------------------------

def fft_mag(arr: np.ndarray, *, max_side: int = 1024) -> np.ndarray:
    """
    Zwraca log-magnitude widma: float32 [0,1], z przesuniętym zerem (fftshift).
    """
    g = to_gray_f32(arr)
    g = _downsample_max_side_gray(g, max_side=max_side)
    # okno Hann dla delikatnego wygładzenia brzegów (redukcja wycieków)
    H, W = g.shape
    wy = np.hanning(max(H, 2))[:H]
    wx = np.hanning(max(W, 2))[:W]
    win = wy[:, None] * wx[None, :]
    gw = g * win

    F = np.fft.fftshift(np.fft.fft2(gw))
    mag = np.abs(F).astype(np.float32)

    # log-scale i normalizacja do [0,1]
    mag = np.log1p(mag)
    mag -= mag.min()
    if mag.max() > 0:
        mag /= mag.max()
    return mag.astype(np.float32, copy=False)


def band_ring(mag: np.ndarray, low: float, high: float) -> float:
    """
    Średnia amplituda w pierścieniu o promieniach [low, high] w normalizowanych
    jednostkach (0..1) liczonych względem pół-przekątnej obrazu.
    """
    if mag.ndim != 2:
        raise ValueError("band_ring: mag must be 2D")
    H, W = mag.shape
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    yy, xx = np.indices((H, W), dtype=np.float32)
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    # normalizacja promienia do [0,1] względem max możliwego (pół-przekątnej)
    r_norm = r / np.sqrt(cy ** 2 + cx ** 2)

    lo = max(0.0, float(low))
    hi = max(lo, float(high))
    mask = (r_norm >= lo) & (r_norm <= hi)
    if not mask.any():
        return 0.0
    val = float(mag[mask].mean())
    return float(np.clip(val, 0.0, 1.0))


def band_sector(mag: np.ndarray, angle_deg: float, width_deg: float) -> float:
    """
    Średnia amplituda w sektorze kątowym (środek=angle_deg, szerokość=width_deg).
    Kąty w stopniach, 0° w prawo (oś X+), rośnie przeciwnie do ruchu wskazówek.
    """
    if mag.ndim != 2:
        raise ValueError("band_sector: mag must be 2D")
    H, W = mag.shape
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    yy, xx = np.indices((H, W), dtype=np.float32)
    ang = np.degrees(np.arctan2(yy - cy, xx - cx))  # [-180, 180]
    a0 = float(angle_deg)
    half = max(0.0, float(width_deg)) / 2.0

    # odległość kątowa z zawijaniem
    da = np.abs((ang - a0 + 180.0) % 360.0 - 180.0)
    mask = da <= half
    if not mask.any():
        return 0.0
    val = float(mag[mask].mean())
    return float(np.clip(val, 0.0, 1.0))


# -----------------------------
# Histogram
# -----------------------------

def hist(arr: np.ndarray, bins: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """
    Histogram jasności Gray [0,1]. Zwraca (bin_centers, counts_norm), gdzie sum(counts_norm)=1.
    """
    g = to_gray_f32(arr)
    h, edges = np.histogram(g, bins=bins, range=(0.0, 1.0))
    counts = h.astype(np.float64)
    s = counts.sum()
    if s > 0:
        counts /= s
    # środki przedziałów
    centers = (edges[:-1] + edges[1:]) * 0.5
    return centers.astype(np.float32), counts.astype(np.float32)
