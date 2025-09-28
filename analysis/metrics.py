# glitchlab/analysis/metrics.py
"""
---
version: 2
kind: module
id: "analysis-metrics"
created_at: "2025-09-11"
name: "glitchlab.analysis.metrics"
author: "GlitchLab v2"
role: "Image Metrics Library"
description: >
  Zestaw szybkich metryk obrazu: konwersja do Gray, downsampling,
  entropia (Shannon), gęstość krawędzi (|∇x|+|∇y|), kontrast RMS oraz
  statystyki blokowe dla mozaiki HUD. Zaprojektowane do pracy <50 ms dla obrazu ~1K.
inputs:
  arr: {dtype: "uint8|float32", shape: "(H,W)|(H,W,3)", colorspace: "RGB/Gray"}
  params:
    block: {type: int, default: 16}
    max_side: {type: int, default: 1024}
outputs:
  global:
    compute_entropy: {type: float, units: "bits"}
    edge_density: {type: float, range: "[0,1]"}
    contrast_rms: {type: float, range: "[0,1]"}
  blocks:
    block_stats: {type: dict[(bx,by)->{entropy:float,edges:float,mean:float,variance:float}]}
record_model:
  stage_keys:
    metrics_in: "stage/{i}/metrics_in"
    metrics_out: "stage/{i}/metrics_out"
  mosaic_features: ["entropy","edges","mean","variance"]
interfaces:
  exports: ["to_gray_f32","downsample_max_side","compute_entropy","edge_density","contrast_rms","block_stats"]
  depends_on: ["numpy","Pillow"]
  used_by: ["glitchlab.core.pipeline","glitchlab.analysis.mosaic","glitchlab.analysis.exporters","glitchlab.gui"]
policy:
  deterministic: true
  side_effects: false
constraints:
  - "no SciPy/OpenCV"
  - "clamp/NaN-safe wyniki"
telemetry:
  global_metrics: ["entropy","edge_density","contrast_rms"]
  block_metrics: ["entropy","edges","mean","variance"]
hud:
  channels: {}  # wizualizację mozaiki realizuje analysis.mosaic/GUI
license: "Proprietary"
---
"""
from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
from PIL import Image

__all__ = [
    "to_gray_f32",
    "downsample_max_side",
    "compute_entropy",
    "edge_density",
    "contrast_rms",
    "block_stats",
]

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


def downsample_max_side(arr: np.ndarray, max_side: int = 1024) -> np.ndarray:
    """
    Jeśli potrzeba, skaluje obraz tak, by max(H,W) == max_side (bicubic, Pillow).
    Zachowuje dtype (uint8/float32). Działa dla Gray i RGB.
    """
    H, W = arr.shape[:2]
    m = max(H, W)
    if m <= max_side:
        return arr
    scale = max_side / float(m)
    new_size = (max(1, int(round(W * scale))), max(1, int(round(H * scale))))

    if arr.dtype != np.uint8:
        # konwertujemy tymczasowo do uint8, potem z powrotem do float32 [0,1]
        tmp = np.clip(arr, 0.0, 1.0) if arr.dtype != np.uint8 else arr
        if tmp.dtype != np.uint8:
            tmp = (tmp * 255.0 + 0.5).astype(np.uint8)
        mode = "L" if arr.ndim == 2 else "RGB"
        im = Image.fromarray(tmp, mode=mode).resize(new_size, resample=Image.BICUBIC)
        out = np.asarray(im, dtype=np.float32) / 255.0
        return out
    else:
        mode = "L" if arr.ndim == 2 else "RGB"
        im = Image.fromarray(arr, mode=mode).resize(new_size, resample=Image.BICUBIC)
        return np.asarray(im, dtype=np.uint8)


# -----------------------------
# Metryki globalne
# -----------------------------

def compute_entropy(arr: np.ndarray, bins: int = 256) -> float:
    """
    Shannon entropy (bits) na gray [0,1] (histogram z 'bins' koszami).
    """
    g = to_gray_f32(arr)
    # histogram w [0,1]
    hist, _ = np.histogram(g, bins=bins, range=(0.0, 1.0))
    p = hist.astype(np.float64)
    s = p.sum()
    if s <= 0:
        return 0.0
    p /= s
    # unikamy log(0)
    p = p[p > 0]
    H = -np.sum(p * (np.log2(p)))
    # maksymalna entropia = log2(bins); nie normalizujemy aby zachować skale "bitową"
    return float(np.clip(H, 0.0, np.log2(bins)))


def edge_density(arr: np.ndarray) -> float:
    """
    Średnia gęstość krawędzi: E[|∇x| + |∇y|] na gray [0,1].
    """
    g = to_gray_f32(arr)
    gx = np.zeros_like(g, dtype=np.float32)
    gy = np.zeros_like(g, dtype=np.float32)
    gx[:, :-1] = g[:, 1:] - g[:, :-1]
    gy[:-1, :] = g[1:, :] - g[:-1, :]
    mag = np.abs(gx) + np.abs(gy)
    # opcjonalne docięcie, by mieściło się ~[0,1]
    return float(np.clip(mag.mean(), 0.0, 1.0))


def contrast_rms(arr: np.ndarray) -> float:
    """
    RMS kontrast: sqrt(mean((gray - mean(gray))^2)) – zakres ~[0,1].
    """
    g = to_gray_f32(arr)
    mu = float(g.mean()) if g.size > 0 else 0.0
    d = g - mu
    rms = float(np.sqrt(np.mean(d * d))) if g.size > 0 else 0.0
    return float(np.clip(rms, 0.0, 1.0))


# -----------------------------
# Metryki blokowe (dla mozaiki)
# -----------------------------

def block_stats(
    arr: np.ndarray,
    block: int = 16,
    max_side: int = 1024,
    bins: int = 64,
) -> Dict[Tuple[int, int], Dict[str, float]]:
    """
    Statystyki kafelkowe: entropy/edges/mean/variance dla bloków block×block.
    Dla przyspieszenia najpierw downsample do max_side.
    Zwraca mapę {(bx,by) -> {...}} gdzie bx,by to indeksy w siatce bloków.
    """
    if block < 4:
        raise ValueError("block must be >= 4")

    # downsample (zachowujemy typ wyjścia z downsample: u8 lub f32)
    small = downsample_max_side(arr, max_side=max_side)
    g = to_gray_f32(small)
    H, W = g.shape
    out: Dict[Tuple[int, int], Dict[str, float]] = {}

    # liczba bloków
    bx_count = int(np.ceil(W / float(block)))
    by_count = int(np.ceil(H / float(block)))

    # prealokacje do histogramu
    hist_bins = bins
    for by in range(by_count):
        y0 = by * block
        y1 = min(H, y0 + block)
        for bx in range(bx_count):
            x0 = bx * block
            x1 = min(W, x0 + block)
            tile = g[y0:y1, x0:x1]
            if tile.size == 0:
                continue

            # mean/variance
            m = float(tile.mean())
            v = float(tile.var())

            # entropy
            h_hist, _ = np.histogram(tile, bins=hist_bins, range=(0.0, 1.0))
            p = h_hist.astype(np.float64)
            s = p.sum()
            if s > 0:
                p /= s
                p = p[p > 0]
                H_bits = -np.sum(p * np.log2(p))
                H_bits = float(np.clip(H_bits, 0.0, np.log2(hist_bins)))
            else:
                H_bits = 0.0

            # edges
            gx = np.zeros_like(tile)
            gy = np.zeros_like(tile)
            gx[:, :-1] = tile[:, 1:] - tile[:, :-1]
            gy[:-1, :] = tile[1:, :] - tile[:-1, :]
            ed = float(np.clip((np.abs(gx) + np.abs(gy)).mean(), 0.0, 1.0))

            out[(bx, by)] = {
                "entropy": H_bits,
                "edges": ed,
                "mean": float(np.clip(m, 0.0, 1.0)),
                "variance": float(np.clip(v, 0.0, 1.0)),
            }

    return out

