# glitchlab/analysis/diff.py
"""
version: 2
kind: module
id: "analysis-diff"
created_at: "2025-09-11"
name: "glitchlab.analysis.diff"
author: "GlitchLab v2"
role: "Visual Diff & Change Statistics"
description: >
  Oblicza różnice pomiędzy dwoma obrazami: mapę |Δ| (Gray), składowe per-kanał,
  prostą „heatmapę” oraz statystyki (mean/p95/max; opcjonalnie PSNR). Dostosowuje
  rozmiary i skaluje do max_side dla wydajności.
inputs:
  a: {dtype: "uint8|float32", shape: "(H,W)|(H,W,3)"}
  b: {dtype: "uint8|float32", shape: "(H,W)|(H,W,3)"}
  max_side: {type: int, default: 1024}
  add_psnr: {type: bool, default: false}
outputs:
  abs: {dtype: "float32", shape: "(H,W)", range: "[0,1]"}
  per_channel: {dtype: "tuple(float32,float32,float32)", shape: "(H,W)×3", range: "[0,1]"}
  heatmap: {dtype: "float32", shape: "(H,W)", range: "[0,1]"}
  stats: {type: "dict{mean,p95,max[,psnr]}", units: "mean/p95/max in [0,1], psnr in dB"}
interfaces:
  exports: ["to_rgb_f32","resize_like","compute_diff"]
  depends_on: ["numpy","Pillow"]
  used_by: ["glitchlab.core.pipeline","glitchlab.analysis.exporters","glitchlab.gui"]
policy:
  deterministic: true
  side_effects: false
constraints:
  - "no SciPy/OpenCV"
  - "wyniki w float32 [0,1] poza PSNR (dB)"
telemetry:
  channels:
    diff_image: "stage/{i}/diff"
    diff_stats: "stage/{i}/diff_stats"
license: "Proprietary"
---
"""

# glitchlab/analysis/diff.py
from __future__ import annotations

from typing import Any, Dict, Tuple
import numpy as np
from PIL import Image

__all__ = ["to_rgb_f32", "resize_like", "compute_diff"]


# -----------------------------
# Konwersje i skalowanie
# -----------------------------

def to_rgb_f32(arr: np.ndarray) -> np.ndarray:
    """
    uint8/float, Gray/RGB -> RGB float32 [0,1]
    """
    if arr.ndim == 2:
        if arr.dtype == np.uint8:
            a = arr.astype(np.float32) / 255.0
        else:
            a = arr.astype(np.float32, copy=False)
        a = np.clip(a, 0.0, 1.0, out=a)
        return np.stack([a, a, a], axis=-1)
    elif arr.ndim == 3:
        if arr.shape[-1] == 3:
            if arr.dtype == np.uint8:
                a = arr.astype(np.float32) / 255.0
            else:
                a = arr.astype(np.float32, copy=False)
            return np.clip(a, 0.0, 1.0, out=a)
        else:
            raise ValueError("to_rgb_f32: last dim must be 3 for 3D arrays")
    else:
        raise ValueError("to_rgb_f32: expected 2D or 3D array")


def _resize_to(arr_u8: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    mode = "L" if (arr_u8.ndim == 2 or arr_u8.shape[-1] == 1) else "RGB"
    im = Image.fromarray(arr_u8, mode=mode).resize(size, resample=Image.BICUBIC)
    return np.asarray(im, dtype=np.uint8)


def resize_like(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Zwraca (a_resized, b_resized) o wspólnym rozmiarze (H',W'), dopasowując do mniejszej strony (bicubic).
    """
    Ha, Wa = a.shape[:2]
    Hb, Wb = b.shape[:2]
    # wybierz rozmiar docelowy jako min(max_side a, max_side b) per wymiar – tu: bierzemy rozmiar a
    size = (Wa, Ha)
    if (Hb, Wb) != (Ha, Wa):
        # u8 konwersja po drodze
        if a.dtype != np.uint8:
            au8 = (np.clip(a, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        else:
            au8 = a
        if b.dtype != np.uint8:
            bu8 = (np.clip(b, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        else:
            bu8 = b
        b_res = _resize_to(bu8, (Wa, Ha))
        a_res = au8
        return a_res, b_res
    else:
        if a.dtype == np.uint8 and b.dtype == np.uint8:
            return a, b
        # wyrównaj typy do u8
        au8 = (np.clip(a, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8) if a.dtype != np.uint8 else a
        bu8 = (np.clip(b, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8) if b.dtype != np.uint8 else b
        return au8, bu8


def _resize_max_side_rgb(a_rgb: np.ndarray, b_rgb: np.ndarray, max_side: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Skaluje oba obrazy (RGB f32 [0,1]) tak, by ich max(H,W) <= max_side i miały identyczny rozmiar.
    Rozmiar docelowy – rozmiar mniejszego po downsamplingu.
    """

    def down(a: np.ndarray) -> np.ndarray:
        H, W = a.shape[:2]
        m = max(H, W)
        if m <= max_side:
            return a
        scale = max_side / float(m)
        new_size = (max(1, int(round(W * scale))), max(1, int(round(H * scale))))
        im = Image.fromarray((np.clip(a, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), mode="RGB")
        im = im.resize(new_size, resample=Image.BICUBIC)
        return (np.asarray(im, dtype=np.float32) / 255.0)

    A = down(a_rgb)
    B = down(b_rgb)
    Ha, Wa = A.shape[:2]
    Hb, Wb = B.shape[:2]
    # dopasuj B do A
    if (Ha, Wa) != (Hb, Wb):
        im = Image.fromarray((np.clip(B, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), mode="RGB")
        im = im.resize((Wa, Ha), resample=Image.BICUBIC)
        B = np.asarray(im, dtype=np.float32) / 255.0
    return A, B


# -----------------------------
# Różnice i statystyki
# -----------------------------
def compute_diff(
        a: np.ndarray,
        b: np.ndarray,
        *,
        max_side: int = 1024,
        add_psnr: bool = False,
) -> Dict[str, Any]:
    """
    Zwraca:
      {
        "abs": gray |Δ| [0,1],
        "per_channel": (dR,dG,dB) [0,1],
        "heatmap": gray [0,1],
        "stats": {"mean":..,"p95":..,"max":.., "psnr"?:..}
      }
    """
    Ar = to_rgb_f32(a)
    Br = to_rgb_f32(b)
    A, B = _resize_max_side_rgb(Ar, Br, max_side=max_side)

    # per channel abs
    d = np.abs(A - B)  # [0,1], shape (H,W,3)
    dR = d[..., 0]
    dG = d[..., 1]
    dB = d[..., 2]
    # gray abs
    dY = 0.299 * dR + 0.587 * dG + 0.114 * dB
    dY = np.clip(dY, 0.0, 1.0)

    # lekka „heatmapa” = samo dY (HUD może pokolorować po swojemu)
    heat = dY

    # statystyki
    flat = dY.reshape(-1)
    mean = float(np.mean(flat)) if flat.size else 0.0
    p95 = float(np.percentile(flat, 95.0)) if flat.size else 0.0
    dmax = float(np.max(flat)) if flat.size else 0.0
    stats = {"mean": mean, "p95": p95, "max": dmax}

    if add_psnr:
        # proste PSNR na gray [0,1] po dopasowaniu rozmiarów
        Ay = 0.299 * A[..., 0] + 0.587 * A[..., 1] + 0.114 * A[..., 2]
        By = 0.299 * B[..., 0] + 0.587 * B[..., 1] + 0.114 * B[..., 2]
        diff = Ay - By
        mse = float(np.mean(diff * diff)) if diff.size else 0.0
        if mse <= 1e-12:
            psnr = float("inf")
        else:
            psnr = 10.0 * np.log10(1.0 / mse)
        stats["psnr"] = psnr

    return {
        "abs": dY.astype(np.float32, copy=False),
        "per_channel": (dR.astype(np.float32, copy=False),
                        dG.astype(np.float32, copy=False),
                        dB.astype(np.float32, copy=False)),
        "heatmap": heat.astype(np.float32, copy=False),
        "stats": stats,
    }
