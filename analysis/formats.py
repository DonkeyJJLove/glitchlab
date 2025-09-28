# glitchlab/analysis/formats.py
"""
---
version: 2
kind: module
id: "analysis-formats"
created_at: "2025-09-11"
name: "glitchlab.analysis.formats"
author: "GlitchLab v2"
role: "Format Forensics (JPEG/PNG)"
description: >
  Heurystyczna mapa siatki 8×8 dla JPEG (na bazie krawędzi i odległości do linii 8 px)
  oraz proste metadane PNG (lossless + wskaźnik gładkości gradientów).
inputs:
  arr: {dtype: "uint8|float32", shape: "(H,W)|(H,W,3)", colorspace: "RGB/Gray"}
outputs:
  jpeg:
    grid8: {dtype: "float32", shape: "(H,W)", range: "[0,1]"}
    notes: {type: "list[str]"}
  png:
    lossless: {type: "bool", default: true}
    notes: {type: "list[str]"}
interfaces:
  exports: ["analyze_jpeg","analyze_png"]
  depends_on: ["numpy"]
  used_by: ["glitchlab.core.pipeline","glitchlab.analysis.exporters","glitchlab.gui"]
policy:
  deterministic: true
  side_effects: false
constraints:
  - "no SciPy/OpenCV"
  - "analiza sygnałowa na pikselach (bez parsowania bitstreamów)"
hud:
  channels:
    jpg_grid: "format/jpg_grid"
    notes: "format/notes"
license: "Proprietary"
---
"""


from __future__ import annotations

from typing import Dict, List
import numpy as np

__all__ = ["analyze_jpeg", "analyze_png"]


def _to_gray_f32(arr: np.ndarray) -> np.ndarray:
    if arr.ndim not in (2, 3):
        raise ValueError("expected 2D gray or 3D RGB")
    if arr.dtype == np.uint8:
        a = arr.astype(np.float32) / 255.0
    else:
        a = arr.astype(np.float32, copy=False)
    if a.ndim == 3:
        if a.shape[-1] != 3:
            raise ValueError("for 3D inputs, last dim must be 3 (RGB)")
        g = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
        return np.clip(g, 0.0, 1.0, out=g)
    return np.clip(a, 0.0, 1.0, out=a)


def _edge_maps(g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gx = np.zeros_like(g, dtype=np.float32)
    gy = np.zeros_like(g, dtype=np.float32)
    gx[:, 1:] = g[:, 1:] - g[:, :-1]
    gy[1:, :] = g[1:, :] - g[:-1, :]
    return np.abs(gx), np.abs(gy)


def analyze_jpeg(arr: np.ndarray, *, sigma: float = 1.0) -> Dict[str, object]:
    """
    Heurystyczna mapa siatki 8x8 dla JPEG: wzmacnia krawędzie w kolumnach/wierszach
    zbliżonych do wielokrotności 8. Zwraca {"grid8": float32[H,W] 0..1, "notes":[...]}.
    """
    g = _to_gray_f32(arr)
    H, W = g.shape
    gx, gy = _edge_maps(g)

    # odległość do najbliższej linii siatki co 8 px
    x = np.arange(W, dtype=np.float32)
    y = np.arange(H, dtype=np.float32)

    # odległość do najbliższego wielokrotności 8: min(x%8, 8 - x%8)
    modx = x % 8.0
    mody = y % 8.0
    dx = np.minimum(modx, 8.0 - modx)
    dy = np.minimum(mody, 8.0 - mody)

    # wagi Gaussa wzdłuż linii siatki; w_x – dla pionowych granic (używa |∂/∂x| = gx),
    # w_y – dla poziomych granic (używa |∂/∂y| = gy)
    s2 = 2.0 * max(1e-6, float(sigma)) ** 2
    wx = np.exp(-(dx ** 2) / s2)  # (W,)
    wy = np.exp(-(dy ** 2) / s2)  # (H,)

    Wx = np.broadcast_to(wx[None, :], (H, W))
    Wy = np.broadcast_to(wy[:, None], (H, W))

    # energia „kratki”: sumujemy wkład krawędzi w okolicach linii siatki
    grid = gx * Wx + gy * Wy

    # normalizacja do [0,1] przez 99.5-percentyl, by ograniczyć wpływ outlierów
    p = float(np.percentile(grid, 99.5)) if grid.size else 0.0
    if p > 1e-8:
        grid = np.clip(grid / p, 0.0, 1.0)
    else:
        grid = np.zeros_like(grid, dtype=np.float32)

    notes: List[str] = []
    # globalny wskaźnik kratki (średnia na liniach)
    v_col = float((gx * Wx).mean()) if grid.size else 0.0
    v_row = float((gy * Wy).mean()) if grid.size else 0.0
    strength = (v_col + v_row) * 0.5
    if strength > 0.03:
        notes.append("Wyraźna siatka 8×8 — prawdopodobny JPEG z artefaktami blokowymi.")
    elif strength > 0.015:
        notes.append("Słaba siatka 8×8 — możliwe subtelne artefakty JPEG.")
    else:
        notes.append("Brak wyraźnej siatki 8×8 — artefakty blokowe niewidoczne.")

    return {"grid8": grid.astype(np.float32, copy=False), "notes": notes}


def analyze_png(arr: np.ndarray) -> Dict[str, object]:
    """
    Minimalne metadane PNG. Nie rozkodowujemy formatu, więc raportujemy jedynie
    'lossless': True i prostą notę o równomierności gradientów (heurystyka bandingu).
    """
    g = _to_gray_f32(arr)
    gx = np.abs(np.diff(g, axis=1))
    gy = np.abs(np.diff(g, axis=0))
    # prosty wskaźnik gładkości (mniejsze gradienty -> gładsze)
    smooth = float(np.median(np.hstack([gx.reshape(-1), gy.reshape(-1)]))) if g.size else 0.0

    notes: List[str] = ["PNG traktowany jako bezstratny (heurystyka)."]
    if smooth < 0.002:
        notes.append("Bardzo gładkie przejścia — małe ryzyko bandingu.")
    elif smooth < 0.01:
        notes.append("Umiarkowanie gładkie przejścia — możliwy delikatny banding.")
    else:
        notes.append("Wysokie lokalne gradienty — banding mało prawdopodobny (obraz teksturowany).")

    return {"lossless": True, "notes": notes}