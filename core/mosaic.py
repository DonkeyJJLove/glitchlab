# glitchlab/core/mosaic.py
"""
---
version: 2
kind: module
id: "core-mosaic"
created_at: "2025-09-11"
name: "glitchlab.core.mosaic"
author: "GlitchLab v2"
role: "Mosaic Grid & Overlay Engine"
description: >
  Generuje siatki mozaiki (square/hex), raster etykiet piksel→komórka, projekcję
  metryk blokowych na kolory oraz prosty blend nakładki. Służy jako wspólna
  „soczewka” wizualizacji struktur (entropia/krawędzie/średnia) w HUD/GUI.
inputs:
  shape_hw: {type: "(H,W)", desc: "rozmiar obrazu w pikselach"}
  mode: {type: "str", enum: ["square","hex"], default: "square"}
  cell_px: {type: "int", min: 4, default: 32}
  block_stats: {type: "dict[(bx,by)->{entropy,edges,mean,variance}]", optional: true}
  overlay:
    image: {dtype: "uint8", shape: "(H,W,3)", desc: "RGB do blendu/projekcji"}
    alpha: {type: "float[0..1]", default: 0.5}
outputs:
  mosaic:
    mode: "square|hex"
    cell_px: "int"
    size: "(H,W)"
    cells: "list[Cell{id, polygon[(x,y)], center(x,y), type, neighbors[]}]"
    raster: "int32 (H,W)  # -1 poza komórkami"
  overlay_rgb?: {dtype: "uint8", shape: "(H,W,3)", desc: "kolorowa projekcja lub blend"}
interfaces:
  exports: ["mosaic_map","mosaic_label_raster","mosaic_project_blocks","mosaic_overlay"]
  depends_on: ["numpy","Pillow"]
  used_by: ["glitchlab.core.pipeline","glitchlab.analysis.exporters","glitchlab.gui"]
policy:
  deterministic: true
  side_effects: false
constraints:
  - "no SciPy/OpenCV"
  - "domyślna projekcja: R←entropy[0..8], G←edges[0..1], B←mean[0..1]"
telemetry:
  legend_defaults: {R: "entropy", G: "edges", B: "mean"}
hud:
  channels:
    mosaic_image: "stage/{i}/mosaic"
    mosaic_meta:  "stage/{i}/mosaic_meta"
license: "Proprietary"
---
"""

# glitchlab/core/mosaic.py
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple
import numpy as np
from PIL import Image

__all__ = [
    "mosaic_map",
    "mosaic_label_raster",
    "mosaic_project_blocks",
    "mosaic_overlay",
]

Cell = Dict[str, Any]   # {"id": int, "polygon": [(x,y)], "center": (x,y), "type": "square|hex", "neighbors": [int]}
Mosaic = Dict[str, Any] # {"mode": str, "cell_px": int, "size": (H,W), "cells": List[Cell], "raster": np.ndarray[int32]}


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def _ensure_hw(shape_hw: Tuple[int, int]) -> Tuple[int, int]:
    if not (isinstance(shape_hw, tuple) and len(shape_hw) == 2):
        raise ValueError("shape_hw must be a (H, W) tuple")
    H, W = int(shape_hw[0]), int(shape_hw[1])
    if H <= 0 or W <= 0:
        raise ValueError("shape_hw must be positive")
    return H, W


def _neighbors_grid(i: int, j: int, nx: int, ny: int) -> List[int]:
    """4-neighborhood for square grid, flattened id = j*nx + i."""
    n: List[int] = []
    def idx(ii: int, jj: int) -> int: return jj * nx + ii
    if i > 0:        n.append(idx(i-1, j))
    if i+1 < nx:     n.append(idx(i+1, j))
    if j > 0:        n.append(idx(i, j-1))
    if j+1 < ny:     n.append(idx(i, j+1))
    return n


def _square_raster(H: int, W: int, cell_px: int) -> Tuple[np.ndarray, List[Cell]]:
    """Buduje raster etykiet i metadane komórek dla siatki kwadratowej."""
    nx = int(np.ceil(W / float(cell_px)))
    ny = int(np.ceil(H / float(cell_px)))
    ids = (np.arange(ny, dtype=np.int32)[:, None] * nx + np.arange(nx, dtype=np.int32)[None, :])
    # upsample przez powtarzanie
    tile = np.ones((cell_px, cell_px), dtype=np.int32)
    raster = np.kron(ids, tile)
    raster = raster[:H, :W]

    cells: List[Cell] = []
    for j in range(ny):
        for i in range(nx):
            cid = int(j * nx + i)
            x0 = i * cell_px
            y0 = j * cell_px
            x1 = min(W, x0 + cell_px)
            y1 = min(H, y0 + cell_px)
            poly = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            cx = (x0 + x1) // 2
            cy = (y0 + y1) // 2
            cells.append({
                "id": cid,
                "polygon": poly,
                "center": (int(cx), int(cy)),
                "type": "square",
                "neighbors": _neighbors_grid(i, j, nx, ny),
            })
    return raster.astype(np.int32, copy=False), cells


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------

def mosaic_map(shape_hw: Tuple[int, int], *, mode: str = "square", cell_px: int = 32) -> Mosaic:
    """
    Generuje definicję mozaiki: etykietowy raster (H,W) oraz listę komórek z geometrią i sąsiedztwami.
    Obsługiwane tryby:
      - "square": pełne wsparcie,
      - "hex": fallback na siatkę kwadratową (spójny interfejs; docelowa rasteryzacja hex w warstwie analysis/HUD).
    """
    H, W = _ensure_hw(shape_hw)
    cell_px = max(5, int(cell_px))

    mode_l = (mode or "square").lower()
    if mode_l == "square":
        raster, cells = _square_raster(H, W, cell_px)
        return {"mode": "square", "cell_px": cell_px, "size": (H, W), "cells": cells, "raster": raster}

    # Fallback "hex" → generujemy identyczny raster jak square, zachowując typ = "hex" (geometry można narysować w HUD)
    raster, cells_sq = _square_raster(H, W, cell_px)
    cells: List[Cell] = []
    for c in cells_sq:
        c_hex = dict(c)
        c_hex["type"] = "hex"
        # lekka aproksymacja sześciokąta na podstawie prostokąta komórki (GUI może nadpisać rysowanie)
        x0, y0 = c["polygon"][0]
        x1, y1 = c["polygon"][2]
        w = x1 - x0
        h = y1 - y0
        dx = int(round(w * 0.2))
        poly_hex = [
            (x0 + dx, y0),
            (x1 - dx, y0),
            (x1, y0 + h // 2),
            (x1 - dx, y1),
            (x0 + dx, y1),
            (x0, y0 + h // 2),
        ]
        c_hex["polygon"] = poly_hex
        cells.append(c_hex)
    return {"mode": "hex", "cell_px": cell_px, "size": (H, W), "cells": cells, "raster": raster}


def mosaic_label_raster(mosaic: Mapping[str, Any]) -> np.ndarray:
    """Zwraca raster etykiet (H,W) int32."""
    lab = mosaic.get("raster")
    if not isinstance(lab, np.ndarray) or lab.ndim != 2 or lab.dtype.kind not in "iu":
        raise ValueError("mosaic_label_raster: invalid mosaic['raster']")
    return lab.astype(np.int32, copy=False)


def mosaic_project_blocks(
    block_stats: Mapping[Tuple[int, int], Mapping[str, float]],
    mosaic: Mosaic,
    *,
    map_spec: Optional[Mapping[str, Tuple[str, Tuple[float, float]]]] = None,
) -> np.ndarray:
    """
    Koloruje komórki mozaiki na podstawie metryk blokowych (np. z analysis.block_stats).
    Dla każdej komórki pobiera metrykę z bloku zawierającego jej środek.
    Domyślna projekcja: R<-entropy, G<-edges, B<-mean (każdy znormalizowany po całej przestrzeni).
    """
    H, W = mosaic.get("size", (0, 0))
    if not (isinstance(H, int) and isinstance(W, int) and H > 0 and W > 0):
        raise ValueError("mosaic_project_blocks: invalid mosaic['size']")
    lab = mosaic_label_raster(mosaic)
    cells = mosaic.get("cells", [])
    overlay = np.zeros((H, W, 3), dtype=np.uint8)
    if not cells or not block_stats:
        return overlay

    # rozmiar siatki bloków w block_stats
    if not block_stats:
        return overlay
    xs = [ij[0] for ij in block_stats.keys()]
    ys = [ij[1] for ij in block_stats.keys()]
    bx = (max(xs) + 1) if xs else 1
    by = (max(ys) + 1) if ys else 1
    bw = max(1, W // bx)
    bh = max(1, H // by)

    # zbierz profile metryk do normalizacji
    def _collect(metric_name: str) -> List[float]:
        vals: List[float] = []
        for v in block_stats.values():
            x = v.get(metric_name)
            if x is not None and np.isfinite(x):
                vals.append(float(x))
        return vals or [0.0]

    if map_spec is None:
        map_spec = {
            "R": ("entropy", (0.0, 0.0)),  # zakresy ustalimy poniżej
            "G": ("edges",   (0.0, 0.0)),
            "B": ("mean",    (0.0, 0.0)),
        }

    # policz globalne zakresy jeśli (0,0)
    ms = dict(map_spec)
    for ch in ("R", "G", "B"):
        name, (lo, hi) = ms[ch]
        if lo == hi:
            vals = _collect(name)
            lo = min(vals)
            hi = max(vals) if max(vals) > lo else lo + 1.0
            ms[ch] = (name, (float(lo), float(hi)))

    def _norm(val: float, lo: float, hi: float) -> float:
        if hi <= lo: return 0.0
        return float(np.clip((val - lo) / (hi - lo), 0.0, 1.0))

    # pokoloruj po komórkach
    lab_int = lab.astype(np.int64, copy=False)
    for c in cells:
        (cx, cy) = c.get("center", (0, 0))
        bi = int(np.clip(cx // bw, 0, bx - 1))
        bj = int(np.clip(cy // bh, 0, by - 1))
        stats = block_stats.get((bi, bj), {})
        r_name, (r_lo, r_hi) = ms["R"]
        g_name, (g_lo, g_hi) = ms["G"]
        b_name, (b_lo, b_hi) = ms["B"]
        r = _norm(float(stats.get(r_name, 0.0)), r_lo, r_hi)
        g = _norm(float(stats.get(g_name, 0.0)), g_lo, g_hi)
        b = _norm(float(stats.get(b_name, 0.0)), b_lo, b_hi)
        rr, gg, bb = int(r * 255.0 + 0.5), int(g * 255.0 + 0.5), int(b * 255.0 + 0.5)
        overlay[lab_int == int(c["id"])] = (rr, gg, bb)

    return overlay


def mosaic_overlay(img_u8: np.ndarray, overlay_u8: np.ndarray, *, alpha: float = 0.5, clamp: bool = True) -> np.ndarray:
    """
    Prosty blend obrazu (RGB) i kolorowego overlay'a mozaiki.
    alpha ∈ [0,1]: 0 → sam obraz, 1 → sam overlay.
    """
    if img_u8.ndim != 3 or img_u8.shape[-1] != 3 or overlay_u8.shape != img_u8.shape:
        raise ValueError("mosaic_overlay: shapes must match and be RGB")
    alpha = float(np.clip(alpha, 0.0, 1.0))
    a = img_u8.astype(np.float32) / 255.0
    o = overlay_u8.astype(np.float32) / 255.0
    y = a * (1.0 - alpha) + o * alpha
    if clamp:
        y = np.clip(y, 0.0, 1.0)
    return (y * 255.0 + 0.5).astype(np.uint8)

