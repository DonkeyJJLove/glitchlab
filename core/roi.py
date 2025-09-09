# glitchlab/core/roi.py
# -*- coding: utf-8 -*-
"""
ROI utilities:
- maski z Sobela i prostych heurystyk (dark bar near top)
- rasteryzacja wielokątów / prostokątów
- loader ROI z YAML (polygons/rects)
- wrapper amplitude_field (deleguje do utils.make_amplitude)

Maski zwracamy jako float32 (0..1), rozmiar (H,W).
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import yaml

from .utils import to_gray as _to_gray
from .utils import make_amplitude as _make_amplitude


# -----------------------------------------------------------------------------
# Morfologia (bez SciPy): dylacja przez Pillow
# -----------------------------------------------------------------------------
def _dilate_mask(mask: np.ndarray, ksize: int = 3, iters: int = 1) -> np.ndarray:
    if iters <= 0 or ksize <= 1:
        return mask.astype(np.float32)
    if ksize % 2 == 0:
        ksize += 1
    im = Image.fromarray((np.clip(mask, 0, 1) * 255).astype(np.uint8), "L")
    for _ in range(int(iters)):
        im = im.filter(ImageFilter.MaxFilter(int(ksize)))
    out = np.asarray(im, dtype=np.uint8).astype(np.float32) / 255.0
    return np.clip(out, 0.0, 1.0)


# -----------------------------------------------------------------------------
# Krawędzie (Sobel) i heurystyki
# -----------------------------------------------------------------------------
def sobel_edges_mask(
    arr: np.ndarray,
    thresh: float = 80.0,
    dilate: int = 0,
    ksize: int = 3,
) -> np.ndarray:
    """
    Binarna maska krawędzi na bazie Sobela.
    - thresh w skali 0..255 (na module gradientu)
    - dilate: liczba iteracji dylacji MaxFilter(ksize)
    """
    y = _to_gray(arr)
    # padding „edge” jak w oryginale
    yp = np.pad(y, ((1, 1), (1, 1)), mode="edge")

    gx = (-1 * yp[:-2, :-2] + 1 * yp[:-2, 2:]
          -2 * yp[1:-1, :-2] + 2 * yp[1:-1, 2:]
          -1 * yp[2:, :-2] + 1 * yp[2:, 2:])
    gy = (-1 * yp[:-2, :-2] - 2 * yp[:-2, 1:-1] - 1 * yp[:-2, 2:]
          +1 * yp[2:, :-2] + 2 * yp[2:, 1:-1] + 1 * yp[2:, 2:])

    mag = np.hypot(gx, gy)  # 0..~(1e3)
    mask = (mag > float(thresh)).astype(np.float32)
    if dilate > 0:
        mask = _dilate_mask(mask, ksize=max(3, int(ksize)), iters=int(dilate))
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def dark_bar_mask(arr: np.ndarray, top_fraction: float = 0.35, thresh: float = 80.0) -> np.ndarray:
    """
    Heurystyka: duży ciemny panel w górnej części (nagłówek, baner).
    Zwraca maskę float32 0..1 o rozmiarze wejścia.
    """
    h, w, _ = arr.shape
    sub = arr[: max(1, int(h * float(top_fraction))), : max(1, int(w * 0.9))]
    gray = _to_gray(sub)
    mask_sub = (gray < float(thresh)).astype(np.float32)
    out = np.zeros((h, w), dtype=np.float32)
    out[: mask_sub.shape[0], : mask_sub.shape[1]] = mask_sub
    return out


# -----------------------------------------------------------------------------
# Rasteryzacja ROI
# -----------------------------------------------------------------------------
def polygons_mask(shape_hw: Tuple[int, int], polys: List[List[Tuple[int, int]]]) -> np.ndarray:
    """
    Maska z listy wielokątów (szybko przez PIL.ImageDraw).
    shape_hw = (H,W)
    """
    H, W = shape_hw
    im = Image.new("L", (W, H), 0)
    dr = ImageDraw.Draw(im)
    for poly in polys or []:
        if len(poly) >= 3:
            dr.polygon([(int(x), int(y)) for (x, y) in poly], fill=255)
    arr = np.asarray(im, dtype=np.uint8).astype(np.float32) / 255.0
    return arr


def rects_mask(shape_hw: Tuple[int, int], rects: List[Tuple[int, int, int, int]]) -> np.ndarray:
    """
    Maska z prostokątów [x,y,w,h] (szybko przez PIL).
    """
    H, W = shape_hw
    im = Image.new("L", (W, H), 0)
    dr = ImageDraw.Draw(im)
    for x, y, w0, h0 in rects or []:
        x0, y0 = int(x), int(y)
        x1, y1 = int(x + w0), int(y + h0)
        dr.rectangle([x0, y0, x1, y1], fill=255)
    arr = np.asarray(im, dtype=np.uint8).astype(np.float32) / 255.0
    return arr


# -----------------------------------------------------------------------------
# Amplituda (wrapper do utils)
# -----------------------------------------------------------------------------
def amplitude_field(shape_hw: Tuple[int, int], kind: str = "linear_x", strength: float = 1.0, **kw) -> np.ndarray:
    """
    Kompatybilny wrapper – deleguje do utils.make_amplitude (obsługuje też 'perlin'/'mask').
    """
    return _make_amplitude(shape_hw, kind=kind, strength=strength, ctx=kw.pop("ctx", None), **kw)


# -----------------------------------------------------------------------------
# Loader ROI z YAML (inline string lub plik)
# -----------------------------------------------------------------------------
def load_rois_from_yaml_text(yaml_text: str, shape_hw: Tuple[int, int]) -> Dict[str, np.ndarray]:
    """
    Parsuje YAML w formacie:
      polygons: [ [[x,y],...], ... ]
      rects:    [ [x,y,w,h], ... ]
    Zwraca dict masek (float32 0..1): {"polygons": ..., "rects": ...} – tylko te, które istnieją.
    """
    data = yaml.safe_load(yaml_text) or {}
    H, W = shape_hw
    out: Dict[str, np.ndarray] = {}
    if isinstance(data.get("polygons"), list) and data["polygons"]:
        out["polygons"] = polygons_mask((H, W), data["polygons"])
    if isinstance(data.get("rects"), list) and data["rects"]:
        out["rects"] = rects_mask((H, W), data["rects"])
    return out


def load_rois_from_yaml_file(path: str, shape_hw: Tuple[int, int]) -> Dict[str, np.ndarray]:
    with open(path, "r", encoding="utf-8") as f:
        return load_rois_from_yaml_text(f.read(), shape_hw)


__all__ = [
    "sobel_edges_mask",
    "dark_bar_mask",
    "polygons_mask",
    "rects_mask",
    "amplitude_field",
    "load_rois_from_yaml_text",
    "load_rois_from_yaml_file",
]
