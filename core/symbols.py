# glitchlab/core/symbols.py
# -*- coding: utf-8 -*-
"""
---
version: 2
kind: module
id: "core-symbols"
created_at: "2025-09-11"
name: "glitchlab.core.symbols"
author: "GlitchLab v2"
role: "Bitmap → Mask & Procedural Symbols"
description: >
  Konwertuje bitmapy RGB/Gray do masek float32 [0,1] oraz generuje proste symbole
  proceduralne (circle, ring, square, triangle, plus/cross, diamond, hex). Zapewnia
  funkcję stamp_mask do nakładania masek na płótno z różnymi trybami łączenia.

inputs:
  bitmap?: {dtype: "uint8", shape: "(H,W[,3])", desc: "wejściowa bitmapa do binaryzacji"}
  channel?: {type: "int", default: 0, desc: "kanał bitmapy RGB użyty do maski"}
  invert?: {type: "bool", default: false, desc: "odwrócenie maski po binaryzacji"}
  thresh?: {type: "int", default: 127, range: "[0,255]", desc: "próg binaryzacji"}
  symbol.name?: {enum: ["circle","ring","square","triangle","plus","cross","diamond","hex"], desc: "nazwa symbolu"}
  symbol.size?: {type: "(int,int)", desc: "rozmiar maski symbolu (H,W)"}
  stamp.dst?: {dtype: "float32[0,1]", shape: "(H,W)"}
  stamp.mask?: {dtype: "float32[0,1]", shape: "(h,w)"}
  stamp.xy?: {type: "(int,int)", desc: "pozycja lewego-górnego rogu dla nakładania"}
  stamp.mode?: {enum: ["max","min","mean","mul"], default: "max"}

outputs:
  mask: {dtype: "float32", shape: "(H,W)", range: "[0,1]", desc: "zbinaryzowana lub proceduralna maska"}
  stamped?: {dtype: "float32", shape: "(H,W)", range: "[0,1]", desc: "wynik stamp_mask"}

interfaces:
  exports:
    - "bitmap_to_mask"
    - "load_symbol"
    - "stamp_mask"
  depends_on: ["numpy","Pillow"]
  used_by: ["glitchlab.core.roi","glitchlab.core.pipeline","glitchlab.filters","glitchlab.gui"]

contracts:
  - "bitmap_to_mask zwraca maskę float32 [0,1]; obsługuje (H,W) i (H,W,3)"
  - "load_symbol zwraca float32 [0,1] o zadanym rozmiarze"
  - "stamp_mask nie modyfikuje dst (zwraca nową macierz)"
  - "nakładanie przy wyjściu poza kadr: bezpieczne przycięcie"

constraints:
  - "no SciPy/OpenCV"
  - "poprawne kształty i zakresy wejść"

hud:
  note: "Moduł nie zapisuje telemetrii; maski konsumowane przez filtry/pipeline."

tests_smoke:
  - "bitmap_to_mask(np.zeros((8,8,3),uint8)) → (8,8) f32 [0,1]"
  - "load_symbol('circle',(32,48)) → (32,48) f32 [0,1]"
  - "stamp_mask(np.zeros((16,16),f32), np.ones((4,4),f32), (6,6)) → shape (16,16) f32"
license: "Proprietary"
---
"""


# glitchlab/core/symbols.py
from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
from PIL import Image, ImageDraw

__all__ = [
    "bitmap_to_mask",
    "load_symbol",
    "stamp_mask",
]

# --------------------------------------------------------------------------------------
# Conversions
# --------------------------------------------------------------------------------------

def bitmap_to_mask(img_u8: np.ndarray, *, channel: int = 0, invert: bool = False, thresh: int = 127) -> np.ndarray:
    """
    RGB/Gray uint8 -> maska float32 [0,1] (H,W).
    - wybiera kanał (dla RGB) lub używa gray,
    - binaryzuje prosto progiem 'thresh' (0..255),
    - opcjonalnie odwraca.
    """
    if img_u8.ndim == 2:
        ch = img_u8
    elif img_u8.ndim == 3 and img_u8.shape[-1] == 3:
        c = int(channel)
        if c < 0 or c > 2:
            c = 0
        ch = img_u8[..., c]
    else:
        raise ValueError("bitmap_to_mask: expected (H,W) or (H,W,3) uint8")

    if not np.issubdtype(img_u8.dtype, np.integer):
        ch = ch.astype(np.uint8)

    m = (ch >= int(thresh)).astype(np.float32)
    if invert:
        m = 1.0 - m
    return np.clip(m, 0.0, 1.0, out=m)


# --------------------------------------------------------------------------------------
# Built-in simple symbols (procedural)
# --------------------------------------------------------------------------------------

def _draw_symbol(name: str, size_hw: Tuple[int, int]) -> np.ndarray:
    """
    Rysuje prosty symbol w L (0..255):
      - 'circle', 'ring', 'square', 'triangle', 'plus', 'cross'
      - 'diamond', 'hex' (przybliżony)
    """
    H, W = int(size_hw[0]), int(size_hw[1])
    if H <= 0 or W <= 0:
        raise ValueError("load_symbol: size must be positive")
    im = Image.new("L", (W, H), 0)
    dr = ImageDraw.Draw(im)

    pad = max(1, int(0.06 * min(H, W)))
    x0, y0 = pad, pad
    x1, y1 = W - pad - 1, H - pad - 1
    cx, cy = W // 2, H // 2
    r = max(1, int(0.45 * min(W, H)))

    n = name.lower().strip()
    if n == "circle":
        dr.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
    elif n == "ring":
        dr.ellipse([cx - r, cy - r, cx + r, cy + r], outline=255, width=max(1, r // 6))
    elif n == "square":
        dr.rectangle([x0, y0, x1, y1], fill=255)
    elif n == "triangle":
        pts = [(cx, y0), (x1, y1), (x0, y1)]
        dr.polygon(pts, fill=255)
    elif n in ("plus", "cross"):
        w = max(1, int(0.18 * min(W, H)))
        dr.rectangle([cx - w//2, y0, cx + w//2, y1], fill=255)
        dr.rectangle([x0, cy - w//2, x1, cy + w//2], fill=255)
    elif n == "diamond":
        pts = [(cx, y0), (x1, cy), (cx, y1), (x0, cy)]
        dr.polygon(pts, fill=255)
    elif n == "hex":
        # przybliżony sześciokąt foremny wpisany w prostokąt
        dx = int(0.25 * (x1 - x0 + 1))
        pts = [
            (x0 + dx, y0),
            (x1 - dx, y0),
            (x1, cy),
            (x1 - dx, y1),
            (x0 + dx, y1),
            (x0, cy),
        ]
        dr.polygon(pts, fill=255)
    else:
        # domyślnie: wypełniony prostokąt (jak 'square')
        dr.rectangle([x0, y0, x1, y1], fill=255)

    return np.asarray(im, dtype=np.uint8)


def load_symbol(name: str, size: Tuple[int, int]) -> np.ndarray:
    """
    Zwraca maskę float32 [0,1] rozmiaru (H,W) z wbudowanego „symbolu”.
    Dostępne: circle, ring, square, triangle, plus/cross, diamond, hex.
    """
    m_u8 = _draw_symbol(name, size)
    m = m_u8.astype(np.float32) / 255.0
    return np.clip(m, 0.0, 1.0, out=m)


# --------------------------------------------------------------------------------------
# Mask stamping / compositing
# --------------------------------------------------------------------------------------

def stamp_mask(dst: np.ndarray, mask: np.ndarray, xy: Tuple[int, int], *, mode: str = "max") -> np.ndarray:
    """
    Wtłacza (nakłada) maskę 'mask' na 'dst' (oba float32 [0,1], shape (H,W)) w pozycji lewego-górnego rogu 'xy'.
    Zwraca NOWĄ maskę (nie modyfikuje dst in-place).
      - mode='max'  : max blend
      - mode='min'  : min blend
      - mode='mean' : uśrednianie
      - mode='mul'  : mnożenie
    Przy maskach wykraczających poza obszar docelowy – odpowiednia część jest obcinana.
    """
    if dst.ndim != 2 or mask.ndim != 2:
        raise ValueError("stamp_mask: dst and mask must be 2D")
    if dst.dtype != np.float32 or mask.dtype != np.float32:
        raise ValueError("stamp_mask: both arrays must be float32 in [0,1]")
    H, W = dst.shape
    h, w = mask.shape
    x, y = int(xy[0]), int(xy[1])

    # prostokąt docelowy
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(W, x + w)
    y1 = min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return dst.copy()

    # odpowiadający region w źródle
    sx0 = x0 - x
    sy0 = y0 - y
    sx1 = sx0 + (x1 - x0)
    sy1 = sy0 + (y1 - y0)

    out = dst.copy()
    d = out[y0:y1, x0:x1]
    s = mask[sy0:sy1, sx0:sx1]

    mode_l = (mode or "max").lower()
    if mode_l == "max":
        d[:] = np.maximum(d, s)
    elif mode_l == "min":
        d[:] = np.minimum(d, s)
    elif mode_l == "mean":
        d[:] = (d + s) * 0.5
    elif mode_l == "mul":
        d[:] = d * s
    else:
        raise ValueError('stamp_mask: mode must be one of {"max","min","mean","mul"}')

    np.clip(out, 0.0, 1.0, out=out)
    return out
