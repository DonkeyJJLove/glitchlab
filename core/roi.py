# glitchlab/core/roi.py
"""
---
version: 2
kind: module
id: "core-roi"
created_at: "2025-09-11"
name: "glitchlab.core.roi"
author: "GlitchLab v2"
role: "ROI Primitives & Mask Ops"
description: >
  Generuje podstawowe maski ROI (polygon/rect/circle) jako float32 [0..1] o kształcie (H,W),
  z opcjonalnym featheringiem (BoxBlur). Udostępnia też operacje łączenia masek
  (max/min/mean/mul). Czyste, deterministyczne prymitywy bez SciPy/OpenCV.

inputs:
  shape_hw: {type: "(int,int)", desc: "rozmiar obrazu: (H, W)"}
  polygon.points?: {type: "list[(x:float,y:float)]", min: 3, desc: "wierzchołki poligonu"}
  rect.xyxy?: {type: "(int,int,int,int)", desc: "lewy-górny i prawy-dolny róg: (x0,y0,x1,y1)"}
  circle.center?: {type: "(int,int)", desc: "środek okręgu (cx, cy)"}
  circle.radius?: {type: "int", min: 0, desc: "promień w px"}
  feather: {type: "int", default: 0, min: 0, desc: "promień wygładzania BoxBlur (px)"}
  merge.op?: {enum: ["max","min","mean","mul"], desc: "sposób łączenia masek"}
  merge.masks?: {type: "list[np.ndarray]", desc: "maski float32 [0..1] o tym samym kształcie"}

outputs:
  mask: {dtype: "float32", shape: "(H,W)", range: "[0,1]", desc: "wynikowa maska ROI"}
  merged?: {dtype: "float32", shape: "(H,W)", range: "[0,1]", desc: "wynik łączenia masek"}

interfaces:
  exports: ["mask_polygon","mask_rect","mask_circle","merge_masks"]
  depends_on: ["numpy","Pillow"]
  used_by: ["glitchlab.core.pipeline","glitchlab.filters","glitchlab.gui"]

contracts:
  - "maski są typu float32 i mieszczą się w [0,1]"
  - "funkcje czyste (nie mutują wejść), deterministyczne"
  - "feather realizowany przez Pillow.ImageFilter.BoxBlur"
  - "merge_masks wymaga spójnych kształtów i dtype float32"

constraints:
  - "no SciPy/OpenCV"
  - "wymiary (H,W) muszą być dodatnie"
  - "polygon wymaga >=3 punktów; rect.x1>x0, rect.y1>y0; radius>=0"

hud:
  note: "Moduł nie zapisuje telemetrii; maski są konsumowane przez pipeline/filtry."

tests_smoke:
  - "mask_rect((64,64),(8,8,40,40),feather=2) → shape (64,64), dtype f32, minmax∈[0,1]"
  - "mask_circle((32,48),(16,16),10) → wartości >0 wewnątrz, 0 poza"
  - "merge_masks('max', m1, m2) → shape = m1.shape, range [0,1]"

license: "Proprietary"
---
"""

from __future__ import annotations

from typing import Iterable, List, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

__all__ = [
    "mask_polygon",
    "mask_rect",
    "mask_circle",
    "merge_masks",
]

FloatMask = np.ndarray  # float32, shape (H, W), values in [0,1]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _ensure_shape(shape_hw: Tuple[int, int]) -> Tuple[int, int]:
    if not (isinstance(shape_hw, tuple) and len(shape_hw) == 2):
        raise ValueError("shape_hw must be a (H, W) tuple")
    H, W = int(shape_hw[0]), int(shape_hw[1])
    if H <= 0 or W <= 0:
        raise ValueError("shape_hw must be positive")
    return H, W


def _feather_l(mask_u8: np.ndarray, feather: int) -> np.ndarray:
    """Feather via Pillow BoxBlur (fast, deterministic)."""
    if feather <= 0:
        return mask_u8
    im = Image.fromarray(mask_u8, mode="L")
    im = im.filter(ImageFilter.BoxBlur(radius=int(feather)))
    return np.asarray(im, dtype=np.uint8)


def _to_float01(mask_u8: np.ndarray) -> FloatMask:
    m = mask_u8.astype(np.float32) / 255.0
    # stabilize numerical range
    np.clip(m, 0.0, 1.0, out=m)
    return m


def _draw_binary_polygon(H: int, W: int, points: List[Tuple[float, float]]) -> np.ndarray:
    im = Image.new("L", (W, H), 0)
    if len(points) >= 3:
        ImageDraw.Draw(im).polygon(points, fill=255, outline=None)
    return np.asarray(im, dtype=np.uint8)


def _draw_binary_rect(H: int, W: int, xyxy: Tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = map(int, xyxy)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((H, W), dtype=np.uint8)
    im = Image.new("L", (W, H), 0)
    ImageDraw.Draw(im).rectangle([x0, y0, x1 - 1, y1 - 1], fill=255, outline=None)
    return np.asarray(im, dtype=np.uint8)


def _draw_binary_circle(H: int, W: int, center: Tuple[int, int], radius: int) -> np.ndarray:
    cx, cy = map(int, center)
    r = max(0, int(radius))
    if r == 0:
        return np.zeros((H, W), dtype=np.uint8)
    x0, y0 = cx - r, cy - r
    x1, y1 = cx + r, cy + r
    im = Image.new("L", (W, H), 0)
    ImageDraw.Draw(im).ellipse([x0, y0, x1, y1], fill=255, outline=None)
    return np.asarray(im, dtype=np.uint8)


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def mask_polygon(shape_hw: Tuple[int, int], points: Iterable[Tuple[float, float]], *, feather: int = 0) -> FloatMask:
    """
    Zwraca maskę poligonową (float32 [0,1]) o kształcie (H,W).
    - shape_hw: (H, W)
    - points: iterowalne (x, y) co najmniej 3 wierzchołki
    - feather: promień wygładzania (px), BoxBlur
    """
    H, W = _ensure_shape(shape_hw)
    pts = [(float(x), float(y)) for (x, y) in points]
    if len(pts) < 3:
        return np.zeros((H, W), dtype=np.float32)
    m_u8 = _draw_binary_polygon(H, W, pts)
    m_u8 = _feather_l(m_u8, feather)
    return _to_float01(m_u8)


def mask_rect(shape_hw: Tuple[int, int], xyxy: Tuple[int, int, int, int], *, feather: int = 0) -> FloatMask:
    """
    Zwraca maskę prostokątną (float32 [0,1]).
    - xyxy: (x0, y0, x1, y1) w pikselach; przedział lewostronnie domknięty, prawostronnie otwarty
    - feather: promień wygładzania (px), BoxBlur
    """
    H, W = _ensure_shape(shape_hw)
    m_u8 = _draw_binary_rect(H, W, xyxy)
    m_u8 = _feather_l(m_u8, feather)
    return _to_float01(m_u8)


def mask_circle(shape_hw: Tuple[int, int], center: Tuple[int, int], radius: int, *, feather: int = 0) -> FloatMask:
    """
    Zwraca maskę kołową (float32 [0,1]).
    - center: (cx, cy), radius: w pikselach
    - feather: promień wygładzania (px), BoxBlur
    """
    H, W = _ensure_shape(shape_hw)
    m_u8 = _draw_binary_circle(H, W, center, radius)
    m_u8 = _feather_l(m_u8, feather)
    return _to_float01(m_u8)


def merge_masks(op: str, *masks: FloatMask) -> FloatMask:
    """
    Łączy maski (float32 [0,1]) operacją:
      - "max"  : punktowy max
      - "min"  : punktowe min
      - "mean" : średnia arytmetyczna
      - "mul"  : mnożenie punktowe
    Zwraca float32 [0,1].
    """
    if not masks:
        raise ValueError("merge_masks: at least one mask required")
    ref_shape = masks[0].shape
    for m in masks:
        if m.shape != ref_shape:
            raise ValueError("merge_masks: all masks must share the same shape")
        if m.dtype != np.float32:
            raise ValueError("merge_masks: masks must be float32 in [0,1]")

    op_l = (op or "").lower()
    if op_l == "max":
        out = np.maximum.reduce(masks)
    elif op_l == "min":
        out = np.minimum.reduce(masks)
    elif op_l == "mean":
        out = np.mean(np.stack(masks, axis=0), axis=0).astype(np.float32)
    elif op_l == "mul":
        out = np.ones_like(masks[0], dtype=np.float32)
        for m in masks:
            out *= m
    else:
        raise ValueError('merge_masks: op must be one of {"max","min","mean","mul"}')

    np.clip(out, 0.0, 1.0, out=out)
    return out
