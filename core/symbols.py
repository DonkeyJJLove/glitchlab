# glitchlab/core/symbols.py
# -*- coding: utf-8 -*-
"""
Maski symboliczne (logotypy, znaki wodne, piktogramy):
- wczytywanie obrazu maski (L), progowanie i (opcjonalnie) dopasowanie do rozmiaru
- rejestracja w ctx.masks z polityką łączenia
"""

from __future__ import annotations
from typing import Dict, Optional, Literal
import numpy as np
from PIL import Image


def load_mask_image(
    path: str,
    shape_hw: Optional[tuple[int, int]] = None,
    threshold: int = 128,
    invert: bool = False,
    resample: Literal["nearest", "bilinear"] = "nearest",
) -> np.ndarray:
    """
    Wczytuje maskę z pliku (kanał L), próg → {0,1}, ew. skaluje do (H,W).
    Zwraca float32 (0..1).
    """
    im = Image.open(path).convert("L")
    if shape_hw is not None:
        H, W = shape_hw
        im = im.resize((W, H), Image.NEAREST if resample == "nearest" else Image.BILINEAR)
    arr = np.asarray(im, dtype=np.uint8).astype(np.float32)
    if invert:
        arr = 255.0 - arr
    mask = (arr >= float(threshold)).astype(np.float32)
    return mask


def register_mask(
    ctx_masks: Dict[str, np.ndarray],
    key: str,
    mask: np.ndarray,
    merge: Literal["max", "or", "add", "replace"] = "max",
) -> None:
    """
    Rejestruje maskę w ctx.masks pod `key`.
    - max/or  : max(current, mask)
    - add     : sum + clamp(0..1)
    - replace : nadpisz
    """
    mask = np.clip(mask.astype(np.float32), 0.0, 1.0)
    if key in ctx_masks and merge != "replace":
        cur = np.clip(ctx_masks[key].astype(np.float32), 0.0, 1.0)
        if merge in ("max", "or"):
            ctx_masks[key] = np.maximum(cur, mask)
        elif merge == "add":
            ctx_masks[key] = np.clip(cur + mask, 0.0, 1.0)
        else:
            ctx_masks[key] = mask
    else:
        ctx_masks[key] = mask


__all__ = ["load_mask_image", "register_mask"]
