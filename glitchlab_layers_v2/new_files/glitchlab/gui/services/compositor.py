# glitchlab/gui/services/compositor.py
# Pure numpy compositor for layer blending (2D RGB uint8).
from __future__ import annotations
from typing import Optional, Literal
import numpy as np

BlendMode = Literal["normal","multiply","screen","overlay","add","subtract","darken","lighten"]

def _to_float(img_u8: np.ndarray) -> np.ndarray:
    if img_u8.dtype != np.uint8:
        raise TypeError(f"Expected uint8 image, got {img_u8.dtype}")
    if img_u8.ndim != 3 or img_u8.shape[2] != 3:
        raise ValueError(f"Expected (H,W,3) RGB, got {img_u8.shape}")
    return img_u8.astype(np.float32) / 255.0

def _to_u8(img_f: np.ndarray) -> np.ndarray:
    return np.clip(img_f * 255.0 + 0.5, 0, 255).astype(np.uint8)

def _apply_blend(base: np.ndarray, top: np.ndarray, mode: BlendMode) -> np.ndarray:
    # base/top: float32 [0,1], shape (H,W,3)
    if mode == "normal":
        return top
    if mode == "multiply":
        return base * top
    if mode == "screen":
        return 1.0 - (1.0 - base) * (1.0 - top)
    if mode == "overlay":
        mask = base <= 0.5
        lo = 2.0 * base * top
        hi = 1.0 - 2.0 * (1.0 - base) * (1.0 - top)
        return np.where(mask, lo, hi)
    if mode == "add":
        return np.clip(base + top, 0.0, 1.0)
    if mode == "subtract":
        return np.clip(base - top, 0.0, 1.0)
    if mode == "darken":
        return np.minimum(base, top)
    if mode == "lighten":
        return np.maximum(base, top)
    # default
    return top

def composite_stack(layers: list[tuple[np.ndarray, float, BlendMode, Optional[np.ndarray]]]) -> np.ndarray:
    """Composite layers from bottom to top.
    Each item: (img_u8, opacity in [0..1], blend_mode, mask_f32_or_None)
    Returns: np.uint8 RGB composite.
    """
    assert layers, "composite_stack(): no layers provided"
    base_u8, op, mode, mask = layers[0]
    out = _to_float(base_u8)

    # Apply initial opacity (kept for conceptual clarity).
    if op < 1.0:
        out = (1.0 - (1.0 - op)) * out + (1.0 - op) * out

    for img_u8, opacity, mode, mask in layers[1:]:
        top = _to_float(img_u8)
        blended = _apply_blend(out, top, mode)
        if mask is not None:
            if mask.ndim == 2:
                mask3 = np.clip(mask[..., None].astype(np.float32), 0.0, 1.0)
            elif mask.ndim == 3 and mask.shape[2] == 1:
                mask3 = np.clip(mask.astype(np.float32), 0.0, 1.0)
            else:
                raise ValueError("Mask must be HxW (float32 [0..1]) or HxWx1")
        else:
            mask3 = 1.0
        out = (1.0 - mask3 * opacity) * out + (mask3 * opacity) * blended
    return _to_u8(out)
