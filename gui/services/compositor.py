# gui/services/compositor.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional, Literal, List, Tuple
import numpy as np

BlendMode = Literal[
    "normal", "multiply", "screen", "overlay", "add", "subtract", "darken", "lighten"
]


# ───────────────────────── helpers: types & ranges ─────────────────────────

def _ensure_rgb_u8(img: np.ndarray) -> np.ndarray:
    """
    Zwraca obraz w formacie uint8 HxWx3.
    Akceptuje:
      - uint8 HxWx3,
      - uint8/float gray HxW,
      - uint8/float HxWx1,
      - uint8/float HxWx4 (premultiply alpha),
      - float (0..1) dla dowolnego z powyższych.
    """
    a = np.asarray(img)

    if a.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D array, got ndim={a.ndim}")

    # ujednolicenie typu/liczby kanałów
    if a.dtype != np.uint8:
        a = a.astype(np.float32)
        # jeśli wygląda jak 0..1 to przeskaluj
        if a.max() <= 1.001:
            a = a * 255.0
        a = np.clip(a, 0, 255).astype(np.uint8)

    if a.ndim == 2:
        return np.stack([a, a, a], axis=-1)

    # 3D
    if a.shape[2] == 1:
        return np.repeat(a, 3, axis=-1)

    if a.shape[2] == 3:
        return a

    if a.shape[2] == 4:
        # premultiply prosto: RGB * A
        rgb = a[..., :3].astype(np.float32)
        alpha = (a[..., 3:4].astype(np.float32) / 255.0)
        out = np.clip(rgb * alpha, 0.0, 255.0).astype(np.uint8)
        return out

    raise ValueError(f"Unsupported channel count: {a.shape}")


def _to_float01(img_u8_rgb: np.ndarray) -> np.ndarray:
    if img_u8_rgb.dtype != np.uint8 or img_u8_rgb.ndim != 3 or img_u8_rgb.shape[2] != 3:
        raise TypeError("Expected uint8 HxWx3 array.")
    return img_u8_rgb.astype(np.float32) / 255.0


def _to_u8(img_f01: np.ndarray) -> np.ndarray:
    return np.clip(img_f01 * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)


def _mask_to_f01(mask: Optional[np.ndarray], h: int, w: int) -> np.ndarray:
    """
    Zwraca maskę float32 HxWx1 w zakresie 0..1. Jeśli mask=None → same jedynki.
    Gdy rozmiar się nie zgadza, maska zostanie dopasowana *po stronie managera*,
    tutaj zakładamy zgodność lub broadcast (HxW/HxWx1).
    """
    if mask is None:
        return np.ones((h, w, 1), dtype=np.float32)

    m = np.asarray(mask)
    if m.ndim == 2:
        m = m[..., None]
    elif not (m.ndim == 3 and m.shape[2] == 1):
        raise ValueError("Mask must be HxW or HxWx1")

    m = m.astype(np.float32)
    if m.max() > 1.001:
        m = m / 255.0
    m = np.clip(m, 0.0, 1.0)

    if m.shape[0] != h or m.shape[1] != w:
        # Ostrożnie: zostawiamy dopasowanie po LayerManager; tu tylko sanity-check.
        # Jeśli rozmiar inny ale możliwy broadcast (np. 1x1x1) — NumPy poradzi sobie.
        pass

    return m


# ───────────────────────── blending ─────────────────────────

def _apply_blend(base: np.ndarray, top: np.ndarray, mode: BlendMode) -> np.ndarray:
    """
    base, top: float32 w zakresie 0..1
    Zwraca 'blended(base, top)' bez uwzględniania opacity/maski.
    """
    if mode == "normal":
        return top
    if mode == "multiply":
        return base * top
    if mode == "screen":
        return 1.0 - (1.0 - base) * (1.0 - top)
    if mode == "overlay":
        lo = 2.0 * base * top
        hi = 1.0 - 2.0 * (1.0 - base) * (1.0 - top)
        return np.where(base <= 0.5, lo, hi)
    if mode == "add":
        return np.clip(base + top, 0.0, 1.0)
    if mode == "subtract":
        return np.clip(base - top, 0.0, 1.0)
    if mode == "darken":
        return np.minimum(base, top)
    if mode == "lighten":
        return np.maximum(base, top)
    # fallback
    return top


# ───────────────────────── main ─────────────────────────

def composite_stack(layers: List[Tuple[np.ndarray, float, BlendMode, Optional[np.ndarray]]]) -> np.ndarray:
    """
    layers: lista krotek (img_u8, opacity, blend_mode, mask)
      - img_u8: HxWx3 uint8 (inne formaty zostaną skonwertowane),
      - opacity: 0..1,
      - blend_mode: jw.,
      - mask: None, HxW lub HxWx1, wartości 0..1 lub 0..255.

    Zwraca RGB uint8 HxWx3.
    """
    if not layers:
        raise ValueError("composite_stack(): no layers provided")

    # Ujednolić rozmiar/scalę 1. warstwy
    base_u8, base_op, base_mode, base_mask = layers[0]
    base_u8 = _ensure_rgb_u8(base_u8)
    h, w = base_u8.shape[:2]

    # tło (czarne) w float
    out = _to_float01(base_u8)

    # opacity/maska pierwszej warstwy (nad czarnym tłem!)
    alpha0 = float(max(0.0, min(1.0, base_op)))
    m0 = _mask_to_f01(base_mask, h, w)  # HxWx1
    a0 = alpha0 * m0  # HxWx1
    out = out * a0  # nad czarnym → (1-a0)*0 znika

    # kolejne warstwy
    for img_u8, opacity, mode, mask in layers[1:]:
        top_u8 = _ensure_rgb_u8(img_u8)

        if top_u8.shape[:2] != (h, w):
            # Dla bezpieczeństwa — dopasowanie wymiarów nie wchodzi w zakres kompozytora.
            # To powinien zapewnić LayerManager. Jeśli jednak różne – rzuć czytelny błąd.
            raise ValueError(f"All layers must have same size. Got {(h, w)} vs {top_u8.shape[:2]}.")

        top = _to_float01(top_u8)
        blended = _apply_blend(out, top, mode)

        op = float(max(0.0, min(1.0, opacity)))
        m = _mask_to_f01(mask, h, w)  # HxWx1
        a = op * m

        # klasyczne mieszanie: out = out*(1-a) + blended*a
        out = (1.0 - a) * out + a * blended

    # sanity
    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
    return _to_u8(np.clip(out, 0.0, 1.0))
