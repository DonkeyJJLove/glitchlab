# glitchlab/core/symbols.py
# -*- coding: utf-8 -*-
"""
Maski symboliczne (logotypy, znaki wodne, piktogramy) i maski z pól amplitudy.

Funkcje:
- load_mask_image: wczytanie i progowanie maski z pliku (L), opcjonalne dopasowanie rozmiaru.
- register_mask  : rejestracja maski w ctx.masks z polityką łączenia.
- amplitude_as_mask: wytworzenie binarnej maski z pola amplitudy (linear/radial/perlin/mask).

Uwaga: do budowy pól amplitudy wykorzystywany jest make_amplitude z core.utils.
"""

from __future__ import annotations

from typing import Dict, Optional, Literal
import numpy as np
from PIL import Image

from glitchlab.core.utils import make_amplitude


def load_mask_image(
    path: str,
    shape_hw: Optional[tuple[int, int]] = None,
    threshold: int = 128,
    invert: bool = False,
    resample: Literal["nearest", "bilinear"] = "nearest",
) -> np.ndarray:
    """
    Wczytuje maskę z pliku (kanał L), proguje → {0,1}, opcjonalnie skaluje do (H,W).
    Zwraca float32 (0..1).

    Parameters
    ----------
    path : str
        Ścieżka do obrazu (PNG/JPG/BMP). Kanał zostanie zredukowany do L.
    shape_hw : Optional[tuple[int,int]]
        Docelowy rozmiar (H,W); jeśli None, pozostawia oryginał.
    threshold : int
        Próg [0..255] – piksele >= threshold → 1, wpp. 0.
    invert : bool
        Jeśli True, odwraca maskę przed progowaniem.
    resample : {"nearest","bilinear"}
        Metoda skalowania (NEAREST/BILINEAR).

    Returns
    -------
    np.ndarray
        Binarna maska float32 (0..1) o kształcie (H,W).
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

    Polityki łączenia:
    - "max"/"or" : pikselowo max(current, mask)
    - "add"      : sumowanie z przycięciem do 0..1
    - "replace"  : nadpisanie

    Parametry
    ---------
    ctx_masks : Dict[str, np.ndarray]
        Słownik masek w kontekście (np. ctx.masks).
    key : str
        Nazwa pod którą zarejestrować maskę.
    mask : np.ndarray
        Maska 0..1 float32, kształt (H,W).
    merge : {"max","or","add","replace"}
        Strategia łączenia z istniejącą maską (jeśli jest).
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


def amplitude_as_mask(
    shape_hw: tuple[int, int],
    kind: str = "linear_x",
    threshold: float = 0.5,
    invert: bool = False,
    strength: float = 1.0,
    **kwargs,
) -> np.ndarray:
    """
    Buduje binarną maskę z pola amplitudy (linear/radial/perlin/mask).

    Parametry
    ---------
    shape_hw : (int,int)
        Docelowy rozmiar maski (H,W).
    kind : str
        Rodzaj pola (zob. make_amplitude: "none", "linear_x", "linear_y",
        "radial", "perlin", "mask").
    threshold : float
        Próg w [0..1] – wartości >= threshold → 1.
    invert : bool
        Odwrócenie maski po progowaniu.
    strength : float
        Mnożnik pola amplitudy przed progowaniem.
    **kwargs :
        Dodatkowe parametry przekazywane do make_amplitude
        (np. scale, octaves, mask_key itd.).

    Zwraca
    ------
    np.ndarray
        Binarna maska float32 0..1.
    """
    amp = make_amplitude(shape_hw, kind=kind, strength=strength, **kwargs)
    # normalizacja do 0..1 już w make_amplitude, ale upewnijmy się
    if amp.max() > 0:
        amp = (amp - amp.min()) / (amp.max() - amp.min() + 1e-8)
    m = (amp >= float(threshold)).astype(np.float32)
    if invert:
        m = 1.0 - m
    return m


__all__ = ["load_mask_image", "register_mask", "amplitude_as_mask"]
