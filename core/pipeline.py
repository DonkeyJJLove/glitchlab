# glitchlab/core/pipeline.py
# -*- coding: utf-8 -*-
"""
Funkcje pipeline:
- load_image / save_image
- load_config (YAML)
- build_ctx: konstruuje Ctx (maski krawędzi + amplitude wg cfg)
- apply_pipeline: odpala kolejne filtry z warstwą zgodności parametrów
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from PIL import Image
import yaml
import inspect

from .utils import Ctx, normalize_image, build_edge_mask, make_amplitude
from . import registry as reg

# -----------------------------------------------------------------------------
# IO
# -----------------------------------------------------------------------------

def load_image(path: str) -> np.ndarray:
    """Wczytaj obraz; zwróć np.ndarray (kanały jak w pliku, dtype uint8)."""
    im = Image.open(path)
    # nie wymuszamy konwersji – normalize_image zrobi resztę, gdy trzeba
    return np.asarray(im)

def save_image(arr: np.ndarray, path: str) -> None:
    """Zapisz obraz (auto-normalizacja do RGB uint8)."""
    a = normalize_image(arr)
    Image.fromarray(a, "RGB").save(path)

def load_config(yaml_text: str) -> Dict[str, Any]:
    """Wczytaj słownik konfiguracyjny z YAML (string)."""
    data = yaml.safe_load(yaml_text) or {}
    if not isinstance(data, dict):
        raise ValueError("YAML must decode to a dict")
    return data

# -----------------------------------------------------------------------------
# Warstwa zgodności starych presetów / parametrów
# -----------------------------------------------------------------------------

# Stare → nowe nazwy parametrów (per-filter)
_COMPAT_PARAM_MAP: Dict[str, Dict[str, str]] = {
    "anisotropic_contour_warp": {
        "amp_px": "use_amp",
        "amp": "use_amp",
        "amplitude": "use_amp",
    },
    "pixel_sort_adaptive": {
        "length": "length_px",
        "threshold_pct": "threshold",
    },
    "spectral_shaper": {
        "low_cut": "low",
        "high_cut": "high",
        "angle": "angle_deg",
        "width": "ang_width",
    },
    "block_mosh_grid": {
        "size_px": "size",
        "prob": "p",
        "posterize": "posterize_bits",
        "jitter": "channel_jitter",
    },
    "phase_glitch": {
        "strength_pct": "strength",
    },
}

# Globalne aliasy nazw parametrów (jeśli filtr akurat taki przyjmuje)
_GLOBAL_PARAM_ALIASES: Dict[str, str] = {
    "mask": "mask_key",
}

def _massage_params(filter_name: str, fn, params: Dict[str, Any],
                    debug_log: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Mapuje stare nazwy parametrów i wycina nieznane pola zgodnie z sygnaturą filtra.
    """
    params = dict(params or {})
    fmap = _COMPAT_PARAM_MAP.get(filter_name, {})
    for old, new in list(fmap.items()):
        if old in params and new not in params:
            params[new] = params.pop(old)

    for old, new in list(_GLOBAL_PARAM_ALIASES.items()):
        if old in params and new not in params:
            params[new] = params.pop(old)

    sig = inspect.signature(fn)
    allowed = {p.name for p in list(sig.parameters.values())[2:]}  # pomiń img, ctx
    cleaned: Dict[str, Any] = {}
    dropped: List[str] = []
    for k, v in params.items():
        if k in allowed:
            cleaned[k] = v
        else:
            dropped.append(k)

    if dropped and debug_log is not None:
        debug_log.append(f"[{filter_name}] dropped params: {', '.join(dropped)}")

    return cleaned

# -----------------------------------------------------------------------------
# Ctx
# -----------------------------------------------------------------------------

def build_ctx(img: np.ndarray,
              seed: int = 7,
              cfg: Optional[Dict[str, Any]] = None) -> Ctx:
    """
    Buduje kontekst przetwarzania:
      - ctx.masks['edge'] – z cfg['edge_mask'] (thresh, dilate, ksize)
      - ctx.amplitude     – z cfg['amplitude'] (kind, strength, itd.)
    """
    cfg = dict(cfg or {})
    a = normalize_image(img)
    H, W = a.shape[:2]

    ctx = Ctx(seed=int(seed))
    ctx.reseed(seed)

    # Edge mask
    edge_cfg = cfg.get("edge_mask") or {}
    try:
        thresh = int(edge_cfg.get("thresh", 60))
        dilate = int(edge_cfg.get("dilate", 0))
        ksize  = int(edge_cfg.get("ksize", 3))
        ctx.masks["edge"] = build_edge_mask(a, thresh=thresh, dilate=dilate, ksize=ksize)
    except Exception:
        # maska krawędzi nie jest obowiązkowa
        pass

    # Amplitude
    amp_cfg = cfg.get("amplitude") or {}
    try:
        kind = str(amp_cfg.get("kind", "none"))
        strength = float(amp_cfg.get("strength", 1.0))
        amp = make_amplitude((H, W), kind=kind, strength=strength, ctx=ctx, **{k:v for k,v in amp_cfg.items() if k not in ("kind","strength")})
        ctx.amplitude = amp
    except Exception:
        ctx.amplitude = None

    return ctx

# -----------------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------------

def apply_pipeline(img: np.ndarray,
                   ctx: Ctx,
                   steps: List[Dict[str, Any]]) -> np.ndarray:
    """
    Odpala kolejne filtry z rejestru. Każdy krok: {"name": ..., "params": {...}}.
    Zastosowana jest warstwa zgodności nazw parametrów.
    """
    out = normalize_image(img)
    for i, step in enumerate(steps or []):
        name = step.get("name")
        params = step.get("params", {}) or {}
        if not name:
            continue

        fn = reg.get(name)
        safe_params = _massage_params(name, fn, params, debug_log=None)
        out = fn(out, ctx, **safe_params)
        # Staramy się utrzymać bufor w „zdrowym” formacie
        try:
            out = normalize_image(out)
        except Exception:
            # jeżeli filtr celowo zwraca floaty 0..1, to tylko przytnijmy
            arr = np.asarray(out)
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            out = arr

    return out
