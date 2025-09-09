# glitchlab/core/pipeline.py
# -*- coding: utf-8 -*-
"""
Silnik przetwarzania glitchlab:
- load_image / save_image  → I/O z normalizacją do RGB uint8
- load_config(yaml_text)   → parser YAML
- normalize_preset(cfg)    → akceptuje format A (root) i B (sekcja nazwana)
- build_ctx(arr, seed, cfg)→ tworzy Ctx, maskę krawędzi i pole amplitudy
- apply_pipeline(arr, ctx, steps, ...) → odpala sekwencję filtrów z rejestrem

Kontrakt filtra:  fn(img, ctx, **params) → (H,W,3) uint8
"""

from __future__ import annotations

import io
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import yaml

from .utils import (
    Ctx,
    normalize_image,
    build_edge_mask,
    make_amplitude,
)
from .registry import get as _get_filter


# =============================================================================
# I/O obrazów
# =============================================================================

def load_image(path: str) -> np.ndarray:
    """
    Wczytuje obraz i zwraca (H,W,3) uint8 (RGB). Obsługuje PNG/JPG/BMP/WEBP itd.
    """
    with Image.open(path) as im:
        im = im.convert("RGB")
        arr = np.asarray(im, dtype=np.uint8)
    return normalize_image(arr)


def save_image(arr: np.ndarray, path: str) -> None:
    """
    Zapisuje obraz (automatycznie normalizuje do (H,W,3) uint8).
    """
    arr = normalize_image(arr)
    im = Image.fromarray(arr, mode="RGB")
    im.save(path)


# =============================================================================
# YAML / PRESETY
# =============================================================================

def load_config(yaml_text: str) -> Dict[str, Any]:
    """
    Parsuje YAML do dict. Nie narzuca formatu A/B – to robi normalize_preset().
    """
    data = yaml.safe_load(io.StringIO(yaml_text))
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError("Config YAML must parse to a mapping (dict).")
    return data


def normalize_preset(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Przyjmuje dowolny z dwóch formatów presetów i zwraca znormalizowany dict,
    który posiada klucze root-level (np. 'steps', 'edge_mask', 'amplitude').
      A) { steps: [...] , edge_mask: {...}, amplitude: {...} }
      B) { some_name: { steps: [...], ... } }  → zwróci inner dict
    """
    if "steps" in cfg or "edge_mask" in cfg or "amplitude" in cfg:
        return cfg
    if len(cfg) == 1 and isinstance(next(iter(cfg.values())), dict):
        return next(iter(cfg.values()))
    # nie znając struktury – zwracamy pustą konfigurację kroków
    return {"steps": []}


# =============================================================================
# Budowa kontekstu
# =============================================================================

def build_ctx(arr: np.ndarray, seed: int = 7, cfg: Optional[Dict[str, Any]] = None) -> Ctx:
    """
    Tworzy Ctx na podstawie obrazu i (opcjonalnie) sekcji 'edge_mask'/'amplitude'.
    """
    arr = normalize_image(arr)
    H, W = arr.shape[:2]
    ctx = Ctx(seed=int(seed))
    ctx.reseed(seed)
    ctx.meta["original"] = arr.copy()  # przydatne np. dla protect_edges

    cfg = cfg or {}
    cfg = normalize_preset(cfg)

    # Edge mask (opcjonalnie)
    edge_cfg = cfg.get("edge_mask")
    if isinstance(edge_cfg, dict):
        thresh = float(edge_cfg.get("thresh", 60))
        dilate = int(edge_cfg.get("dilate", 0))
        ksize = int(edge_cfg.get("ksize", 3))
        ctx.masks["edge"] = build_edge_mask(arr, thresh=thresh, dilate=dilate, ksize=ksize).astype(np.float32)

    # Amplitude field (opcjonalnie)
    amp_cfg = cfg.get("amplitude")
    if isinstance(amp_cfg, dict):
        kind = str(amp_cfg.get("kind", "none"))
        strength = float(amp_cfg.get("strength", 1.0))
        # przekazujemy resztę parametrów (np. scale, octaves, mask_key...)
        params = {k: v for k, v in amp_cfg.items() if k not in ("kind", "strength")}
        ctx.amplitude = make_amplitude((H, W), kind=kind, strength=strength, ctx=ctx, **params)

    return ctx


# =============================================================================
# Główny silnik kroków
# =============================================================================

def apply_pipeline(arr: np.ndarray,
                   ctx: Ctx,
                   steps: List[Dict[str, Any]],
                   fail_fast: bool = True,
                   debug_log: Optional[List[str]] = None) -> np.ndarray:
    """
    Odpala filtry zdefiniowane w `steps` (lista dictów: {'name': ..., 'params': {...}}).

    - fail_fast=True  → w razie błędu przerywa (RuntimeError z kontekstem: krok/nazwa/params)
      fail_fast=False → pomija niesprawny krok (loguje do debug_log) i idzie dalej
    - debug_log: lista stringów; jeżeli podana – wpisujemy tam zdarzenia kroków i czasy.

    Zwraca (H,W,3) uint8.
    """
    out = normalize_image(arr)
    steps = steps or []

    for idx, spec in enumerate(steps):
        t0 = time.time()
        if not isinstance(spec, dict):
            raise RuntimeError(f"Step {idx}: step spec must be a mapping, got {type(spec).__name__}")

        name = spec.get("name")
        params = spec.get("params", {}) or {}
        if not isinstance(name, str) or not name:
            raise RuntimeError(f"Step {idx}: missing 'name'")
        if not isinstance(params, dict):
            raise RuntimeError(f"Step {idx}: 'params' must be a mapping (dict)")

        # filtry mogą honorować 'enabled: false'
        enabled = spec.get("enabled", True)
        if not enabled:
            if debug_log is not None:
                debug_log.append(f"[skip] {idx}:{name}")
            continue

        try:
            fn = _get_filter(name)
        except KeyError as e:
            msg = f"Step {idx} '{name}': {e}"
            if fail_fast:
                raise RuntimeError(msg) from e
            if debug_log is not None:
                debug_log.append("[unknown] " + msg)
            continue

        try:
            out = fn(out, ctx, **params)
            out = normalize_image(out)
        except Exception as e:
            tb = traceback.format_exc(limit=1)
            msg = f"Step {idx} '{name}' failed: {e} | params={params} | at={tb.strip()}"
            if fail_fast:
                raise RuntimeError(msg) from e
            if debug_log is not None:
                debug_log.append("[error] " + msg)
            # w trybie nie-fail-fast: pomijamy, zachowując poprzedni 'out'

        dt = (time.time() - t0) * 1000.0
        if debug_log is not None:
            debug_log.append(f"[ok] {idx}:{name}  {dt:.1f} ms")

    return normalize_image(out)
