# glitchlab/filters/rgb_offset_2.py
"""
---
version: 2
kind: filter
id: "filter-rgb-offset"
created_at: "2025-09-11"
name: "glitchlab.filters.rgb_offset"
author: "GlitchLab v2"
role: "RGB Channel Offset Filter"
description: >
  Cyklicznie przesuwa kanały R/G/B o zadane wektory (dx, dy). Wspiera maskowanie
  przez ctx.masks[mask_key] i modulację amplitudą ctx.amplitude (use_amp). Wejście/wyjście: uint8 RGB.

inputs:
  image:  {dtype: "uint8", shape: "(H,W,3)", desc: "obraz wejściowy RGB"}
  ctx:    {type: "Ctx", desc: "RNG, amplitude (H×W, f32 0..1), masks, cache/meta"}
params:
  r:        {type: "(int,int)", default: [1, 0],  desc: "przesunięcie kanału R (dx, dy)"}
  g:        {type: "(int,int)", default: [-1, 0], desc: "przesunięcie kanału G (dx, dy)"}
  b:        {type: "(int,int)", default: [2, 0],  desc: "przesunięcie kanału B (dx, dy)"}
  mask_key: {type: "str|null", default: null,     desc: "klucz maski w ctx.masks"}
  use_amp:  {type: "float|bool", default: 1.0,    desc: "mnożnik modulacji amplitudą (False/0.0 wyłącza)"}
  clamp:    {type: "bool", default: true,         desc: "przycięcie 0..1 i konwersja do uint8"}

outputs:
  image: {dtype: "uint8", shape: "(H,W,3)", desc: "obraz po przesunięciu kanałów"}

interfaces:
  exports:    ["rgb_offset"]
  depends_on: ["numpy","Pillow"]
  used_by:    ["glitchlab.core.pipeline","glitchlab.gui"]

contracts:
  - "wejście: uint8 RGB; przetwarzanie: float32 [0..1]; wyjście: uint8 RGB"
  - "deterministyczne dla danego obrazu/parametrów (RNG nieużywany)"
  - "mask blend i amplitude blend: m ∈ [0,1], rozmiar dopasowany do (H,W)"
  - "diagnostyka do ctx.cache: diag/rgb_offset/mask, diag/rgb_offset/amp"

hud:
  channels:
    diag_mask: "diag/rgb_offset/mask"
    diag_amp:  "diag/rgb_offset/amp"
    stage_io:  "stage/{i}/in|out|diff (zapewniane przez pipeline)"

constraints:
  - "no SciPy/OpenCV"
  - "bez pętli per-piksel (używaj np.roll/operacji wektorowych)"

defaults:
  r: [1, 0]
  g: [-1, 0]
  b: [2, 0]
  mask_key: null
  use_amp: 1.0
  clamp: true

tests_smoke:
  - "np.zeros((128,128,3),uint8) + 40 → kształt (H,W,3), dtype uint8; brak wyjątków"
  - "use_amp=0.0 → wynik ≈ wejście"
license: "Proprietary"
---
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, Tuple

from glitchlab.core.registry import register
from glitchlab.core.pipeline import Ctx  # dataclass z pipeline.py

FILTER_NAME = "rgb_offset"

DEFAULTS: Dict[str, Any] = {
    "r": (1, 0),         # (dx, dy) dla kanału R
    "g": (-1, 0),        # (dx, dy) dla kanału G
    "b": (2, 0),         # (dx, dy) dla kanału B
    "mask_key": None,    # wspólne
    "use_amp": 1.0,      # wspólne
    "clamp": True,       # wspólne
}

DOC = """Per-channel RGB channel shift (cyclic). Rolls each channel by (dx, dy).
Blending honors optional ctx.masks[mask_key] and ctx.amplitude (use_amp).
Input: uint8 RGB, internal float32 [0..1], output: uint8 RGB."""


def _as_pair(v: Any) -> Tuple[int, int]:
    if isinstance(v, (tuple, list)) and len(v) == 2:
        return int(v[0]), int(v[1])
    if isinstance(v, (int, float)):
        return int(v), 0
    raise ValueError("offset must be (dx, dy) or int")


def _blend(img: np.ndarray, fx: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # img, fx: float32 [0..1] RGB; mask: float32 [0..1] (H,W)
    if mask is None:
        return fx
    m = mask.astype(np.float32)
    if m.ndim == 2:
        m = np.clip(m, 0.0, 1.0)[:, :, None]
    return img * (1.0 - m) + fx * m


@register(FILTER_NAME, defaults=DEFAULTS, doc=DOC)
def rgb_offset(img: np.ndarray, ctx: Ctx, **params) -> np.ndarray:
    """
    v2 filter: (img, ctx, **params) -> np.ndarray (uint8 RGB)
    wspólne: mask_key/use_amp/clamp
    """
    if not (isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[-1] == 3):
        raise ValueError("rgb_offset: expected uint8 RGB array (H,W,3)")
    H, W, _ = img.shape

    # scal parametry z DEFAULTS
    p = {**DEFAULTS, **(params or {})}
    r = _as_pair(p.get("r"))
    g = _as_pair(p.get("g"))
    b = _as_pair(p.get("b"))
    mask_key = p.get("mask_key")
    use_amp = p.get("use_amp")
    clamp = bool(p.get("clamp", True))

    # wejście → float32 [0..1]
    if img.dtype == np.uint8:
        base = img.astype(np.float32) / 255.0
    else:
        base = np.clip(img.astype(np.float32, copy=False), 0.0, 1.0)

    # efekt: per-channel roll (cykliczny)
    eff = base.copy()
    for c, (dx, dy) in enumerate([r, g, b]):
        if dx or dy:
            ch = eff[..., c]
            if dy:
                ch = np.roll(ch, dy, axis=0)
            if dx:
                ch = np.roll(ch, dx, axis=1)
            eff[..., c] = ch

    # maska z ctx (opcjonalna)
    m = None
    if mask_key:
        m_candidate = ctx.masks.get(mask_key)
        if m_candidate is not None:
            # zakładamy już w (H,W) float32 [0..1]
            if m_candidate.shape != (H, W):
                # prosty, szybki resize przez Pillow jeśli jest; inaczej nearest
                from PIL import Image
                im = Image.fromarray((np.clip(m_candidate, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), mode="L")
                im = im.resize((W, H), resample=Image.BICUBIC)
                m = np.asarray(im, dtype=np.float32) / 255.0
            else:
                m = np.clip(m_candidate.astype(np.float32, copy=False), 0.0, 1.0)

    # amplituda globalna z ctx (opcjonalna)
    amp = ctx.amplitude if getattr(ctx, "amplitude", None) is not None else None
    if isinstance(use_amp, (float, int)) and amp is not None:
        scale = float(use_amp)
        if scale > 0.0:
            a = np.clip(amp.astype(np.float32, copy=False) * scale, 0.0, 1.0)
            m = a if m is None else np.clip(m * a, 0.0, 1.0)
        else:
            # use_amp == 0 ⇒ brak efektu (blend do zera)
            m = np.zeros((H, W), dtype=np.float32)
    elif use_amp is False:
        m = np.zeros((H, W), dtype=np.float32)

    # blend
    out_f = _blend(base, eff, m)

    # diagnostyki
    if m is not None:
        ctx.cache[f"diag/{FILTER_NAME}/mask"] = m
    if amp is not None:
        ctx.cache[f"diag/{FILTER_NAME}/amp"] = amp

    # wyjście
    if clamp:
        out_f = np.clip(out_f, 0.0, 1.0)
    return (out_f * 255.0 + 0.5).astype(np.uint8)
