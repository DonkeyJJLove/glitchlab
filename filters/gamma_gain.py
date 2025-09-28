# glitchlab/filters/gamma_gain.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from typing import Any, Dict
try:
    from glitchlab.core.registry import register
    from glitchlab.core.utils import to_gray_f32_u8
except Exception:  # pragma: no cover
    from core.registry import register
    from core.utils import to_gray_f32_u8

DOC = "Simple tonemap: per-channel gamma with optional amplitude & mask modulation."
DEFAULTS: Dict[str, Any] = {
    "gamma": 1.0,        # >1 ciemniej, <1 jaśniej
    "mask_key": None,
    "use_amp": 1.0,
    "clamp": True,
}

@register("gamma_gain", defaults=DEFAULTS, doc=DOC)
def gamma_gain(img_u8: np.ndarray, ctx, **p) -> np.ndarray:
    g = float(p.get("gamma", 1.0))
    use_amp = p.get("use_amp", 1.0)
    mask_key = p.get("mask_key", None)
    clamp = bool(p.get("clamp", True))

    x = img_u8.astype(np.float32) / 255.0
    H, W = x.shape[:2]

    amp = getattr(ctx, "amplitude", None)
    if amp is None:
        amp_map = np.ones((H,W), np.float32)
    else:
        amp_map = np.clip(amp.astype(np.float32), 0.0, 1.0)
        if isinstance(use_amp, (int,float)):
            amp_map = np.clip(amp_map * float(use_amp), 0.0, 1.0)
        elif not use_amp:
            amp_map[:] = 1.0

    m = None
    if isinstance(mask_key, str):
        mm = ctx.masks.get(mask_key)
        if isinstance(mm, np.ndarray) and mm.shape[:2] == (H,W):
            m = np.clip(mm.astype(np.float32), 0.0, 1.0)

    # efektywna gamma (1 ↔ brak zmiany), modulowana amp/maską
    eff = np.clip(g, 0.05, 5.0)
    # mieszanie: przy amp=0 lub mask=0 → brak zmiany
    if m is not None:
        w = amp_map * m
    else:
        w = amp_map
    # podnieś do potęgi tylko część sygnału (lerp: x -> x^eff)
    y = x**eff
    out = x*(1.0 - w[...,None]) + y*(w[...,None])

    out = np.clip(out, 0.0, 1.0)
    out_u8 = (out*255.0 + 0.5).astype(np.uint8) if clamp else (out*255.0).astype(np.uint8)

    # diagnostyki
    ctx.cache["diag/gamma_gain/amp"] = (np.stack([amp_map]*3, -1)*255+0.5).astype(np.uint8)
    if m is not None:
        ctx.cache["diag/gamma_gain/mask"] = (np.stack([m]*3, -1)*255+0.5).astype(np.uint8)
    return out_u8
