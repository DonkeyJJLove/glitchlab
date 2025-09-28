# glitchlab/filters/default_identity.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from typing import Any, Dict, Optional

try:
    from glitchlab.core.registry import register  # normal route
except Exception:  # pragma: no cover
    from core.registry import register  # type: ignore

DOC = (
    "Default diagnostic filter: identity/gray/edges/edges_overlay/"
    "mask_overlay/amp_overlay/r/g/b. Emits basic HUD diagnostics."
)
DEFAULTS: Dict[str, Any] = {
    "mode": "identity",          # identity|gray|edges|edges_overlay|mask_overlay|amp_overlay|r|g|b
    "strength": 1.0,             # 0..2 (overlay intensity / mix)
    "mask_key": None,            # optional ROI key
    "use_amp": 1.0,              # float|bool - amplitude modulation
    "clamp": True,
    "edge_ksize": 3,             # 3|5 ; 5 ≈ wstępny blur + sobel 3x3
}

def _to_f32(img_u8: np.ndarray) -> np.ndarray:
    return img_u8.astype(np.float32) / 255.0

def _to_u8(img_f32: np.ndarray) -> np.ndarray:
    x = np.clip(img_f32, 0.0, 1.0)
    return (x * 255.0 + 0.5).astype(np.uint8)

def _gray(x: np.ndarray) -> np.ndarray:
    # x in [0,1], shape (H,W,3) -> (H,W)
    return (0.299 * x[...,0] + 0.587 * x[...,1] + 0.114 * x[...,2]).astype(np.float32)

def _gray3(x: np.ndarray) -> np.ndarray:
    g = np.clip(_gray(x), 0.0, 1.0)
    return np.stack([g, g, g], axis=-1)

def _box_blur3(g: np.ndarray) -> np.ndarray:
    # szybki 3x3 blur pudełkowy (bez SciPy)
    H, W = g.shape
    pad = np.pad(g, 1, mode="edge")
    out = np.zeros_like(g, dtype=np.float32)
    for i in range(3):
        for j in range(3):
            out += pad[i:i+H, j:j+W]
    out *= (1.0 / 9.0)
    return out

def _sobel_mag_gray(g: np.ndarray) -> np.ndarray:
    # sobel na obrazie szarości g in [0,1]
    H, W = g.shape
    Kx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=np.float32)
    Ky = np.array([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=np.float32)
    pad = np.pad(g, 1, mode="edge")
    gx = np.zeros_like(g, dtype=np.float32)
    gy = np.zeros_like(g, dtype=np.float32)
    for i in range(3):
        for j in range(3):
            sl = pad[i:i+H, j:j+W]
            gx += sl * Kx[i, j]
            gy += sl * Ky[i, j]
    mag = np.sqrt(gx*gx + gy*gy) * (1.0/8.0)
    return np.clip(mag, 0.0, 1.0)

def _edges_mag(x: np.ndarray, ksize: int) -> np.ndarray:
    g = _gray(x)
    if int(ksize) == 5:
        g = _box_blur3(_box_blur3(g))  # przybliżenie 5x5
    mag = _sobel_mag_gray(g)
    return mag

@register("default_identity", defaults=DEFAULTS, doc=DOC)
def default_identity(img_u8: np.ndarray, ctx, **p) -> np.ndarray:
    """
    Minimal-yet-useful default filter that always works; emits diagnostics for HUD.
    Supports mask and amplitude overlays without SciPy/OpenCV.
    """
    mode = str(p.get("mode", "identity")).lower()
    strength = float(p.get("strength", 1.0))
    use_amp = p.get("use_amp", 1.0)
    mask_key = p.get("mask_key", None)
    clamp = bool(p.get("clamp", True))
    edge_ksize = int(p.get("edge_ksize", 3))

    x = _to_f32(img_u8)
    H, W = x.shape[:2]

    # amplitude map
    amp = getattr(ctx, "amplitude", None)
    if amp is not None:
        if isinstance(use_amp, (int, float)):
            amp_map = np.clip(amp.astype(np.float32) * float(use_amp), 0.0, 1.0)
        elif isinstance(use_amp, bool) and use_amp:
            amp_map = np.clip(amp.astype(np.float32), 0.0, 1.0)
        else:
            amp_map = np.ones((H, W), dtype=np.float32)
    else:
        amp_map = np.ones((H, W), dtype=np.float32)

    # optional mask
    m = None
    if mask_key and isinstance(mask_key, str):
        mm = ctx.masks.get(mask_key)
        if isinstance(mm, np.ndarray) and mm.shape[:2] == (H, W):
            m = np.clip(mm.astype(np.float32), 0.0, 1.0)

    # diagnostics (HUD)
    gray3 = _gray3(x)
    ctx.cache["diag/default/gray"] = _to_u8(gray3)
    try:
        edges = _edges_mag(x, edge_ksize)
        ctx.cache["diag/default/edges"] = _to_u8(np.stack([edges, edges, edges], axis=-1))
    except Exception:
        pass
    ctx.cache["diag/default/amplitude"] = _to_u8(np.stack([amp_map, amp_map, amp_map], axis=-1))
    if m is not None:
        ctx.cache["diag/default/mask"] = _to_u8(np.stack([m, m, m], axis=-1))

    # select effect
    eff = x
    if mode == "identity":
        eff = x
    elif mode == "gray":
        eff = gray3
    elif mode == "edges":
        e = _edges_mag(x, edge_ksize)
        eff = np.stack([e, e, e], axis=-1)
    elif mode == "edges_overlay":
        e = _edges_mag(x, edge_ksize)
        alpha = np.clip(strength, 0.0, 2.0) * e[..., None]
        # zielony overlay krawędzi
        green = np.zeros_like(x); green[...,1] = 1.0
        eff = x * (1.0 - alpha) + green * alpha
        ctx.cache["diag/default/alpha"] = _to_u8(np.stack([e, e, e], axis=-1))
    elif mode == "mask_overlay":
        local = m if m is not None else (np.ones((H, W), dtype=np.float32) * 0.25)
        alpha = np.clip(strength, 0.0, 2.0) * amp_map * local
        red = np.zeros_like(x); red[...,0] = 1.0
        eff = x * (1.0 - alpha[..., None]) + red * (alpha[..., None])
        ctx.cache["diag/default/alpha"] = _to_u8(np.stack([alpha, alpha, alpha], axis=-1))
    elif mode == "amp_overlay":
        a3 = np.stack([amp_map, amp_map, amp_map], axis=-1)
        alpha = np.clip(strength, 0.0, 2.0) * 0.5
        eff = x * (1.0 - alpha) + a3 * alpha
        ctx.cache["diag/default/alpha"] = _to_u8(np.stack([amp_map, amp_map, amp_map], axis=-1))
    elif mode in ("r", "g", "b"):
        idx = {"r":0, "g":1, "b":2}[mode]
        chan = np.clip(x[..., idx], 0.0, 1.0)
        eff = np.stack([chan, chan, chan], axis=-1)
    else:
        eff = x

    # If mask provided and we didn't use it inside, blend now (ROI)
    if m is not None and mode not in ("mask_overlay",):
        eff = x*(1.0 - m[...,None]) + eff*(m[...,None])

    out = _to_u8(eff) if clamp else (eff*255.0).astype(np.uint8)
    return out
