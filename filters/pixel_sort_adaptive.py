# -*- coding: utf-8 -*-
"""
pixel_sort_adaptive — sortowanie pikseli w segmentach wyzwalanych maską/gradientem.

Parametry:
    direction   : 'vertical' | 'horizontal'
    trigger     : 'edges' | 'luma' | 'mask'
    threshold   : float [0..1]      (dla 'edges'/'luma'/'mask')
    mask_key    : str|None          (dla trigger='mask')
    length_px   : int               (bazowa długość segmentu)
    length_gain : float             (ile mnożyć przez (1+amp))
    prob        : float [0..1]      (szansa aktywacji segmentu)
    key         : 'luma' | 'r' | 'g' | 'b' | 'sat' | 'hue'
    reverse     : bool              (odwrócić porządek)

Diag:
    ctx.cache["pixel_sort/trigger"] — binarna mapa (uint8)
"""

from __future__ import annotations
import numpy as np
from typing import Any, Dict

try:
    from glitchlab.core.registry import register
except Exception:  # pragma: no cover
    from core.registry import register  # type: ignore

DOC = "Adaptacyjne pixel-sorting w segmentach: wyzwalanie edges/luma/mask; sterowanie amplitudą; HSV klucze."
DEFAULTS: Dict[str, Any] = {
    "direction": "vertical",
    "trigger": "edges",       # edges|luma|mask
    "threshold": 0.35,
    "mask_key": None,
    "length_px": 160,
    "length_gain": 1.0,
    "prob": 1.0,
    "key": "luma",            # luma|r|g|b|sat|hue
    "reverse": False,
}

def _fit_hw(m: np.ndarray, H: int, W: int) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    mh, mw = m.shape[:2]
    if (mh, mw) == (H, W):
        return np.clip(m, 0.0, 1.0)
    ys = np.linspace(0, mh - 1, H).astype(np.int32)
    xs = np.linspace(0, mw - 1, W).astype(np.int32)
    out = m[ys][:, xs]
    return np.clip(out, 0.0, 1.0)

def _sobel_mag01(a_u8: np.ndarray) -> np.ndarray:
    # szybki Sobel magnitude 3x3 na luminancji → [0,1]
    a = a_u8.astype(np.float32) / 255.0
    g = (0.299*a[...,0] + 0.587*a[...,1] + 0.114*a[...,2]).astype(np.float32)
    Kx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=np.float32)
    Ky = np.array([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=np.float32)
    pad = np.pad(g, 1, mode="reflect")
    H, W = g.shape
    gx = np.zeros_like(g, dtype=np.float32)
    gy = np.zeros_like(g, dtype=np.float32)
    for i in range(3):
        for j in range(3):
            sl = pad[i:i+H, j:j+W]
            gx += sl * Kx[i, j]
            gy += sl * Ky[i, j]
    mag = np.sqrt(gx*gx + gy*gy) * (1.0/8.0)
    return np.clip(mag, 0.0, 1.0)

def _luma_u8(a_u8: np.ndarray) -> np.ndarray:
    return (0.299*a_u8[...,0] + 0.587*a_u8[...,1] + 0.114*a_u8[...,2]).astype(np.float32)

def _rgb_to_hsv_u8(a_u8: np.ndarray):
    arr = a_u8.astype(np.float32) / 255.0
    r, g, b = arr[...,0], arr[...,1], arr[...,2]
    cmax = np.max(arr, axis=-1); cmin = np.min(arr, axis=-1)
    d = cmax - cmin + 1e-8
    h = np.zeros_like(cmax)
    m = d > 0
    i = (cmax == r) & m; h[i] = ((g[i]-b[i]) / d[i]) % 6
    i = (cmax == g) & m; h[i] = (b[i]-r[i]) / d[i] + 2
    i = (cmax == b) & m; h[i] = (r[i]-g[i]) / d[i] + 4
    h = (h / 6.0) % 1.0
    s = d / (cmax + 1e-8)
    v = cmax
    return h, s, v

@register("pixel_sort_adaptive", defaults=DEFAULTS, doc=DOC)
def pixel_sort_adaptive(img: np.ndarray, ctx, **p: Any) -> np.ndarray:
    a = img.astype(np.uint8, copy=False)
    H, W, C = a.shape
    if C < 3:
        raise ValueError("pixel_sort_adaptive: expected RGB-like image (H,W,C>=3)")

    direction   = str(p.get("direction",   DEFAULTS["direction"])).lower()
    trigger     = str(p.get("trigger",     DEFAULTS["trigger"])).lower()
    threshold   = float(np.clip(p.get("threshold", DEFAULTS["threshold"]), 0.0, 1.0))
    mask_key    = p.get("mask_key", DEFAULTS["mask_key"])
    length_px   = int(max(4, int(p.get("length_px", DEFAULTS["length_px"]))))
    length_gain = float(p.get("length_gain", DEFAULTS["length_gain"]))
    prob        = float(np.clip(p.get("prob", DEFAULTS["prob"]), 0.0, 1.0))
    key_name    = str(p.get("key", DEFAULTS["key"])).lower()
    reverse     = bool(p.get("reverse", DEFAULTS["reverse"]))

    # amplituda → [0,1]
    amp = getattr(ctx, "amplitude", None)
    if amp is None:
        amp = np.ones((H, W), dtype=np.float32)
    else:
        amp = amp.astype(np.float32)
        rng = amp.max() - amp.min()
        if rng > 1e-8:
            amp = (amp - amp.min()) / (rng + 1e-8)
        else:
            amp = np.ones((H, W), dtype=np.float32)

    # trigger map
    if trigger == "edges":
        trig = (_sobel_mag01(a) >= threshold).astype(np.uint8)
    elif trigger == "luma":
        L = _luma_u8(a) / 255.0
        trig = ((L >= threshold) | (L <= 1.0 - threshold)).astype(np.uint8)
    elif trigger == "mask" and isinstance(mask_key, str) and getattr(ctx, "masks", None) and (mask_key in ctx.masks):
        m = ctx.masks[mask_key]
        if m.shape != (H, W):
            m = _fit_hw(m, H, W)
        trig = (np.asarray(m, dtype=np.float32) >= threshold).astype(np.uint8)
    else:
        trig = np.ones((H, W), dtype=np.uint8)

    # diag
    try:
        ctx.cache["pixel_sort/trigger"] = (trig * 255).astype(np.uint8)
    except Exception:
        pass

    # key matrix
    if key_name == "luma":
        key_mat = _luma_u8(a)
    elif key_name == "r":
        key_mat = a[...,0].astype(np.float32)
    elif key_name == "g":
        key_mat = a[...,1].astype(np.float32)
    elif key_name == "b":
        key_mat = a[...,2].astype(np.float32)
    else:
        Hh, Ss, Vv = _rgb_to_hsv_u8(a)
        if key_name == "sat":
            key_mat = Ss * 255.0
        elif key_name == "hue":
            key_mat = Hh * 255.0
        else:
            key_mat = _luma_u8(a)

    out = a.copy()
    rng = getattr(ctx, "rng", np.random.default_rng(int(getattr(ctx, "seed", 7))))

    def sort_line(line_pixels: np.ndarray, line_keys: np.ndarray, line_trig: np.ndarray, line_amp: np.ndarray) -> np.ndarray:
        n = line_pixels.shape[0]
        result = line_pixels.copy()
        i = 0
        while i < n:
            if line_trig[i] == 0 or rng.random() > prob:
                i += 1
                continue
            # długość segmentu zależna od amplitudy (okoliczne średnie)
            amp_win = float(np.mean(line_amp[max(0, i-2): min(n, i+3)]))
            L = int(max(4, min(n - i, round(length_px * (1.0 + length_gain * amp_win)))))
            j = i + L
            idx = np.argsort(line_keys[i:j])
            if reverse:
                idx = idx[::-1]
            result[i:j] = line_pixels[i:j][idx]
            i = j
        return result

    if direction == "vertical":
        for x in range(W):
            out[:, x, :] = sort_line(out[:, x, :], key_mat[:, x], trig[:, x], amp[:, x])
    else:
        for y in range(H):
            out[y, :, :] = sort_line(out[y, :, :], key_mat[y, :], trig[y, :], amp[y, :])

    return out.astype(np.uint8)
