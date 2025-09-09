# glitchlab/filters/pixel_sort_adaptive.py
# -*- coding: utf-8 -*-
"""
pixel_sort_adaptive — sortowanie pikseli w segmentach wyzwalanych maską/gradientem.

Pomysł: najpierw wyznaczamy *trigger* (gdzie sortować), potem w tych przedziałach
sortujemy po wybranym kluczu (luma/hue/sat/R/G/B). Długość segmentu i szansa
zależą od amplitudy (ctx.amplitude) i RNG kontekstu.

Parametry:
    direction   : 'vertical' | 'horizontal'
    trigger     : 'edges' | 'luma' | 'mask'
    threshold   : float [0..1]      (dla 'edges'/'luma')
    mask_key    : str|None          (dla trigger='mask')
    length_px   : int               (bazowa długość segmentu)
    length_gain : float             (ile mnożyć przez (1+amp))
    prob        : float [0..1]      (szansa aktywacji segmentu)
    key         : 'luma' | 'r' | 'g' | 'b' | 'sat' | 'hue'
    reverse     : bool              (odwrócić porządek)

Diagnostyka:
    ctx.cache["pixel_sort/trigger"] — binarna mapa (uint8)
"""

from __future__ import annotations
import numpy as np
from glitchlab.core.registry import register
from glitchlab.core.utils import compute_edges, resize_mask_to

def _luma(a):
    return (0.299 * a[...,0] + 0.587 * a[...,1] + 0.114 * a[...,2]).astype(np.float32)

def _rgb_to_hsv(a):
    # a: uint8
    arr = a.astype(np.float32) / 255.0
    r, g, b = arr[...,0], arr[...,1], arr[...,2]
    cmax = np.max(arr, axis=-1)
    cmin = np.min(arr, axis=-1)
    delta = cmax - cmin + 1e-8

    h = np.zeros_like(cmax)
    mask = (delta > 0)
    idx = (cmax == r) & mask
    h[idx] = ((g[idx]-b[idx]) / delta[idx]) % 6
    idx = (cmax == g) & mask
    h[idx] = (b[idx]-r[idx]) / delta[idx] + 2
    idx = (cmax == b) & mask
    h[idx] = (r[idx]-g[idx]) / delta[idx] + 4
    h = (h / 6.0) % 1.0

    s = delta / (cmax + 1e-8)
    v = cmax
    return h, s, v

@register("pixel_sort_adaptive", schema={
    "direction": {"enum": ["vertical", "horizontal"]},
    "trigger": {"enum": ["edges", "luma", "mask"]},
    "threshold": {"type": "float", "min": 0.0, "max": 1.0, "step": 0.01},
    "length_px": {"type": "int", "min": 1, "max": 2048, "step": 1},
    "length_gain": {"type": "float", "min": 0.0, "max": 4.0, "step": 0.05},
    "prob": {"type": "float", "min": 0.0, "max": 1.0, "step": 0.01},
    "key": {"enum": ["luma", "r", "g", "b", "sat", "hue"]},
    "reverse": {"type": "bool"},
})
def pixel_sort_adaptive(
    img, ctx,
    direction: str = "vertical",
    trigger: str = "edges",
    threshold: float = 0.35,
    mask_key: str | None = None,
    length_px: int = 160,
    length_gain: float = 1.0,
    prob: float = 1.0,
    key: str = "luma",
    reverse: bool = False,
):
    a = img.astype(np.uint8, copy=False)
    h, w, _ = a.shape

    # --- amplituda
    amp = getattr(ctx, "amplitude", None)
    if amp is None:
        amp = np.ones((h, w), dtype=np.float32)
    else:
        amp = amp.astype(np.float32)
        if (amp.max() - amp.min()) > 1e-8:
            amp = (amp - amp.min()) / (amp.max() - amp.min() + 1e-8)

    # --- trigger
    if trigger == "edges":
        trig = (compute_edges(a, gain=1.2) >= float(threshold)).astype(np.uint8)
    elif trigger == "luma":
        L = _luma(a) / 255.0
        trig = ((L >= float(threshold)) | (L <= 1.0 - float(threshold))).astype(np.uint8)
    elif trigger == "mask" and isinstance(mask_key, str) and hasattr(ctx, "masks") and mask_key in ctx.masks:
        m = ctx.masks[mask_key]
        if m.shape != (h, w):
            m = resize_mask_to(m, (h, w))
        trig = (m >= float(threshold)).astype(np.uint8)
    else:
        trig = np.ones((h, w), dtype=np.uint8)

    ctx.cache["pixel_sort/trigger"] = (trig * 255).astype(np.uint8)

    # --- klucz sortowania
    if key == "luma":
        key_mat = _luma(a)
    elif key == "r":
        key_mat = a[...,0].astype(np.float32)
    elif key == "g":
        key_mat = a[...,1].astype(np.float32)
    elif key == "b":
        key_mat = a[...,2].astype(np.float32)
    else:
        H, S, V = _rgb_to_hsv(a)
        key_mat = {"sat": S*255.0, "hue": H*255.0}.get(key, _luma(a))

    out = a.copy()
    rng = getattr(ctx, "rng", np.random.default_rng(7))

    def sort_line(line_pixels, line_keys, line_trig, line_amp):
        n = line_pixels.shape[0]
        i = 0
        while i < n:
            if line_trig[i] == 0 or rng.random() > float(prob):
                i += 1
                continue
            # długość segmentu sterowana amplitudą (średnia w oknie)
            L = int(length_px * (1.0 + float(length_gain) * float(np.mean(line_amp[max(0, i-2): min(n, i+3)]))))
            L = max(4, min(L, n - i))
            j = i + L
            # sortuj po kluczu w [i:j]
            idx = np.argsort(line_keys[i:j])
            if reverse:
                idx = idx[::-1]
            segment = line_pixels[i:j][idx]
            out_line[i:j] = segment
            i = j

    if direction == "vertical":
        # każda kolumna osobno
        for x in range(w):
            line = out[:, x, :]
            out_line = out[:, x, :]
            sort_line(line, key_mat[:, x], trig[:, x], amp[:, x])
            out[:, x, :] = out_line
    else:
        # każda linia
        for y in range(h):
            line = out[y, :, :]
            out_line = out[y, :, :]
            sort_line(line, key_mat[y, :], trig[y, :], amp[y, :])
            out[y, :, :] = out_line

    return out.astype(np.uint8)
