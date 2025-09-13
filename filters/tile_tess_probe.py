# glitchlab/filters/tile_tess_probe.py
# -*- coding: utf-8 -*-
"""
tile_tess_probe — diagnostyka parkietażu/kafelkowania (tesselacji).
Wykrywa okres w pikselach X/Y (autokorelacja 1D agregowana) i wykonuje akcje:
 - 'overlay_grid'  : półprzezroczysta siatka na oryginale,
 - 'phase_paint'   : kolorowanie fazy (R=x%Px, G=y%Py),
 - 'avg_tile'      : estymata kafla i rekonstrukcja,
 - 'quilt'         : delikatne przetasowanie kafli.
Diag do ctx.cache:
   diag/tess/fft_mag  : log|F| (downsample 256),
   diag/tess/acf_x    : linia ACF_x (256x64),
   diag/tess/acf_y    : linia ACF_y (256x64),
   diag/tess/template : kafel-średnia (avg_tile),
   diag/tess/period_xy: [py, px].
Obsługa mask_key: miks ROI w przestrzeni obrazu po wyliczeniu efektu.
"""

from __future__ import annotations
import numpy as np
from typing import Any, Dict, Tuple, Optional

try:
    from glitchlab.core.registry import register
except Exception:  # pragma: no cover
    from core.registry import register  # type: ignore

DOC = "Tessellation probe: wykrywa okres (X/Y) i wizualizuje mozaikę siatką/fazą/rekonstrukcją; wspiera ROI mask."
DEFAULTS: Dict[str, Any] = {
    "mode": "overlay_grid",     # overlay_grid | phase_paint | avg_tile | quilt
    "min_period": 4,
    "max_period": 256,
    "method": "acf",            # acf | fft
    "alpha": 0.5,               # krycie overlayu / miks z rekonstrukcją
    "grid_thickness": 1,        # px
    "quilt_jitter": 2,          # px (max odchył w quilt)
    "use_amp": 1.0,             # wpływ ctx.amplitude (skalowanie alpha/miksu)
    "mask_key": None,           # ROI (miks po policzeniu efektu)
    "clamp": True,
}

def _to_f32(u8: np.ndarray) -> np.ndarray:
    return u8.astype(np.float32) / 255.0

def _to_u8(f32: np.ndarray) -> np.ndarray:
    x = np.clip(f32, 0.0, 1.0)
    return (x * 255.0 + 0.5).astype(np.uint8)

def _downsample_2d(x: np.ndarray, tgt: int = 256) -> np.ndarray:
    H, W = x.shape
    s0 = max(1, H // tgt); s1 = max(1, W // tgt)
    return x[::s0, ::s1]

def _fft2c(x: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft2(x))

def _fit_hw(m: np.ndarray, H: int, W: int) -> np.ndarray:
    """Nearest-neighbor bez depsów — dopasuj maskę do (H,W)."""
    m = np.asarray(m, dtype=np.float32)
    mh, mw = m.shape[:2]
    if (mh, mw) == (H, W):
        return np.clip(m, 0.0, 1.0)
    ys = (np.linspace(0, mh - 1, H)).astype(np.int32)
    xs = (np.linspace(0, mw - 1, W)).astype(np.int32)
    return np.clip(m[ys][:, xs], 0.0, 1.0)

def _acorr_1d_line(line: np.ndarray) -> np.ndarray:
    f = np.fft.rfft(line - line.mean())
    acf = np.fft.irfft((f * np.conj(f))).real
    return acf

def _detect_period_1d(img: np.ndarray, axis: int, min_p: int, max_p: int, method: str) -> int:
    prof = img.mean(axis=0) if axis == 1 else img.mean(axis=1)
    prof = prof.astype(np.float32)
    N = prof.shape[0]
    lo = int(np.clip(min_p, 2, N-1))
    hi = int(np.clip(max_p, lo+1, N-1))
    if method == "fft":
        F = np.abs(np.fft.rfft(prof))
        F[0] = 0.0
        idx = np.argmax(F[1:]) + 1
        period = int(round(N / max(1, idx)))
    else:
        acf = _acorr_1d_line(prof)
        seg = acf[lo:hi].astype(np.float32)
        off = int(np.argmax(seg))
        period = lo + off
    return int(np.clip(period, lo, hi))

def _grid_overlay(x: np.ndarray, px: int, py: int, alpha: float, thick: int) -> np.ndarray:
    H, W, _ = x.shape
    g = np.zeros((H, W), dtype=np.float32)
    if px > 0:
        g[:, ::max(1, px)] = 1.0
        for t in range(1, int(thick)):
            if t < px: g[:, t::px] = 1.0
    if py > 0:
        g[::max(1, py), :] = 1.0
        for t in range(1, int(thick)):
            if t < py: g[t::py, :] = 1.0
    g3 = np.stack([g, g, g], axis=-1)
    return np.clip(x * (1.0 - alpha) + g3 * alpha, 0.0, 1.0)

def _phase_paint(x: np.ndarray, px: int, py: int, alpha: float) -> np.ndarray:
    H, W, _ = x.shape
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    rx = (xx % max(1, px)) / float(max(1, px))
    ry = (yy % max(1, py)) / float(max(1, py))
    phase_rgb = np.stack([rx, ry, np.zeros_like(rx)], axis=-1)
    return np.clip(x * (1.0 - alpha) + phase_rgb * alpha, 0.0, 1.0)

def _avg_tile_recon(a: np.ndarray, px: int, py: int):
    H, W, C = a.shape
    px = max(1, int(px)); py = max(1, int(py))
    H2 = (H // py) * py
    W2 = (W // px) * px
    if H2 <= 0 or W2 <= 0:
        return a, a
    tiles = a[:H2, :W2].reshape(H2//py, py, W2//px, px, C)
    templ = tiles.mean(axis=(0,2))
    recon = np.tile(templ, (H2//py, W2//px, 1))
    out = a.copy()
    out[:H2, :W2] = recon
    return out, templ

def _quilt_shuffle(a: np.ndarray, px: int, py: int, jitter: int, rng) -> np.ndarray:
    H, W, C = a.shape
    nx = max(1, W // max(1, px))
    ny = max(1, H // max(1, py))
    out = a.copy()
    for yi in range(ny):
        for xi in range(nx):
            y0, x0 = yi*py, xi*px
            y1, x1 = min(H, y0+py), min(W, x0+px)
            dx = int(rng.integers(-jitter, jitter+1)) if jitter>0 else 0
            dy = int(rng.integers(-jitter, jitter+1)) if jitter>0 else 0
            yy = np.clip(np.arange(y0, y1)+dy, 0, H-1)
            xx = np.clip(np.arange(x0, x1)+dx, 0, W-1)
            out[y0:y1, x0:x1, :] = out[np.ix_(yy, xx)]
    return out

def _emit_diag_fft(ctx, gray: np.ndarray):
    try:
        mag = np.log1p(np.abs(_fft2c(gray)))
        vis = _downsample_2d(mag)
        vis = (255.0 * vis / (vis.max() + 1e-8)).astype(np.uint8)
        ctx.cache["diag/tess/fft_mag"] = vis
    except Exception:
        pass

def _emit_diag_acf(ctx, acfx: np.ndarray, acfy: np.ndarray):
    try:
        def _line_to_img(line):
            L = line.shape[0]
            W = 256
            H = 64
            xx = np.linspace(0, L-1, W)
            yl = np.interp(xx, np.arange(L), (line - line.min())/(line.max()-line.min()+1e-8))
            img = np.zeros((H, W), dtype=np.float32)
            for x in range(W):
                h = int(yl[x] * (H-1))
                img[H-1-h:, x] = 1.0
            u8 = (img*255+0.5).astype(np.uint8)
            return np.stack([u8,u8,u8], axis=-1)
        ctx.cache["diag/tess/acf_x"] = _line_to_img(acfx)
        ctx.cache["diag/tess/acf_y"] = _line_to_img(acfy)
    except Exception:
        pass

@register("tile_tess_probe", defaults=DEFAULTS, doc=DOC)
def tile_tess_probe(img: np.ndarray, ctx, **p) -> np.ndarray:
    a = img.astype(np.uint8, copy=False)
    H, W, _ = a.shape
    rng = getattr(ctx, "rng", np.random.default_rng(7))

    mode   = str(p.get("mode", DEFAULTS["mode"])).lower()
    min_p  = int(p.get("min_period", DEFAULTS["min_period"]))
    max_p  = int(p.get("max_period", DEFAULTS["max_period"]))
    method = str(p.get("method", DEFAULTS["method"])).lower()
    alpha  = float(np.clip(p.get("alpha", DEFAULTS["alpha"]), 0.0, 1.0))
    thick  = int(max(1, p.get("grid_thickness", DEFAULTS["grid_thickness"])))
    jitter = int(max(0, p.get("quilt_jitter", DEFAULTS["quilt_jitter"])))
    use_amp= p.get("use_amp", DEFAULTS["use_amp"])
    clamp  = bool(p.get("clamp", DEFAULTS["clamp"]))
    mkey   = p.get("mask_key", DEFAULTS["mask_key"])

    x = _to_f32(a)
    gray = (0.299*x[...,0] + 0.587*x[...,1] + 0.114*x[...,2]).astype(np.float32)

    _emit_diag_fft(ctx, gray)

    px = _detect_period_1d(gray, axis=1, min_p=min_p, max_p=max_p, method=method)
    py = _detect_period_1d(gray, axis=0, min_p=min_p, max_p=max_p, method=method)

    acfx = _acorr_1d_line(gray.mean(axis=0).astype(np.float32))
    acfy = _acorr_1d_line(gray.mean(axis=1).astype(np.float32))
    _emit_diag_acf(ctx, acfx, acfy)

    amp_scale = 1.0
    if hasattr(ctx, "amplitude") and ctx.amplitude is not None:
        amp = ctx.amplitude.astype(np.float32)
        amp_scale = float(0.25 + 0.75 * amp.mean())
    if isinstance(use_amp, (int, float)):
        amp_scale *= float(use_amp)

    alpha_eff = np.clip(alpha * amp_scale, 0.0, 1.0)

    # efekt bazowy (full-frame)
    if mode == "overlay_grid":
        eff = _grid_overlay(x, px, py, alpha_eff, thick)
    elif mode == "phase_paint":
        eff = _phase_paint(x, px, py, alpha_eff)
    elif mode == "avg_tile":
        recon, templ = _avg_tile_recon(x, px, py)
        try: ctx.cache["diag/tess/template"] = _to_u8(np.clip(templ,0,1))
        except Exception: pass
        eff = x*(1.0 - alpha_eff) + recon*alpha_eff
    elif mode == "quilt":
        out = _quilt_shuffle(_to_u8(x), px, py, jitter, rng)
        eff = out.astype(np.float32) / 255.0
    else:
        eff = x

    # miks maską ROI (jeśli podano mask_key)
    if isinstance(mkey, str) and getattr(ctx, "masks", None) and (mkey in ctx.masks):
        m = ctx.masks[mkey]
        if m.shape[:2] != (H, W):
            m = _fit_hw(m, H, W)
        m = np.clip(m, 0.0, 1.0).astype(np.float32)[..., None]
        eff = eff * m + x * (1.0 - m)

    out = _to_u8(eff) if clamp else (np.clip(eff,0,1)*255.0).astype(np.uint8)

    try:
        ctx.cache["diag/tess/period_xy"] = np.array([py, px], dtype=np.int32)
    except Exception:
        pass
    return out
