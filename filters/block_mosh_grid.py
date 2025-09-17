# glitchlab/filters/block_mosh_grid.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from typing import Any, Dict, Tuple

try:
    from glitchlab.core.registry import register
except Exception:  # pragma: no cover
    from core.registry import register  # type: ignore

DOC = (
    "Block Mosh (grid): losowe przestawianie bloków. "
    "Tryby shift/swap/shift+swap, opcjonalna rotacja (k*90°), channel jitter, posterize, maska i amplitude. "
    "Emitowane diagnostyki: diag/bmg_select, diag/bmg_dx, diag/bmg_dy, diag/bmg_alpha."
)

DEFAULTS: Dict[str, Any] = {
    "size": 24,              # px
    "p": 0.35,               # prawdopodobieństwo działania na blok
    "max_shift": 16,         # px (dla trybu 'shift')
    "mode": "shift",         # 'shift' | 'swap' | 'shift+swap'
    "swap_radius": 2,        # zasięg (w blokach) dla 'swap'
    "rot_p": 0.0,            # prawdopodobieństwo rotacji (90/180/270) bloku po operacji
    "wrap": True,            # True: roll wewnątrz bloku; False: klamrowanie w obrębie bloku
    "channel_jitter": 0.0,   # dodatkowy +/- jitter per kanał (px; całkowity)
    "posterize_bits": 0,     # 0 = off; 1..7 = redukcja bitów
    "mix": 1.0,              # 0..1 blend z oryginałem
    "mask_key": None,        # ROI
    "use_amp": 1.0,          # 0..2 typowo; bool => 1.0/0.0
    "amp_influence": 1.0,    # skala wpływu amplitude na p i siłę
    "clamp": True,           # końcowe przycięcie do u8
}

# ----------------------------- utils --------------------------------

def _to_f32(u8: np.ndarray) -> np.ndarray:
    return u8.astype(np.float32) / 255.0

def _to_u8(f32: np.ndarray) -> np.ndarray:
    x = np.clip(f32, 0.0, 1.0)
    return (x * 255.0 + 0.5).astype(np.uint8)

def _fit_hw(m: np.ndarray, H: int, W: int) -> np.ndarray:
    mh, mw = m.shape[:2]
    out = np.zeros((H, W), dtype=np.float32)
    h = min(H, mh); w = min(W, mw)
    out[:h, :w] = m[:h, :w].astype(np.float32)
    if h < H: out[h:, :w] = out[h-1:h, :w]
    if w < W: out[:H, w:] = out[:H, w-1:w]
    return out

def _resolve_mask(ctx, mask_key: str | None, H: int, W: int) -> np.ndarray:
    m = None
    if mask_key and getattr(ctx, "masks", None):
        m = ctx.masks.get(mask_key)
    if m is None:
        return np.ones((H, W), dtype=np.float32)
    m = np.asarray(m).astype(np.float32)
    if m.shape != (H, W): m = _fit_hw(m, H, W)
    return np.clip(m, 0.0, 1.0)

def _amplitude_weight_map(ctx, H: int, W: int, use_amp) -> np.ndarray:
    if not hasattr(ctx, "amplitude") or ctx.amplitude is None:
        return np.ones((H, W), dtype=np.float32)
    amp = np.asarray(ctx.amplitude).astype(np.float32)
    if amp.shape != (H, W): amp = _fit_hw(amp, H, W)
    amp -= amp.min(); amp /= (amp.max() + 1e-12)
    base = 0.25 + 0.75 * amp
    if isinstance(use_amp, bool):
        return base if use_amp else np.ones((H, W), dtype=np.float32)
    return base * float(max(0.0, use_amp))

def _posterize_u8(patch_u8: np.ndarray, bits: int) -> np.ndarray:
    if bits <= 0: return patch_u8
    keep = int(np.clip(bits, 1, 7))
    levels = 2 ** keep
    step = 256 // levels
    return (patch_u8 // step) * step

def _shift_patch(patch: np.ndarray, dx: int, dy: int, wrap: bool) -> np.ndarray:
    """
    Shift 2D lub 3D patch (H,W[,C]) w px. Gdy wrap=False, brzegi dociągamy „clamp” w obrębie patcha.
    """
    if dx == 0 and dy == 0:
        return patch.copy()
    if wrap:
        out = np.roll(patch, dy, axis=0)
        out = np.roll(out, dx, axis=1)
        return out
    H, W = patch.shape[:2]
    out = patch.copy()
    y_src0 = max(0, -dy); y_src1 = min(H, H - dy)
    x_src0 = max(0, -dx); x_src1 = min(W, W - dx)
    y_dst0 = max(0,  dy); y_dst1 = min(H, H + dy)
    x_dst0 = max(0,  dx); x_dst1 = min(W, W + dx)
    if y_src0 < y_src1 and x_src0 < x_src1:
        out_slice = (slice(y_dst0, y_dst1), slice(x_dst0, x_dst1))
        src_slice = (slice(y_src0, y_src1), slice(x_src0, x_src1))
        if patch.ndim == 3:
            out[out_slice[0], out_slice[1], :] = patch[src_slice[0], src_slice[1], :]
        else:
            out[out_slice] = patch[src_slice]
    return out

def _jitter_channels(patch: np.ndarray, jitter: float, rng, wrap: bool) -> np.ndarray:
    if jitter <= 0: return patch
    J = int(round(abs(jitter)))
    if J == 0: return patch
    if patch.ndim == 2:
        return _shift_patch(patch, int(rng.integers(-J, J+1)), int(rng.integers(-J, J+1)), wrap)
    out = patch.copy()
    for ch in range(min(3, out.shape[2])):
        dx = int(rng.integers(-J, J+1))
        dy = int(rng.integers(-J, J+1))
        out[..., ch] = _shift_patch(out[..., ch], dx, dy, wrap)  # 2D slice
    return out

def _pad_to_square(patch: np.ndarray, wrap: bool) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Dopaduj patch do kwadratu (max(H,W)). Zwraca (patch_pad, (t,b,l,r))."""
    h, w = patch.shape[:2]
    if h == w:
        return patch, (0, 0, 0, 0)
    side = max(h, w)
    t = (side - h) // 2
    b = side - h - t
    l = (side - w) // 2
    r = side - w - l
    mode = "wrap" if wrap else "edge"
    if patch.ndim == 3:
        pad = ((t, b), (l, r), (0, 0))
    else:
        pad = ((t, b), (l, r))
    out = np.pad(patch, pad, mode=mode)
    return out, (t, b, l, r)

def _crop_from_square(patch_pad: np.ndarray, pads: Tuple[int, int, int, int]) -> np.ndarray:
    t, b, l, r = pads
    h, w = patch_pad.shape[:2]
    return patch_pad[t:h-b if b else h, l:w-r if r else w, ...] if patch_pad.ndim == 3 else patch_pad[t:h-b if b else h, l:w-r if r else w]

def _rot_random_keep_shape(patch: np.ndarray, rot_p: float, rng, wrap: bool) -> np.ndarray:
    """
    Obróć patch k*90° nie zmieniając kształtu:
      - dopaduj do kwadratu,
      - obróć,
      - wytnij do oryginalnego H×W.
    """
    if rot_p <= 0.0 or rng.random() >= rot_p:
        return patch
    # losuj 1..3 (90/180/270)
    k = int(rng.integers(1, 4))
    sq, pads = _pad_to_square(patch, wrap)
    rot = np.rot90(sq, k, axes=(0, 1))
    return _crop_from_square(rot, pads)

# ------------------------------ main --------------------------------

@register("block_mosh_grid", defaults=DEFAULTS, doc=DOC)
def block_mosh_grid(img: np.ndarray, ctx, **p) -> np.ndarray:
    a = np.asarray(img)
    if a.ndim != 3 or a.shape[2] < 3:
        raise ValueError("block_mosh_grid: expected RGB-like image (H,W,C>=3)")
    H, W, _ = a.shape

    # RNG (spójne z resztą systemu – preferuj ctx.rng, wpp fallback)
    rng = getattr(ctx, "rng", np.random.default_rng(7))

    size = max(4, int(p.get("size", DEFAULTS["size"])))
    prob = float(p.get("p", DEFAULTS["p"]))
    max_shift = int(p.get("max_shift", DEFAULTS["max_shift"]))
    mode = str(p.get("mode", DEFAULTS["mode"])).lower()
    swap_radius = max(0, int(p.get("swap_radius", DEFAULTS["swap_radius"])))
    rot_p = float(p.get("rot_p", DEFAULTS["rot_p"]))
    wrap = bool(p.get("wrap", DEFAULTS["wrap"]))
    jitter = float(p.get("channel_jitter", DEFAULTS["channel_jitter"]))
    poster_bits = int(p.get("posterize_bits", DEFAULTS["posterize_bits"]))
    mix = float(np.clip(p.get("mix", DEFAULTS["mix"]), 0.0, 1.0))
    mask_key = p.get("mask_key", DEFAULTS["mask_key"])
    use_amp = p.get("use_amp", DEFAULTS["use_amp"])
    amp_infl = float(max(0.0, p.get("amp_influence", DEFAULTS["amp_influence"])))
    clamp = bool(p.get("clamp", DEFAULTS["clamp"]))

    # mapy wag
    mask_map = _resolve_mask(ctx, mask_key, H, W)
    amp_map  = _amplitude_weight_map(ctx, H, W, use_amp)

    # diagnostyki
    sel_map   = np.zeros((H, W), dtype=np.float32)
    dx_map    = np.zeros((H, W), dtype=np.float32) + 0.5
    dy_map    = np.zeros((H, W), dtype=np.float32) + 0.5
    alpha_map = np.zeros((H, W), dtype=np.float32)

    out = a.copy()

    # lista bloków
    bys = list(range(0, H, size))
    bxs = list(range(0, W, size))

    def _rand_block_near(bxi: int, byi: int) -> Tuple[int, int]:
        bx0 = max(0, bxi - swap_radius); bx1 = min(len(bxs)-1, bxi + swap_radius)
        by0 = max(0, byi - swap_radius); by1 = min(len(bys)-1, byi + swap_radius)
        rx = int(rng.integers(bx0, bx1+1))
        ry = int(rng.integers(by0, by1+1))
        return rx, ry

    for yi, by in enumerate(bys):
        y2 = min(H, by + size)
        for xi, bx in enumerate(bxs):
            x2 = min(W, bx + size)

            # wagi/selektory dla aktualnego bloku
            m_avg = float(mask_map[by:y2, bx:x2].mean())
            a_avg = float(amp_map [by:y2, bx:x2].mean())
            weight = np.clip(m_avg * (0.25 + 0.75*a_avg) * amp_infl, 0.0, 2.0)

            p_eff = np.clip(prob * (0.5 + 0.5 * weight), 0.0, 1.0)
            if rng.random() >= p_eff:
                continue  # blok pominięty

            patch = out[by:y2, bx:x2, :].copy()
            base_patch = patch.copy()

            did_shift = False
            did_swap  = False

            # --- tryb operacji ---
            if mode in ("shift", "shift+swap"):
                dx = int(rng.integers(-max_shift, max_shift+1)) if max_shift > 0 else 0
                dy = int(rng.integers(-max_shift, max_shift+1)) if max_shift > 0 else 0
                # skalowanie siły przez weight
                dx = int(round(dx * weight))
                dy = int(round(dy * weight))
                patch = _shift_patch(patch, dx, dy, wrap)
                did_shift = True

                # zapisz dx/dy diag (0.0..1.0)
                if max_shift > 0:
                    dxx = 0.5 + 0.5 * float(np.clip(dx, -max_shift, max_shift)) / float(max_shift)
                    dyy = 0.5 + 0.5 * float(np.clip(dy, -max_shift, max_shift)) / float(max_shift)
                else:
                    dxx = dyy = 0.5
                dx_map[by:y2, bx:x2] = dxx
                dy_map[by:y2, bx:x2] = dyy

            if mode in ("swap", "shift+swap"):
                rx_i, ry_i = _rand_block_near(xi, yi)
                rb_x, rb_y = bxs[rx_i], bys[ry_i]
                rb_x2, rb_y2 = min(W, rb_x + size), min(H, rb_y + size)
                other = out[rb_y:rb_y2, rb_x:rb_x2, :].copy()
                # różne rozmiary (krawędzie) → dopasuj do wspólnego min
                hh = min(patch.shape[0], other.shape[0])
                ww = min(patch.shape[1], other.shape[1])
                if hh > 0 and ww > 0:
                    tmp = patch[:hh, :ww, :].copy()
                    patch[:hh, :ww, :] = other[:hh, :ww, :]
                    out[rb_y:rb_y+hh, rb_x:rb_x+ww, :] = tmp
                    did_swap = True

            # rotacja k*90° z zachowaniem kształtu
            patch = _rot_random_keep_shape(patch, rot_p, rng, wrap)

            # jitter per-channel
            if jitter > 0:
                patch = _jitter_channels(patch, jitter, rng, wrap)

            # posterize (na końcu w domenie u8, potem wracamy do f32 do miksu)
            if poster_bits > 0:
                patch = _posterize_u8(_to_u8(patch.astype(np.float32)), poster_bits).astype(np.uint8)
                patch = _to_f32(patch)

            # alpha/mix (z wagą)
            alpha = np.clip(mix * (0.5 + 0.5 * weight), 0.0, 1.0)
            alpha_map[by:y2, bx:x2] = alpha

            # upewnij się, że domena float do miksu
            if base_patch.dtype != np.float32:
                base_f = _to_f32(base_patch)
            else:
                base_f = base_patch
            if patch.dtype != np.float32:
                patch_f = _to_f32(patch)
            else:
                patch_f = patch

            mixed = base_f*(1.0 - alpha) + patch_f*alpha
            out[by:y2, bx:x2, :] = mixed

            # diagnostyka: zaznaczenie wybranych bloków
            sel_map[by:y2, bx:x2] = np.maximum(sel_map[by:y2, bx:x2], float(p_eff))

    # diagnostyki HUD
    if getattr(ctx, "cache", None) is not None:
        def _u(g: np.ndarray) -> np.ndarray:
            g = np.clip(g.astype(np.float32), 0.0, 1.0)
            u = (g*255.0 + 0.5).astype(np.uint8)
            return np.stack([u, u, u], axis=-1)
        ctx.cache["diag/bmg_select"] = _u(sel_map)
        ctx.cache["diag/bmg_dx"]     = _u(dx_map)
        ctx.cache["diag/bmg_dy"]     = _u(dy_map)
        ctx.cache["diag/bmg_alpha"]  = _u(alpha_map)

    # wyjście
    if clamp:
        return _to_u8(np.asarray(out, dtype=np.float32))
    else:
        x = np.asarray(out, dtype=np.float32)
        return (np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)
