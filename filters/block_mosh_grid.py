# glitchlab/filters/block_mosh_grid.py
# -*- coding: utf-8 -*-
"""
Block Mosh Grid
===============
Glitch typu *datamosh* wykonywany w siatce bloków. Dla losowo wybranych bloków
wykonuje przesunięcia (per-blok), ew. swapy bloków, jitter koloru, z możliwością
sterowania prawdopodobieństwem przez maskę i skalowania siły przez amplitude.

Rejestr:  @register("block_mosh_grid")

Parametry:
  size            : int        rozmiar bloku (px) – min 4
  p               : float      prawdopodobieństwo wybrania bloku (0..1)
  max_shift       : int        maks. |dx| i |dy| w pikselach (na blok)
  mode            : str        'shift' (domyślnie) lub 'swap'
  wrap            : bool       True: zawijanie na krawędziach; False: clamp
  mask_key        : str|None   jeżeli podane – prawdopodobieństwo ważone maską
  mask_power      : float      potęga modyfikująca wpływ maski (np. 1..3)
  amp_influence   : float      0..2 – ile amplitude skaluje max_shift (0 wyłącza)
  channel_jitter  : float      0..128 – szum dodany per blok (RGB) w wartości 8-bit
  posterize_bits  : int|None   jeżeli >0 – redukcja do n bitów po moshu
  mix             : float      0..1 – mieszanie (0: oryginał, 1: pełny mosh)
  swap_fraction   : float      tylko dla mode='swap' – jaka część wybranych bloków tworzy pary do podmiany

Diagnostyka (ctx.cache):
  - bmg_select : mapa 0..1 wybranych bloków (upscalowana do rozdz. obrazu)
  - bmg_dx     : mapa przesunięć X (px)  (dla mode='swap' będzie 0)
  - bmg_dy     : mapa przesunięć Y (px)  (dla mode='swap' będzie 0)
"""

from __future__ import annotations
import numpy as np
from glitchlab.core.registry import register


@register("block_mosh_grid")
def block_mosh_grid(
    img: np.ndarray,
    ctx,
    size: int = 16,
    p: float = 0.5,
    max_shift: int = 24,
    mode: str = "shift",
    wrap: bool = True,
    mask_key: str | None = None,
    mask_power: float = 1.0,
    amp_influence: float = 1.0,
    channel_jitter: float = 0.0,
    posterize_bits: int | None = None,
    mix: float = 1.0,
    swap_fraction: float = 0.5,
):
    # ——— sanity ———
    a = np.asarray(img)
    if a.ndim != 3 or a.shape[2] < 3:
        raise ValueError("block_mosh_grid: expected RGB-like image (H,W,C>=3)")
    H, W, C = a.shape
    size = max(4, int(size))
    p = float(np.clip(p, 0.0, 1.0))
    max_shift = max(0, int(max_shift))
    mode = str(mode or "shift").lower()
    wrap = bool(wrap)
    mix = float(np.clip(mix, 0.0, 1.0))

    # ——— RNG deterministyczny ———
    rng = getattr(ctx, "rng", None)
    if rng is None:
        seed = int(getattr(ctx, "seed", 0))
        rng = np.random.default_rng(seed)

    # ——— padding do wielokrotności rozmiaru bloku ———
    Gh = (H + size - 1) // size  # liczba bloków w pionie
    Gw = (W + size - 1) // size
    Hp = Gh * size
    Wp = Gw * size

    if (Hp, Wp) != (H, W):
        pad = np.pad(
            a[..., :3],
            ((0, Hp - H), (0, Wp - W), (0, 0)),
            mode="edge"
        )
    else:
        pad = a[..., :3].copy()

    # ——— maska blokowa (prawdopodobieństwo) ———
    block_prob = np.full((Gh, Gw), p, dtype=np.float32)

    if mask_key and getattr(ctx, "masks", None) and ctx.masks.get(mask_key) is not None:
        m = np.asarray(ctx.masks[mask_key]).astype(np.float32)
        if m.shape != (H, W):
            m = _fit_mask_hw(m, H, W)
        # uśrednij maskę w blokach
        m_pad = np.pad(m, ((0, Hp - H), (0, Wp - W)), mode="edge")
        m_blocks = m_pad.reshape(Gh, size, Gw, size).mean(axis=(1, 3))
        m_blocks = np.clip(m_blocks, 0.0, 1.0) ** float(max(0.0, mask_power))
        block_prob *= m_blocks

    # ——— amplitude → skala przesunięć ———
    amp_scale = np.ones((Gh, Gw), dtype=np.float32)
    if amp_influence > 0 and getattr(ctx, "amplitude", None) is not None:
        amp = np.asarray(ctx.amplitude).astype(np.float32)
        if amp.shape != (H, W):
            amp = _fit_mask_hw(amp, H, W)
        amp_pad = np.pad(amp, ((0, Hp - H), (0, Wp - W)), mode="edge")
        amp_blocks = amp_pad.reshape(Gh, size, Gw, size).mean(axis=(1, 3))
        amp_blocks = (amp_blocks - amp_blocks.min()) / (amp_blocks.max() - amp_blocks.min() + 1e-12)
        # skala 0.5..1.5 * amp_influence (żeby 0 nie wyłączał)
        amp_scale = 0.5 + 1.0 * amp_blocks * float(amp_influence)

    # ——— wybór bloków ———
    pick = rng.random((Gh, Gw)) < block_prob
    # diagnostyka (selection mapa upscalowana)
    if getattr(ctx, "cache", None) is not None:
        sel_vis = pick.astype(np.float32).repeat(size, 0).repeat(size, 1)
        ctx.cache["bmg_select"] = sel_vis[:H, :W].copy()

    # ——— tryb 'swap' ———
    if mode == "swap":
        outp = pad.copy()
        ys, xs = np.where(pick)
        idx = list(zip(ys.tolist(), xs.tolist()))
        rng.shuffle(idx)
        # ile par do wymiany?
        n_pairs = int(len(idx) * float(np.clip(swap_fraction, 0.0, 1.0)) // 2)
        for k in range(n_pairs):
            (y1, x1) = idx[2 * k]
            (y2, x2) = idx[2 * k + 1]
            y1p, x1p = y1 * size, x1 * size
            y2p, x2p = y2 * size, x2 * size
            b1 = outp[y1p:y1p + size, x1p:x1p + size, :3].copy()
            b2 = outp[y2p:y2p + size, x2p:x2p + size, :3].copy()
            outp[y1p:y1p + size, x1p:x1p + size, :3] = b2
            outp[y2p:y2p + size, x2p:x2p + size, :3] = b1
        # jitter koloru
        if channel_jitter > 0:
            _apply_channel_jitter_blocks(outp, pick, size, rng, channel_jitter)
        # posterize + mix
        out = _post_and_mix(a[..., :3], outp[:H, :W, :3], posterize_bits, mix)

        # diagnostyka dx/dy dla swapów (zerowa)
        if getattr(ctx, "cache", None) is not None:
            ctx.cache["bmg_dx"] = np.zeros((H, W), np.float32)
            ctx.cache["bmg_dy"] = np.zeros((H, W), np.float32)
        return _return_like_input(img, out)

    # ——— tryb 'shift' ———
    outp = pad.copy()

    # mapy dx/dy (px) tylko dla wybranych bloków
    dx_blocks = np.zeros((Gh, Gw), dtype=np.int32)
    dy_blocks = np.zeros((Gh, Gw), dtype=np.int32)

    # losuj przesunięcia per blok
    if max_shift > 0:
        # losuj w zakresie [-max_shift, max_shift], z wagą amplitude
        dx_rand = rng.integers(-max_shift, max_shift + 1, size=(Gh, Gw))
        dy_rand = rng.integers(-max_shift, max_shift + 1, size=(Gh, Gw))
        dx_blocks[pick] = (dx_rand[pick] * amp_scale[pick]).astype(np.int32)
        dy_blocks[pick] = (dy_rand[pick] * amp_scale[pick]).astype(np.int32)

    # zastosuj przesunięcia per blok
    for gy in range(Gh):
        y0 = gy * size
        y1 = y0 + size
        for gx in range(Gw):
            if not pick[gy, gx]:
                continue
            x0 = gx * size
            x1 = x0 + size
            dx = int(dx_blocks[gy, gx])
            dy = int(dy_blocks[gy, gx])

            if wrap:
                xs = (np.arange(x0, x1) + dx) % Wp
                ys = (np.arange(y0, y1) + dy) % Hp
                blk = outp[np.ix_(ys, xs)]
            else:
                xs0, xs1 = _shift_span(x0, x1, dx, 0, Wp)
                ys0, ys1 = _shift_span(y0, y1, dy, 0, Hp)
                src = outp[ys0:ys1, xs0:xs1, :3]
                blk = np.zeros((size, size, 3), dtype=outp.dtype)
                blk[:src.shape[0], :src.shape[1], :] = src

            outp[y0:y1, x0:x1, :3] = blk

    # jitter koloru
    if channel_jitter > 0:
        _apply_channel_jitter_blocks(outp, pick, size, rng, channel_jitter)

    # diagnostyka dx/dy upscalowana
    if getattr(ctx, "cache", None) is not None:
        dx_vis = dx_blocks.astype(np.float32).repeat(size, 0).repeat(size, 1)[:H, :W]
        dy_vis = dy_blocks.astype(np.float32).repeat(size, 0).repeat(size, 1)[:H, :W]
        ctx.cache["bmg_dx"] = dx_vis
        ctx.cache["bmg_dy"] = dy_vis

    # posterize + mix
    out = _post_and_mix(a[..., :3], outp[:H, :W, :3], posterize_bits, mix)
    return _return_like_input(img, out)


# -------------------------- Helpers --------------------------

def _shift_span(x0: int, x1: int, dx: int, lo: int, hi: int) -> tuple[int, int]:
    """Zwróć [xs0,xs1) po przesunięciu, z clampingiem do [lo,hi)."""
    xs0 = max(lo, min(hi, x0 + dx))
    xs1 = max(lo, min(hi, x1 + dx))
    return xs0, xs1


def _apply_channel_jitter_blocks(pad: np.ndarray, pick: np.ndarray, size: int, rng, jitter: float):
    """Dodaj per-blokowy jitter RGB w zakresie ±jitter (8-bit)."""
    Gh, Gw = pick.shape
    if jitter <= 0:
        return
    # los per blok, jedna wartość na kanał (R,G,B)
    noise = rng.normal(0.0, float(jitter), size=(Gh, Gw, 3)).astype(np.float32)
    for gy in range(Gh):
        y0 = gy * size
        y1 = y0 + size
        for gx in range(Gw):
            if not pick[gy, gx]:
                continue
            x0 = gx * size
            x1 = x0 + size
            pad[y0:y1, x0:x1, :3] = np.clip(pad[y0:y1, x0:x1, :3].astype(np.float32) + noise[gy, gx], 0, 255).astype(pad.dtype)


def _post_and_mix(src_rgb: np.ndarray, moshed_rgb: np.ndarray, bits: int | None, mix: float) -> np.ndarray:
    out = moshed_rgb.astype(np.float32)
    if bits and int(bits) > 0:
        levels = float(2 ** int(bits) - 1)
        out = (np.round(out / 255.0 * levels) / levels) * 255.0
    out = (1.0 - mix) * src_rgb.astype(np.float32) + mix * out
    return np.clip(out, 0, 255).astype(np.uint8)


def _fit_mask_hw(m: np.ndarray, H: int, W: int) -> np.ndarray:
    """Dopasuj maskę 2D do (H,W) przez proste docięcie/padding."""
    mh, mw = m.shape[:2]
    out = np.zeros((H, W), dtype=np.float32)
    h = min(H, mh); w = min(W, mw)
    out[:h, :w] = m[:h, :w].astype(np.float32)
    if h < H:
        out[h:, :w] = out[h - 1:h, :w]
    if w < W:
        out[:H, w:] = out[:H, w - 1:w]
    return out


def _return_like_input(inp: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """Zwróć wynik o tym samym „kształcie” kanałów co wejście (RGB/RGBA)."""
    if inp.shape[2] >= 4:
        out = np.concatenate([rgb, inp[..., 3:4]], axis=2)
    else:
        out = rgb
    return out
