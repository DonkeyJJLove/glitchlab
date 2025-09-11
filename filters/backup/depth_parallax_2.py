import numpy as np
from glitchlab.core.registry import register  # standaryzujemy

try:
    import noise
except Exception:
    noise = None

@register("depth_parallax")
def depth_parallax(img, ctx, scale=70, freq=100.0, octaves=5, shading=True, stereo=True):
    """
    Pseudo-3D parallax: 2D warp zależny od 'głębokości' + cieniowanie + lekki stereo.
    Sygnatura zgodna z pipeline: fn(img, ctx, **params).
    """
    h, w, c = img.shape
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    # mapa głębi: Perlin (jeśli jest), fallback: sin*cos
    if noise is not None:
        depth = np.zeros((h, w), dtype=np.float32)
        base = getattr(ctx, "seed", 0)
        for y in range(h):
            for x in range(w):
                depth[y, x] = noise.pnoise2(
                    x / freq, y / freq, octaves=octaves,
                    repeatx=w, repeaty=h, base=base
                )
    else:
        depth = np.sin(xx / 25.0) * np.cos(yy / 25.0)

    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

    # paralaksa względem środka
    cx, cy = w // 2, h // 2
    dx = ((xx - cx) / max(1, w) * depth * scale).astype(int)
    dy = ((yy - cy) / max(1, h) * depth * scale).astype(int)

    out = np.zeros_like(img)
    for i in range(c):
        for y in range(h):
            src_y = np.clip(y + dy[y, :], 0, h - 1)
            src_x = np.clip(np.arange(w) + dx[y, :], 0, w - 1)
            out[y, :, i] = img[src_y, src_x, i]

    # cieniowanie głębią
    if shading:
        out = (out * (0.6 + 0.4 * depth[..., None])).astype(np.uint8)

    # lekki stereo (anaglifowy vibe)
    if stereo and c >= 3:
        out[:, :, 0] = np.roll(out[:, :, 0], 4, axis=1)   # R
        out[:, :, 2] = np.roll(out[:, :, 2], -4, axis=1)  # B

    return out
