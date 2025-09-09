import numpy as np
from glitchlab.core.registry import register


@register("nazca_lines")
def nazca_lines(img, ctx, density=20, thickness=2, angle=45):
    """
    Nazca-style line overlay: geometric glitch effect.
    Draws sinusoidal/geometric lines as displacement masks.
    """
    h, w, c = img.shape
    out = img.copy()

    # tworzymy siatkę
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    # wzór linii (sinusoida + kąt obrotu)
    angle_rad = np.deg2rad(angle)
    coords = np.cos(angle_rad) * xx + np.sin(angle_rad) * yy
    lines = (np.sin(coords / density) > 0).astype(np.uint8)

    # maska z grubością
    mask = np.zeros_like(lines)
    for shift in range(-thickness, thickness + 1):
        mask |= np.roll(lines, shift, axis=1)

    # nakładamy efekt: np. odwracamy kolory w maskach
    for i in range(c):
        out[:, :, i][mask == 1] = 255 - out[:, :, i][mask == 1]

    return out
