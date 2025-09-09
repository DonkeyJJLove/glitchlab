import numpy as np
from ..core.registry import register

@register("wave_distort")
def wave_distort(arr, ctx, axis="x", amplitude=20, frequency=0.05):
    """
    Fala sinusoidalna: przesuwa piksele wzdłuż osi X lub Y.
    """
    h, w, c = arr.shape
    out = np.zeros_like(arr)

    if axis == "x":
        for y in range(h):
            shift = int(amplitude * np.sin(2 * np.pi * frequency * y))
            out[y] = np.roll(arr[y], shift, axis=0)
    else:
        for x in range(w):
            shift = int(amplitude * np.sin(2 * np.pi * frequency * x))
            out[:, x] = np.roll(arr[:, x], shift, axis=0)

    return out
