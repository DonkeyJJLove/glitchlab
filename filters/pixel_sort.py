import numpy as np
from ..core.registry import register

@register("pixel_sort")
def pixel_sort(arr, ctx, direction="horizontal", length=100):
    """
    Prosty efekt pixel-sorting:
    - direction = "horizontal" albo "vertical"
    - length = długość segmentu do sortowania
    """
    out = arr.copy()
    h, w, c = out.shape

    if direction == "horizontal":
        for y in range(h):
            for x in range(0, w, length):
                seg = out[y, x:x+length]
                if seg.shape[0] > 0:
                    out[y, x:x+length] = np.sort(seg, axis=0)
    else:  # vertical
        for x in range(w):
            for y in range(0, h, length):
                seg = out[y:y+length, x]
                if seg.shape[0] > 0:
                    out[y:y+length, x] = np.sort(seg, axis=0)

    return out
