import numpy as np
from typing import Callable
"""RGB channel offset operator.

This module defines both the legacy filter registration and the new
operator-based registration used by the graph execution engine. The
legacy function ``rgb_offset`` accepts a raw NumPy array and context
and applies per-channel shifts. The new factory function
``make_rgb_offset`` follows the operator API: given a context and
parameters, it returns a transformer that accepts an ``Artifact``
and produces a new ``Artifact`` with the modified image data.
"""

import numpy as np

from ..core.registry import register  # legacy filter registry
from ..core.utils import Ctx

from ..core.artifact import Artifact
from ..core.operator import OperatorSpec, register_operator


@register("rgb_offset")
def rgb_offset(arr: np.ndarray, ctx: Ctx, r=(1, 0), g=(-1, 0), b=(2, 0)) -> np.ndarray:
    """Legacy RGB offset filter.

    Parameters
    ----------
    arr : np.ndarray
        Input image array (H×W×C).
    ctx : Ctx
        Context object providing RNG, masks, etc.
    r, g, b : tuple
        Displacements for each channel in (dx, dy) format.

    Returns
    -------
    np.ndarray
        The shifted image array.
    """
    out = arr.copy()
    for c, (dx, dy) in enumerate([r, g, b]):
        ch = out[..., c]
        ch = np.roll(ch, dy, axis=0)
        ch = np.roll(ch, dx, axis=1)
        out[..., c] = ch
    return out


def make_rgb_offset(ctx: Ctx, r=(1, 0), g=(-1, 0), b=(2, 0)) -> Callable[[Artifact], Artifact]:
    """Factory function for the RGB offset operator.

    This function captures the context and parameter values and
    returns a transformer that operates on an ``Artifact``. The
    returned transformer expects an artifact of kind ``"image"`` and
    produces a new artifact with the same meta but with the RGB
    channels shifted according to the parameters.

    Parameters
    ----------
    ctx : Ctx
        The processing context; currently unused for this operator.
    r, g, b : tuple
        Displacements for the red, green and blue channels in (dx, dy).

    Returns
    -------
    Callable[[Artifact], Artifact]
        A function that applies the RGB offset to a given image artifact.
    """
    def transform(art: Artifact) -> Artifact:
        if art.kind != "image":
            raise TypeError(f"rgb_offset expects an image artifact, got {art.kind}")
        arr = art.data
        out = arr.copy()
        for c, (dx, dy) in enumerate([r, g, b]):
            ch = out[..., c]
            ch = np.roll(ch, dy, axis=0)
            ch = np.roll(ch, dx, axis=1)
            out[..., c] = ch
        return Artifact("image", out, dict(art.meta))

    return transform


# Register the operator with the new operator registry
register_operator(
    OperatorSpec(
        name="rgb_offset",
        fn_factory=make_rgb_offset,
        inputs=1,
        params_schema={"r": (1, 0), "g": (-1, 0), "b": (2, 0)},
    )
)
