"""Utility functions for computing simple image metrics.

The functions defined here operate on NumPy arrays and return scalar
values summarising certain aspects of an image's structure.  These
metrics are used in the new graph-based GlitchLab to label edges and
nodes with informative diagnostics (e.g. how much entropy increases
across an operator, or how dense are the edges in the output).

Note: the functions assume RGB images with shape (H,W,3) and dtype
``uint8`` or a floating-point normalised to 0..1.  They do not modify
the input array.
"""

from __future__ import annotations

import numpy as np

__all__ = ["compute_entropy", "edge_density"]


def compute_entropy(arr: np.ndarray) -> float:
    """Compute the Shannon entropy of an image.

    Parameters
    ----------
    arr : np.ndarray
        Input image array.  The function flattens the array and
        computes a histogram of values to estimate the distribution.
        It will auto-cast to 8-bit if the dtype is floating.

    Returns
    -------
    float
        The estimated entropy in bits.
    """
    # Flatten to 1D and normalise to 0..255
    a = arr
    if a.ndim == 3:
        # convert to grayscale by averaging channels
        a = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    a = a.astype(np.float64)
    if a.max() > 1.0:
        a = a / 255.0
    # compute histogram
    hist, _ = np.histogram(a, bins=256, range=(0, 1), density=True)
    p = hist + 1e-12  # avoid log(0)
    entropy = -np.sum(p * np.log2(p))
    return float(entropy)


def edge_density(arr: np.ndarray) -> float:
    """Compute the average edge magnitude of an image.

    This metric measures how "busy" the image is by computing a simple
    gradient using finite differences.  It is normalised to the range
    0..1 by dividing by the maximum possible gradient magnitude.

    Parameters
    ----------
    arr : np.ndarray
        Input image array of shape (H,W,3) or (H,W).  The function
        first converts the image to grayscale and normalises it.

    Returns
    -------
    float
        A scalar representing the mean gradient magnitude.
    """
    if arr.ndim == 3:
        gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    else:
        gray = arr.astype(np.float64)
    gray = gray.astype(np.float64)
    if gray.max() > 1.0:
        gray = gray / 255.0
    # Sobel-like finite differences
    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    # Pad to original shape for mean
    gx = np.pad(gx, ((0, 0), (0, 1)), constant_values=0)
    gy = np.pad(gy, ((0, 1), (0, 0)), constant_values=0)
    grad_mag = (gx + gy) / 2.0
    return float(np.mean(grad_mag))