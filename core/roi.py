from typing import Dict, Tuple, List, Optional
import numpy as np, yaml

def to_gray(arr: np.ndarray) -> np.ndarray:
    r,g,b = arr[...,0], arr[...,1], arr[...,2]
    return (0.299*r + 0.587*g + 0.114*b).astype(np.float32)

def sobel_edges_mask(arr: np.ndarray, thresh: float=80, dilate: int=2) -> np.ndarray:
    y = to_gray(arr)
    yp = np.pad(y, ((1,1),(1,1)), mode='edge')
    gx = (-1*yp[:-2,:-2] + 1*yp[:-2,2:] +
          -2*yp[1:-1,:-2] + 2*yp[1:-1,2:] +
          -1*yp[2:,:-2] + 1*yp[2:,2:])
    gy = (-1*yp[:-2,:-2] -2*yp[:-2,1:-1] -1*yp[:-2,2:] +
           1*yp[2:,:-2] +2*yp[2:,1:-1] +1*yp[2:,2:])
    mag = np.hypot(gx, gy)
    mask = (mag > thresh).astype(np.uint8)
    if dilate > 0:
        r = dilate
        acc = mask.copy()
        for dy in range(-r, r+1):
            for dx in range(-r, r+1):
                if dx==0 and dy==0: 
                    continue
                acc = np.maximum(acc, np.roll(np.roll(mask, dy, axis=0), dx, axis=1))
        mask = acc
    return mask

def dark_bar_mask(arr: np.ndarray, top_fraction: float=0.35, thresh: int=80) -> np.ndarray:
    """Heuristic: find large dark rectangular bar near top-left (e.g., headline panel)."""
    h,w,_ = arr.shape
    sub = arr[:int(h*top_fraction), :int(w*0.9)]
    gray = to_gray(sub)
    mask_sub = (gray < thresh).astype(np.uint8)
    # expand to original size
    out = np.zeros((h,w), dtype=np.uint8)
    out[:mask_sub.shape[0], :mask_sub.shape[1]] = mask_sub
    return out

def polygons_mask(shape: Tuple[int,int], polys: List[List[Tuple[int,int]]]) -> np.ndarray:
    # simple rasterization via ray casting
    h,w = shape
    mask = np.zeros((h,w), dtype=np.uint8)
    for poly in polys:
        if len(poly) < 3: continue
        xs = np.array([p[0] for p in poly])
        ys = np.array([p[1] for p in poly])
        minx,maxx = xs.min(), xs.max()
        miny,maxy = ys.min(), ys.max()
        for y in range(int(miny), int(maxy)+1):
            inside = False
            xints = []
            for i in range(len(poly)):
                x1,y1 = poly[i]
                x2,y2 = poly[(i+1)%len(poly)]
                if y1==y2: continue
                if (y>=min(y1,y2)) and (y<max(y1,y2)):
                    xin = x1 + (y-y1)*(x2-x1)/(y2-y1)
                    xints.append(xin)
            xints.sort()
            for i in range(0, len(xints), 2):
                x_start = int(max(minx, xints[i]))
                x_end   = int(min(maxx, xints[i+1])) if i+1<len(xints) else int(maxx)
                mask[y, x_start:x_end+1] = 1
    return mask

def amplitude_field(shape: Tuple[int,int], kind: str="linear_x", strength: float=1.0) -> np.ndarray:
    h,w = shape
    if kind == "linear_x":
        x = np.linspace(0, 1, w, dtype=np.float32)
        amp = np.tile(x, (h,1))
    elif kind == "linear_y":
        y = np.linspace(0, 1, h, dtype=np.float32)[:,None]
        amp = np.tile(y, (1,w))
    elif kind == "radial":
        yy,xx = np.mgrid[0:h,0:w]
        cx,cy = w/2,h/2
        r = np.hypot(xx-cx, yy-cy)
        amp = (r / r.max()).astype(np.float32)
    else:
        amp = np.ones((h,w), dtype=np.float32)
    return (amp * strength).astype(np.float32)

# Note: the original GlitchLab supported both ``load_rois_from_yaml_text``
# and ``load_rois_from_yaml_file`` helpers.  In this fork we provide a
# unified dispatcher ``load_rois_from_yaml`` that will attempt to load
# ROI definitions from either a YAML string or a filename on disk.  If
# the argument contains newlines or does not correspond to a readable
# file path, it is treated as inline YAML content.  Otherwise, the
# file will be read and parsed.  See ``load_rois_from_yaml_text`` for
# accepted YAML format.

def load_rois_from_yaml(source: str, shape: Tuple[int, int]) -> Dict[str, np.ndarray]:
    """Load ROI definitions from a YAML string or file.

    Parameters
    ----------
    source : str
        Either inline YAML text describing polygons/rects or a path to
        a YAML file on disk.  If the string contains a newline or
        ``shape_hw`` extension, it is assumed to be YAML text.  If
        reading the file fails, the function will fall back to
        interpreting the argument as YAML content.
    shape : (int, int)
        Tuple of (H, W) specifying the shape of the target image for
        rasterisation.

    Returns
    -------
    dict
        A mapping from region key to a binary mask (float32 0..1).
    """
    # Heuristic: if the string looks like it contains YAML (newlines
    # or braces) or is not an existing file, treat it as YAML
    import os
    try:
        if os.path.isfile(source) and not any(ch in source for ch in '\n{}[]'):
            return load_rois_from_yaml_file(source, shape)
    except Exception:
        # If any error occurs (e.g. invalid path), treat as inline YAML
        pass
    return load_rois_from_yaml_text(source, shape)

