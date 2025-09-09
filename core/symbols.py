
from typing import Dict
import numpy as np
from PIL import Image
def load_mask_image(path: str, shape=None, threshold: int=128) -> np.ndarray:
    img = Image.open(path).convert("L")
    if shape is not None:
        img = img.resize((shape[1], shape[0]), Image.Resampling.NEAREST)
    arr = np.array(img)
    return (arr >= threshold).astype(np.uint8)
def register_mask(ctx_masks: Dict[str, np.ndarray], key: str, mask: np.ndarray):
    if key in ctx_masks:
        ctx_masks[key] = np.maximum(ctx_masks[key], mask)
    else:
        ctx_masks[key] = mask
