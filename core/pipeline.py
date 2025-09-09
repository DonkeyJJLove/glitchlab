from typing import Dict, Any, List
import numpy as np, yaml
from PIL import Image
from .registry import get, available
from .utils import Ctx
from .roi import sobel_edges_mask, dark_bar_mask, amplitude_field, load_rois_from_yaml


def load_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGBA")
    return np.array(img).astype(np.int16)


def save_image(arr, path):
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.shape[-1] == 4:
        mode = "RGBA"
    elif arr.shape[-1] == 3:
        mode = "RGB"
    else:
        raise ValueError(f"Unsupported channel count: {arr.shape[-1]}")

    Image.fromarray(arr, mode).save(path)


def build_ctx(arr: np.ndarray, seed: int, cfg: Dict[str, Any]) -> Ctx:
    rng = np.random.default_rng(seed)
    masks = {}
    # default text/edge protection
    e = cfg.get("edge_mask", {})
    masks["edges"] = sobel_edges_mask(arr, thresh=e.get("thresh", 70), dilate=e.get("dilate", 3))
    masks["headline"] = dark_bar_mask(arr, top_fraction=0.4, thresh=80)
    if "rois_yaml" in cfg and cfg["rois_yaml"]:
        rois = load_rois_from_yaml(cfg["rois_yaml"], arr.shape[:2])
        masks.update({f"roi_{k}": v for k, v in rois.items()})
    amp_cfg = cfg.get("amplitude", {"kind": "linear_x", "strength": 1.0})
    amp = amplitude_field(arr.shape[:2], amp_cfg.get("kind", "linear_x"), amp_cfg.get("strength", 1.0))
    return Ctx(rng=rng, masks=masks, amplitude=amp, meta={"filters": available()})


def apply_pipeline(arr: np.ndarray, ctx: Ctx, steps: List[Dict[str, Any]]) -> np.ndarray:
    out = arr.copy()
    for step in steps:
        name = step["name"]
        fn = get(name)
        params = step.get("params", {})
        out = fn(out, ctx, **params)
    return out


def load_config(yaml_text: str) -> Dict[str, Any]:
    return yaml.safe_load(yaml_text)
