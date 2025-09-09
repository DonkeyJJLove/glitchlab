
from ..core.registry import register
from ..core.utils import Ctx
from ..core.symbols import load_mask_image, register_mask
@register("load_mask")
def load_mask(arr, ctx: Ctx, key: str, path: str, threshold: int=128):
    mask = load_mask_image(path, shape=arr.shape[:2], threshold=threshold)
    register_mask(ctx.masks, key, mask)
    return arr
