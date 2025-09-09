import argparse, sys, yaml, os, json
import numpy as np
from ..core.pipeline import load_image, save_image, build_ctx, apply_pipeline, load_config
from ..core.utils import Ctx
from ..filters import *  # register filters


def parse_args():
    p = argparse.ArgumentParser(description="glitchlab — control over error")
    p.add_argument("-i", "--input", required=True, help="Input image (PNG/JPG)")
    p.add_argument("-o", "--output", required=True, help="Output image (PNG)")
    p.add_argument("--preset", default="default", help="Preset name (default, focus_text)")
    p.add_argument("--config", default=None, help="Custom YAML config to merge/override")
    p.add_argument("--seed", type=int, default=7, help="RNG seed for reproducibility")
    p.add_argument("--debug-masks", action="store_true", help="Export masks & amplitude fields")
    return p.parse_args()


def load_preset(name: str) -> str:
    here = os.path.dirname(__file__)
    path = os.path.join(os.path.dirname(here), "presets", f"{name}.yaml")
    if not os.path.exists(path):
        raise SystemExit(f"Unknown preset '{name}'")
    return open(path, "r", encoding="utf-8").read()


def main():
    args = parse_args()
    arr = load_image(args.input)
    preset_yaml = load_preset(args.preset)
    cfg = load_config(preset_yaml)
    if args.config:
        user_yaml = open(args.config, "r", encoding="utf-8").read()
        user_cfg = load_config(user_yaml)
        cfg.update(user_cfg or {})
    ctx = build_ctx(arr, seed=args.seed, cfg=cfg)
    # stash original to ctx.meta for protect_edges
    import copy
    ctx.meta = {"original": copy.deepcopy(arr)}
    steps = cfg.get("steps", [])
    out = apply_pipeline(arr, ctx, steps)
    save_image(out, args.output)
    if args.debug_masks:
        from PIL import Image
        h, w, _ = arr.shape
        for k, m in ctx.masks.items():
            img = Image.fromarray((m * 255).astype(np.uint8), 'L')
            img.save(os.path.splitext(args.output)[0] + f".mask_{k}.png")
        if ctx.amplitude is not None:
            amp = (ctx.amplitude / ctx.amplitude.max() * 255).astype(np.uint8)
            Image.fromarray(amp, 'L').save(os.path.splitext(args.output)[0] + ".amp.png")


if __name__ == "__main__":
    main()

