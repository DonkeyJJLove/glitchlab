# glitchlab/gui/main.py
# -*- coding: utf-8 -*-
"""
Bootstrap GUI dla GlitchLab (Tkinter).
Opcjonalne uruchomienie z obrazem/presetem/filtr-em:

  python -m glitchlab.gui.main --image in.png
  python -m glitchlab.gui.main --image in.png --preset spectral_ring_lab
  python -m glitchlab.gui.main --image in.png --filter pixel_sort_adaptive --seed 11
"""

from __future__ import annotations
import argparse
import sys
import traceback

# High-DPI (Windows) – nie szkodzi na innych platformach
try:
    import ctypes  # type: ignore
    ctypes.windll.shcore.SetProcessDpiAwareness(1)  # PER_MONITOR_AWARE
except Exception:
    pass

from glitchlab.gui.app import App, _to_rgb_uint8
from glitchlab.core.pipeline import load_image
from glitchlab.core.utils import normalize_image


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="GlitchLab GUI")
    p.add_argument("-i", "--image", help="Ścieżka do obrazu do wczytania na start")
    p.add_argument("--preset", help="Nazwa presetu do natychmiastowego zastosowania")
    p.add_argument("--filter", help="Nazwa filtra do natychmiastowego zastosowania (single)")
    p.add_argument("--seed", type=int, default=None, help="Seed RNG")
    return p.parse_args(argv)


def preload(app: App, image_path: str | None, preset: str | None, filter_name: str | None, seed: int | None):
    # wczytaj obraz, jeśli wskazany
    if image_path:
        try:
            arr = normalize_image(load_image(image_path))
            app.arr = arr
            app.image_path = image_path
            app.result = None
            app.ctx = None
            app.show_image(arr)
            app.set_status(f"Loaded: {image_path}")
        except Exception as e:
            tb = traceback.format_exc(limit=1)
            app.set_status("Image load error")
            print(f"[error] cannot load image: {e}\n{tb}", file=sys.stderr)

    # ustaw seed
    if seed is not None:
        try:
            app.seed_var.set(int(seed))
        except Exception:
            pass

    # automatyczne zastosowanie presetu/filtra (w tej kolejności)
    if image_path and preset:
        try:
            app.preset_var.set(preset)
            app.on_apply_preset()
        except Exception as e:
            tb = traceback.format_exc(limit=1)
            print(f"[error] preset apply failed: {e}\n{tb}", file=sys.stderr)
    elif image_path and filter_name:
        try:
            app.filter_var.set(filter_name)
            app.on_filter_changed()   # zbuduj panel
            app.on_apply_single()
        except Exception as e:
            tb = traceback.format_exc(limit=1)
            print(f"[error] single filter apply failed: {e}\n{tb}", file=sys.stderr)


def main():
    args = parse_args()
    app = App()
    # pre-load po zbudowaniu UI (panel-holder istnieje)
    preload(app, args.image, args.preset, args.filter, args.seed)
    app.mainloop()


if __name__ == "__main__":
    main()
