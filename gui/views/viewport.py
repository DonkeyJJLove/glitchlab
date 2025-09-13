"""
---
version: 3
kind: module
id: "view-viewport"
created_at: "2025-09-13"
name: "glitchlab.gui.views.viewport"
author: "GlitchLab v3"
role: "Podgląd obrazu z nakładkami (overlay) i publikacją zdarzeń pick (Tk-safe)"
description: >
  Lekki wrapper na ImageCanvas do renderu obrazu RGB u8, półprzezroczystych
  nakładek (RGBA/gray) i publikacji kliknięć w współrzędnych obrazu przez EventBus
  (domyślnie topic 'ui.viewport.pick'). Obsługuje crosshair, dopasowanie (fit),
  oraz bezpieczny fallback gdy widgets.image_canvas nie jest dostępny.
inputs:
  image:   {type: "PIL.Image|np.ndarray", shape: "(H,W[,(3|4)])", dtype: "uint8"}
  overlay: {type: "PIL.Image|np.ndarray|None", shape: "(H,W[,1|3|4])", note: "auto-resize, alpha mix"}
  config:
    show_crosshair: {type: "bool", default: false}
    overlay_alpha:  {type: "float", default: 0.65}
    publish_pick_topic: {type: "str|None", default: "ui.viewport.pick"}
  bus: {type: "EventBus|None", desc: "opcjonalny; do publikacji zdarzeń"}
outputs:
  render:  {type: "UI", note: "aktualizacja widoku obrazu/overlay"}
  events:
    ui.viewport.pick: {x: "int", y: "int"}
interfaces:
  exports: ["Viewport", "ViewportConfig"]
depends_on: ["tkinter/ttk","Pillow","NumPy","glitchlab.gui.widgets.image_canvas?","glitchlab.gui.event_bus?"]
used_by: ["glitchlab.gui.app_shell","glitchlab.gui.views.notebook"]
policy:
  deterministic: true
  ui_thread_only: true
constraints:
  - "brak ciężkich obliczeń (render-only)"
  - "brak SciPy/OpenCV"
hud:
  overlays: ["stage/*/mosaic","diag/*/*","format/jpg_grid"]
license: "Proprietary"
---
"""
# glitchlab/gui/views/viewport.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Callable, Union

import tkinter as tk
from tkinter import ttk

import numpy as np
from PIL import Image

try:
    # Prefer the newer widget if available
    from glitchlab.gui.widgets.image_canvas import ImageCanvas  # type: ignore
except Exception:  # Fallback to a simple shim
    class ImageCanvas(tk.Canvas):  # type: ignore
        """Minimal fallback: show a single PhotoImage; no overlays / coords mapping."""
        def __init__(self, master: tk.Misc, **kw):
            super().__init__(master, **kw)
            self._photo = None
            self._img_size: Tuple[int, int] = (1, 1)
            self.bind("<Configure>", lambda e: self._redraw())

        def set_image(self, img_u8: np.ndarray) -> None:
            from PIL import ImageTk
            self._img_size = (img_u8.shape[1], img_u8.shape[0])
            pil = Image.fromarray(img_u8, mode="RGB")
            self._photo = ImageTk.PhotoImage(pil)
            self._redraw()

        def set_overlay(self, _overlay_u8: Optional[np.ndarray]) -> None:
            # Not supported in fallback
            pass

        def fit(self) -> None:
            self._redraw()

        def to_image_coords(self, x: int, y: int) -> Tuple[int, int]:
            # Naive 1:1 mapping (no zoom/pan in fallback)
            return x, y

        def _redraw(self) -> None:
            self.delete("all")
            if self._photo is None:
                return
            w = self.winfo_width()
            h = self.winfo_height()
            iw, ih = self._img_size
            # center
            ox = (w - iw) // 2
            oy = (h - ih) // 2
            self.create_image(max(0, ox), max(0, oy), image=self._photo, anchor="nw")


# --- helpers ------------------------------------------------------------------------------

def _to_rgb_u8(img: Union[np.ndarray, Image.Image]) -> np.ndarray:
    if isinstance(img, Image.Image):
        pil = img.convert("RGB")
        return np.asarray(pil, dtype=np.uint8)
    if isinstance(img, np.ndarray):
        arr = img
        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, None], 3, axis=2)
        if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
            raise TypeError("Viewport: expected (H,W,3|4) or gray (H,W)")
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.shape[-1] == 4:
            # flatten alpha to black
            pil = Image.fromarray(arr, mode="RGBA")
            bg = Image.new("RGB", pil.size, (0, 0, 0))
            bg.paste(pil, mask=pil.split()[-1])
            return np.asarray(bg, dtype=np.uint8)
        return arr
    raise TypeError("Viewport: unsupported image type")


def _overlay_to_rgba_u8(overlay: Union[np.ndarray, Image.Image], alpha: float) -> np.ndarray:
    """Accept (H,W), (H,W,3), (H,W,4) → RGBA u8 with given global alpha multiplier."""
    if isinstance(overlay, Image.Image):
        if overlay.mode == "RGBA":
            rgba = np.asarray(overlay, dtype=np.uint8)
        elif overlay.mode == "RGB":
            rgb = np.asarray(overlay, dtype=np.uint8)
            a = np.full((rgb.shape[0], rgb.shape[1], 1), int(round(alpha * 255)), dtype=np.uint8)
            rgba = np.concatenate([rgb, a], axis=2)
        else:
            ov = np.asarray(overlay.convert("L"), dtype=np.uint8)
            a = (ov.astype(np.float32) * alpha).astype(np.uint8)
            rgb = np.zeros((ov.shape[0], ov.shape[1], 3), dtype=np.uint8)
            rgba = np.concatenate([rgb, a[:, :, None]], axis=2)
        return rgba
    arr = np.asarray(overlay)
    if arr.ndim == 2:
        a = (np.clip(arr, 0, 255).astype(np.uint8).astype(np.float32) * alpha).astype(np.uint8)
        rgb = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
        return np.concatenate([rgb, a[:, :, None]], axis=2)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        a = np.full((arr.shape[0], arr.shape[1], 1), int(round(alpha * 255)), dtype=np.uint8)
        return np.concatenate([np.clip(arr, 0, 255).astype(np.uint8), a], axis=2)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        rgba = np.clip(arr, 0, 255).astype(np.uint8).copy()
        rgba[..., 3] = (rgba[..., 3].astype(np.float32) * alpha).astype(np.uint8)
        return rgba
    raise TypeError("Viewport: overlay must be 2D gray or 3/4-channel array")


# --- public API ----------------------------------------------------------------------------

@dataclass
class ViewportConfig:
    show_crosshair: bool = False
    publish_pick_topic: Optional[str] = "ui.viewport.pick"  # None to disable
    overlay_alpha: float = 0.65


class Viewport(ttk.Frame):
    """
    Wrapper nad ImageCanvas z obsługą:
      - ustawiania obrazu wejściowego (RGB u8),
      - lekkich nakładek (overlay RGBA u8),
      - publikacji zdarzenia pick (x,y) w przestrzeni obrazu przez EventBus,
      - crosshair (opcjonalnie).
    """

    def __init__(self, master: tk.Misc, *, bus: Any | None = None, config: Optional[ViewportConfig] = None) -> None:
        super().__init__(master)
        self.bus = bus
        self.cfg = config or ViewportConfig()
        self._image: Optional[np.ndarray] = None
        self._overlay_rgba: Optional[np.ndarray] = None

        self.canvas = ImageCanvas(self)
        self.canvas.pack(fill="both", expand=True)

        # interactions
        self.canvas.bind("<Button-1>", self._on_click_left)
        self.bind_all("<Key-f>", lambda e: self.fit())

        # crosshair items (if supported by widget)
        self._crosshair_on = bool(self.cfg.show_crosshair)
        self._crosshair_ids: Tuple[Optional[int], Optional[int]] = (None, None)
        self.canvas.bind("<Motion>", self._on_motion)

    # -- image / overlay -----------------------------------------------------

    def set_image(self, img: Union[np.ndarray, Image.Image]) -> None:
        u8 = _to_rgb_u8(img)
        self._image = u8
        if hasattr(self.canvas, "set_image"):
            self.canvas.set_image(u8)
        else:
            # Fallback shim handled in fallback class
            pass
        # When image changes, re-apply overlay if size differs
        if self._overlay_rgba is not None and self._overlay_rgba.shape[:2] != u8.shape[:2]:
            self.clear_overlay()

    def set_overlay(self, overlay: Optional[Union[np.ndarray, Image.Image]], *, alpha: Optional[float] = None) -> None:
        if overlay is None:
            self.clear_overlay()
            return
        if self._image is None:
            # No image yet; ignore to avoid size mismatch
            return
        a = float(self.cfg.overlay_alpha if alpha is None else alpha)
        rgba = _overlay_to_rgba_u8(overlay, a)
        # Resize if needed
        if rgba.shape[:2] != self._image.shape[:2]:
            from PIL import Image as _PILImage
            pil = _PILImage.fromarray(rgba, mode="RGBA").resize(
                (self._image.shape[1], self._image.shape[0]), resample=_PILImage.BICUBIC
            )
            rgba = np.asarray(pil, dtype=np.uint8)
        self._overlay_rgba = rgba
        if hasattr(self.canvas, "set_overlay"):
            self.canvas.set_overlay(rgba)
        else:
            # Fallback: bake overlay into image (destructive to view only)
            self._bake_overlay_fallback()

    def clear_overlay(self) -> None:
        self._overlay_rgba = None
        if hasattr(self.canvas, "set_overlay"):
            self.canvas.set_overlay(None)
        else:
            # Re-show base
            if self._image is not None:
                self.canvas.set_image(self._image)

    # -- interactions --------------------------------------------------------

    def fit(self) -> None:
        if hasattr(self.canvas, "fit"):
            self.canvas.fit()

    def _on_click_left(self, ev: tk.Event) -> None:
        if not self.cfg.publish_pick_topic or self._image is None:
            return
        try:
            if hasattr(self.canvas, "to_image_coords"):
                ix, iy = self.canvas.to_image_coords(ev.x, ev.y)
            else:
                ix, iy = int(ev.x), int(ev.y)
            ix = max(0, min(self._image.shape[1] - 1, int(ix)))
            iy = max(0, min(self._image.shape[0] - 1, int(iy)))
        except Exception:
            return

        # publish via EventBus if present, else ignore
        if self.bus is not None and hasattr(self.bus, "publish"):
            self.bus.publish(self.cfg.publish_pick_topic, {"x": ix, "y": iy})

    def _on_motion(self, ev: tk.Event) -> None:
        if not self._crosshair_on or self._image is None:
            return
        # If the underlying canvas offers a crosshair API, prefer it; otherwise draw lightweight lines
        if not isinstance(self.canvas, tk.Canvas):
            # Assume widget manages its own crosshair; ignore
            return
        self.canvas.delete("crosshair")
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        self.canvas.create_line(ev.x, 0, ev.x, h, fill="#888", tags="crosshair")
        self.canvas.create_line(0, ev.y, w, ev.y, fill="#888", tags="crosshair")

    # -- utilities -----------------------------------------------------------

    def toggle_crosshair(self, on: Optional[bool] = None) -> None:
        if on is None:
            self._crosshair_on = not self._crosshair_on
        else:
            self._crosshair_on = bool(on)
        if not self._crosshair_on and isinstance(self.canvas, tk.Canvas):
            self.canvas.delete("crosshair")

    def _bake_overlay_fallback(self) -> None:
        """Composite overlay onto base image (view-only) when widget lacks overlay support."""
        if self._image is None or self._overlay_rgba is None:
            return
        base = self._image.astype(np.float32) / 255.0
        ov = self._overlay_rgba.astype(np.float32) / 255.0
        a = ov[..., 3:4]
        rgb = ov[..., :3]
        comp = base * (1.0 - a) + rgb * a
        comp_u8 = np.clip(comp * 255.0 + 0.5, 0, 255).astype(np.uint8)
        self.canvas.set_image(comp_u8)
