# glitchlab/gui/widgets/tools/tool_ellipse_select.py
# -*- coding: utf-8 -*-
"""
tool_ellipse_select — narzędzie zaznaczenia eliptycznego (ROI).

Spójne z: tool_rect_select + overlay_renderer
Publikuje: ui.image.region.selected
  {
    "shape": "ellipse",
    "bbox": [x0,y0,x1,y1],  # w koord. OBRAZU, posortowane
    "mask_key": "current"
  }
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import tkinter as tk

# ── Overlay utils ────────────────────────────────────────────────────────────
try:
    from glitchlab.gui.widgets.overlay_renderer import OverlayRenderer, OverlayStyle
except Exception:  # pragma: no cover
    OverlayRenderer = None  # type: ignore
    OverlayStyle = None  # type: ignore

# ── Kontrakt narzędzi (fallback jeśli base.py niedostępny) ───────────────────
try:
    from glitchlab.gui.widgets.tools.base import ToolBase, ToolEventContext  # type: ignore
except Exception:  # pragma: no cover
    from typing import Callable

    @dataclass
    class ToolEventContext:  # type: ignore
        publish: Callable[[str, Dict[str, Any]], None]
        to_image_xy: Callable[[int, int], Tuple[int, int]]
        invalidate: Callable[[], None]
        get_mask: Callable[[], Any] | None = None
        set_mask: Callable[[Any], None] | None = None
        get_image: Callable[[], Any] | None = None
        get_zoom_pan: Callable[[], tuple[float, tuple[int, int]]] | None = None

    class ToolBase:  # type: ignore
        name: str = "base"
        def __init__(self, ctx: ToolEventContext): self.ctx = ctx
        def on_activate(self, opts: Dict[str, Any] | None = None): ...
        def on_deactivate(self): ...
        def on_mouse_down(self, e): ...
        def on_mouse_move(self, e): ...
        def on_mouse_up(self, e): ...
        def on_key(self, e): ...
        def on_wheel(self, e): ...
        def draw_overlay(self, canvas): ...


# ═════════════════════════════════════════════════════════════════════════════
class EllipseSelectTool(ToolBase):
    name = "ellipse"

    def __init__(self, ctx: ToolEventContext) -> None:
        super().__init__(ctx)
        # punkty w PRZESTRZENI OBRAZU (dla dokładnej publikacji)
        self._p0_img: Optional[Tuple[int, int]] = None
        self._p1_img: Optional[Tuple[int, int]] = None
        # overlay rysujemy w przestrzeni EKRANU (przez get_zoom_pan)
        self._zoom_cache: float = 1.0
        self._pan_cache: Tuple[int, int] = (0, 0)
        # opcje
        self._shift_circle: bool = True
        self._min_size: int = 1
        # styl
        self._style = OverlayStyle() if OverlayStyle else None

    # ── lifecycle ────────────────────────────────────────────────────────────
    def on_activate(self, opts: Optional[Dict[str, Any]] = None) -> None:
        if opts:
            self._shift_circle = bool(opts.get("shift_circle", True))
            self._min_size = int(opts.get("min_size", 1))
        self._p0_img = None
        self._p1_img = None
        self._update_view_cache()
        self.ctx.invalidate()

    def on_deactivate(self) -> None:
        self._p0_img = None
        self._p1_img = None
        self.ctx.invalidate()

    # ── mouse ────────────────────────────────────────────────────────────────
    def on_mouse_down(self, ev: Any) -> None:
        ix, iy = self.ctx.to_image_xy(int(ev.x), int(ev.y))
        self._p0_img = (ix, iy)
        self._p1_img = (ix, iy)
        self._update_view_cache()
        self.ctx.invalidate()

    def on_mouse_move(self, ev: Any) -> None:
        if self._p0_img is None:
            return
        ix, iy = self.ctx.to_image_xy(int(ev.x), int(ev.y))

        # SHIFT => wymuś okrąg (równy promień w X/Y)
        make_circle = False
        try:
            # Tk: bit 0x0001 często oznacza Shift; różne platformy mogą się różnić
            make_circle = self._shift_circle and ((int(getattr(ev, "state", 0)) & 0x0001) != 0)
        except Exception:
            make_circle = False

        if make_circle:
            x0, y0 = self._p0_img
            dx = ix - x0
            dy = iy - y0
            r = max(abs(dx), abs(dy))
            ix = x0 + (r if dx >= 0 else -r)
            iy = y0 + (r if dy >= 0 else -r)

        self._p1_img = (ix, iy)
        self.ctx.invalidate()

    def on_mouse_up(self, _ev: Any) -> None:
        if self._p0_img is None or self._p1_img is None:
            return
        x0, y0, x1, y1 = self._normalized_bbox(self._p0_img, self._p1_img)
        if (x1 - x0) >= self._min_size and (y1 - y0) >= self._min_size:
            try:
                self.ctx.publish("ui.image.region.selected", {
                    "shape": "ellipse",
                    "bbox": [int(x0), int(y0), int(x1), int(y1)],
                    "mask_key": "current",
                })
            except Exception:
                pass
        # zostawiamy overlay do następnej akcji? Na razie czyścimy:
        self._p0_img = None
        self._p1_img = None
        self.ctx.invalidate()

    def on_key(self, e: Any) -> None:
        # Esc: anuluj
        if str(getattr(e, "keysym", "")).lower() == "escape":
            self._p0_img = None
            self._p1_img = None
            self.ctx.invalidate()

    # ── overlay render ───────────────────────────────────────────────────────
    def draw_overlay(self, tk_canvas: Any) -> None:
        if self._p0_img is None or self._p1_img is None:
            return

        # konwersja obraz→ekran
        z, (px, py) = self._safe_zoom_pan()
        def img_to_screen(ix: int, iy: int) -> Tuple[int, int]:
            return (int(round(ix * z + px)), int(round(iy * z + py)))

        x0, y0, x1, y1 = self._normalized_bbox(self._p0_img, self._p1_img)
        sx0, sy0 = img_to_screen(x0, y0)
        sx1, sy1 = img_to_screen(x1, y1)

        if OverlayRenderer is None:
            # fallback: zwykła elipsa
            tk_canvas.create_oval(sx0, sy0, sx1, sy1, outline="#65b0ff", dash=(4, 2))
            return

        OverlayRenderer.draw_ellipse(
            tk_canvas, sx0, sy0, sx1, sy1,
            tag="tool_ellipse",
            zoom=max(1e-6, z),
            style=self._style,
        )
        # opcjonalne uchwyty (rogi)
        OverlayRenderer.draw_handles(
            tk_canvas, (sx0, sy0, sx1, sy1),
            tag="handles",
            zoom=max(1e-6, z),
            style=self._style,
            corners_only=True,
        )

    # ── helpers ──────────────────────────────────────────────────────────────
    def _update_view_cache(self) -> None:
        try:
            if callable(getattr(self.ctx, "get_zoom_pan", None)):
                z, pan = self.ctx.get_zoom_pan()  # type: ignore[call-arg]
                self._zoom_cache = float(z)
                self._pan_cache = (int(pan[0]), int(pan[1]))
            else:
                self._zoom_cache, self._pan_cache = 1.0, (0, 0)
        except Exception:
            self._zoom_cache, self._pan_cache = 1.0, (0, 0)

    def _safe_zoom_pan(self) -> Tuple[float, Tuple[int, int]]:
        # zawsze zwraca sensowne wartości (1.0, (0,0)) gdy brak kontekstu
        if self._zoom_cache <= 0:
            self._update_view_cache()
        return (self._zoom_cache or 1.0), (self._pan_cache or (0, 0))

    @staticmethod
    def _normalized_bbox(p0: Tuple[int, int], p1: Tuple[int, int]) -> Tuple[int, int, int, int]:
        x0, y0 = p0
        x1, y1 = p1
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        return int(x0), int(y0), int(x1), int(y1)
