# glitchlab/gui/widgets/tools/tool_rect_select.py
# -*- coding: utf-8 -*-
"""
tool_rect_select — narzędzie zaznaczania prostokątnego (ROI).

Założenia:
- Brak zależności od Core; czysto GUI.
- Rysuje overlay za pomocą OverlayRenderer (tag: "tool_rect").
- Publikuje EventBus: ui.image.region.selected
    payload: { "shape":"rect", "bbox":[x0,y0,x1,y1], "mask_key":"current" }

Integracja:
- Oczekuje środowiska narzędzi jak w ToolBase (ctx: ToolEventContext).
- Koordynaty wewnętrzne trzymamy w przestrzeni *ekranu* (canvas),
  a przy publikacji konwertujemy na koordynaty *obrazu*.

API (zgodnie z ToolBase):
    on_activate(opts)
    on_deactivate()
    on_mouse_down(e)
    on_mouse_move(e)
    on_mouse_up(e)
    on_key(e)
    on_wheel(e)
    draw_overlay(tk_canvas)

Wymaga:
- glitchlab.gui.widgets.overlay_renderer.OverlayRenderer
- glitchlab.gui.widgets.tools.base.ToolBase / ToolEventContext
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import tkinter as tk

# ── Overlay utils ────────────────────────────────────────────────────────────
try:
    from glitchlab.gui.widgets.overlay_renderer import OverlayRenderer, OverlayStyle
except Exception:  # pragma: no cover
    OverlayRenderer = None  # type: ignore
    OverlayStyle = None  # type: ignore

# ── Kontrakt narzędzi (fallback jeżeli nie ma base.py) ───────────────────────
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
class RectSelectTool(ToolBase):
    name = "rect"

    def __init__(self, ctx: ToolEventContext):
        super().__init__(ctx)
        self._drag_active: bool = False
        self._p0: Optional[tuple[int, int]] = None  # screen space
        self._p1: Optional[tuple[int, int]] = None  # screen space
        self._last_drawn_tag: str = "tool_rect"
        # lekki styl (domyślne z OverlayRenderer, ale możemy dodać kontrast)
        self._style = OverlayStyle() if OverlayStyle else None
        self._zoom_cache: float = 1.0

    # ── lifecycle ────────────────────────────────────────────────────────────
    def on_activate(self, opts: Dict[str, Any] | None = None):
        # opcjonalne parametry, np. "from_center": bool
        self._drag_active = False
        self._p0 = None
        self._p1 = None
        self.ctx.invalidate()

    def on_deactivate(self):
        self._drag_active = False
        self._clear_overlay()
        self.ctx.invalidate()

    # ── input ────────────────────────────────────────────────────────────────
    def on_mouse_down(self, e: tk.Event):
        self._drag_active = True
        self._p0 = (int(e.x), int(e.y))
        self._p1 = self._p0
        self._update_zoom_cache()
        self.ctx.invalidate()

    def on_mouse_move(self, e: tk.Event):
        if not self._drag_active or self._p0 is None:
            return
        self._p1 = (int(e.x), int(e.y))
        self.ctx.invalidate()

    def on_mouse_up(self, e: tk.Event):
        if not self._drag_active:
            return
        self._drag_active = False
        if self._p0 is None or self._p1 is None:
            self._clear_overlay()
            self.ctx.invalidate()
            return
        # publikacja regionu (w koordach obrazu)
        (x0s, y0s), (x1s, y1s) = self._p0, self._p1
        x0i, y0i = self.ctx.to_image_xy(x0s, y0s)
        x1i, y1i = self.ctx.to_image_xy(x1s, y1s)
        if x1i < x0i:
            x0i, x1i = x1i, x0i
        if y1i < y0i:
            y0i, y1i = y1i, y0i
        try:
            self.ctx.publish("ui.image.region.selected", {
                "shape": "rect",
                "bbox": [int(x0i), int(y0i), int(x1i), int(y1i)],
                "mask_key": "current",
            })
        except Exception:
            pass
        # zostaw ramkę do czasu kliknięcia gdzie indziej
        self.ctx.invalidate()

    def on_key(self, e: tk.Event):
        # Esc = anuluj zaznaczenie
        if getattr(e, "keysym", "").lower() == "escape":
            self._drag_active = False
            self._p0 = None
            self._p1 = None
            self._clear_overlay()
            self.ctx.invalidate()

    # ── render overlay ───────────────────────────────────────────────────────
    def draw_overlay(self, canvas: tk.Canvas):
        # czyść poprzednią ramkę
        self._clear_overlay(cv=canvas)
        if self._p0 is None or self._p1 is None:
            return
        if OverlayRenderer is None:
            # awaryjnie — prosta ramka na Canvas
            x0, y0 = self._p0
            x1, y1 = self._p1
            canvas.create_rectangle(x0, y0, x1, y1, outline="#65b0ff", dash=(4, 2), tags=(self._last_drawn_tag,))
            return
        # rysuj za pomocą OverlayRenderer
        x0, y0 = self._p0
        x1, y1 = self._p1
        OverlayRenderer.draw_rect(
            canvas, x0, y0, x1, y1,
            tag=self._last_drawn_tag,
            zoom=self._zoom_cache,
            style=self._style,
            with_cross=True,
            with_ticks=True,
        )

    # ── utils ────────────────────────────────────────────────────────────────
    def _clear_overlay(self, cv: Optional[tk.Canvas] = None):
        if cv is None:
            # lazy: spróbuj zdobyć canvas poprzez invalidate() zewnętrznie
            return
        try:
            if OverlayRenderer is not None:
                OverlayRenderer.clear(cv, self._last_drawn_tag)
            else:
                cv.delete(self._last_drawn_tag)
        except Exception:
            pass

    def _update_zoom_cache(self):
        try:
            if callable(getattr(self.ctx, "get_zoom_pan", None)):
                z, _ = self.ctx.get_zoom_pan()  # type: ignore[call-arg]
                self._zoom_cache = float(z)
        except Exception:
            self._zoom_cache = 1.0
