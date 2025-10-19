# glitchlab/app/widgets/tools/tool_measure.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import math

from .base import ToolBase, ToolEventContext


class MeasureTool(ToolBase):
    """
    Narzędzie pomiaru odległości:
      - LPM down: p0 (image-space)
      - Move:    podgląd linii p0->p1
      - LPM up:  p1 (image-space) + publikacja ui.image.measure.done

    Publikuje:
      ui.image.measure.done {
        "p0": [x0,y0],
        "p1": [x1,y1],
        "length_px": float,
        "angle_deg": float     # 0..180, 0 = poziomo w prawo
      }
    """
    name = "measure"

    def __init__(self, ctx: ToolEventContext) -> None:
        super().__init__(ctx)
        self._p0_img: Optional[Tuple[int, int]] = None
        self._p1_img: Optional[Tuple[int, int]] = None
        self._preview: bool = False
        # opcje
        self._snap_ortho: bool = True   # przytrzymany SHIFT -> przyciągaj do 0/45/90
        self._snap_step_deg: float = 45.0

    # ── cykl życia ────────────────────────────────────────────────────────────
    def on_activate(self, opts: Optional[Dict] = None) -> None:
        super().on_activate(opts)
        if opts:
            self._snap_ortho = bool(opts.get("snap_ortho", True))
            self._snap_step_deg = float(opts.get("snap_step_deg", 45.0))
        self._p0_img = None
        self._p1_img = None
        self._preview = False
        self.ctx.invalidate(None)

    def on_deactivate(self) -> None:
        super().on_deactivate()
        self._p0_img = None
        self._p1_img = None
        self._preview = False
        self.ctx.invalidate(None)

    # ── zdarzenia myszy ───────────────────────────────────────────────────────
    def on_mouse_down(self, ev: Any) -> None:
        super().on_mouse_down(ev)
        ix, iy = self.ctx.to_image_xy(int(ev.x), int(ev.y))
        self._p0_img = (ix, iy)
        self._p1_img = (ix, iy)
        self._preview = True
        self.ctx.invalidate(None)

    def on_mouse_move(self, ev: Any) -> None:
        if not self._active or self._p0_img is None:
            return
        ix, iy = self.ctx.to_image_xy(int(ev.x), int(ev.y))
        x0, y0 = self._p0_img

        # SHIFT -> przyciągaj do najbliższego kąta (0/45/90/...)
        snap = False
        try:
            snap = (getattr(ev, "state", 0) & 0x0001) != 0 and self._snap_ortho  # Shift
        except Exception:
            snap = False
        if snap:
            dx = ix - x0
            dy = iy - y0
            ang = math.degrees(math.atan2(dy, dx)) if (dx or dy) else 0.0
            step = max(1.0, self._snap_step_deg)
            snapped = round(ang / step) * step
            r = math.hypot(dx, dy)
            rad = math.radians(snapped)
            ix = int(round(x0 + r * math.cos(rad)))
            iy = int(round(y0 + r * math.sin(rad)))

        self._p1_img = (ix, iy)
        self._preview = True
        self.ctx.invalidate(None)

    def on_mouse_up(self, ev: Any) -> None:
        super().on_mouse_up(ev)
        if self._p0_img is None or self._p1_img is None:
            return
        x0, y0 = self._p0_img
        x1, y1 = self._p1_img
        length = float(math.hypot(x1 - x0, y1 - y0))
        angle = float((math.degrees(math.atan2(y1 - y0, x1 - x0)) + 360.0) % 180.0)
        self.ctx.publish("ui.image.measure.done", {
            "p0": [x0, y0],
            "p1": [x1, y1],
            "length_px": length,
            "angle_deg": angle,
        })
        # zostawiamy linię do czasu kolejnego kliknięcia jako wizualną referencję
        self._preview = False
        self.ctx.invalidate(None)

    # ── overlay ───────────────────────────────────────────────────────────────
    def draw_overlay(self, tk_canvas: Any) -> None:
        if self._p0_img is None or self._p1_img is None:
            return
        zoom, (pan_x, pan_y) = self.ctx.get_zoom_pan()

        def img_to_screen(ix: int, iy: int) -> Tuple[int, int]:
            sx = int(round(ix * zoom + pan_x))
            sy = int(round(iy * zoom + pan_y))
            return sx, sy

        x0, y0 = self._p0_img
        x1, y1 = self._p1_img
        sx0, sy0 = img_to_screen(x0, y0)
        sx1, sy1 = img_to_screen(x1, y1)

        # Linia + uchwyty końcowe
        color = "#FFD54F" if self._preview else "#FFB300"
        tk_canvas.create_line(sx0, sy0, sx1, sy1, fill=color, width=1)

        r = max(2, int(round(3 * zoom)))
        for cx, cy in ((sx0, sy0), (sx1, sy1)):
            tk_canvas.create_oval(cx - r, cy - r, cx + r, cy + r, outline=color, width=1)

        # Etykieta: odległość i kąt
        length = math.hypot(x1 - x0, y1 - y0)
        angle = (math.degrees(math.atan2(y1 - y0, x1 - x0)) + 360.0) % 180.0
        label = f"{length:.1f}px  {angle:.1f}°"
        # umieść etykietę blisko środka linii
        mx = (sx0 + sx1) // 2
        my = (sy0 + sy1) // 2
        tk_canvas.create_rectangle(mx + 6, my - 10, mx + 6 + 70, my + 10, fill="#000000", outline="")
        tk_canvas.create_text(mx + 6 + 4, my, anchor="w", fill="#FFFFFF", text=label)
