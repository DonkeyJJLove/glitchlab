# glitchlab/app/widgets/tools/tool_view.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from .base import ToolBase, ToolEventContext


class ViewTool(ToolBase):
    """
    Narzędzie 'View/Hand' dla ImageCanvas:
      - Pan: przeciąganie lewym lub środkowym przyciskiem
      - Zoom: kółkiem myszy, z zachowaniem punktu pod kursorem
    Publikuje: ui.image.view.changed { zoom, pan:[x,y], viewport:[x0,y0,x1,y1] }
    """
    name = "view"

    # ustawienia
    ZOOM_MIN = 0.05
    ZOOM_MAX = 32.0
    ZOOM_STEP = 1.1  # multiplikatywnie (wheel)

    def __init__(self, ctx: ToolEventContext) -> None:
        super().__init__(ctx)
        self._dragging = False
        self._drag_start_screen: Optional[Tuple[int, int]] = None
        self._drag_start_pan: Optional[Tuple[int, int]] = None

    # ── cykl życia ────────────────────────────────────────────────────────────
    def on_activate(self, opts: Optional[Dict[str, Any]] = None) -> None:
        super().on_activate(opts)
        self._dragging = False
        self._drag_start_screen = None
        self._drag_start_pan = None

    def on_deactivate(self) -> None:
        super().on_deactivate()
        self._dragging = False
        self._drag_start_screen = None
        self._drag_start_pan = None

    # ── zdarzenia myszy ───────────────────────────────────────────────────────
    def on_mouse_down(self, ev: Any) -> None:
        # LPM lub środkowy = pan drag
        super().on_mouse_down(ev)
        self._dragging = True
        self._drag_start_screen = (int(ev.x), int(ev.y))
        _, pan = self.ctx.get_zoom_pan()
        self._drag_start_pan = pan

    def on_mouse_move(self, ev: Any) -> None:
        if not self._dragging or self._drag_start_screen is None or self._drag_start_pan is None:
            return
        sx0, sy0 = self._drag_start_screen
        dx = int(ev.x) - sx0
        dy = int(ev.y) - sy0
        zoom, (pan_x, pan_y) = self.ctx.get_zoom_pan()

        # Pan w screen-space -> image-space; przyjmujemy skalowanie 1/zoom
        new_pan = (self._drag_start_pan[0] + dx, self._drag_start_pan[1] + dy)
        self._publish_view_changed(zoom, new_pan)

    def on_mouse_up(self, ev: Any) -> None:
        super().on_mouse_up(ev)
        self._dragging = False
        self._drag_start_screen = None
        self._drag_start_pan = None

    def on_wheel(self, ev: Any) -> None:
        """
        Zoom wokół punktu pod kursorem. W Tk 'delta' bywa +/-120*k (Windows),
        na macOS to małe wartości; na X11 często używa się Button-4/5 (obsługiwane w ImageCanvas).
        Zakładamy, że ImageCanvas route’uje tutaj event z polami .x, .y, .delta (opcjonalnie .num=4/5).
        """
        # Ustal kierunek zoomu
        dz = 0
        if hasattr(ev, "delta") and ev.delta:
            dz = 1 if ev.delta > 0 else -1
        elif getattr(ev, "num", None) in (4, 5):  # X11 emulacja
            dz = 1 if ev.num == 4 else -1
        if dz == 0:
            return

        zoom, (pan_x, pan_y) = self.ctx.get_zoom_pan()
        factor = (self.ZOOM_STEP if dz > 0 else 1.0 / self.ZOOM_STEP)
        new_zoom = max(self.ZOOM_MIN, min(self.ZOOM_MAX, zoom * factor))
        if abs(new_zoom - zoom) < 1e-6:
            return

        # Zachowaj punkt pod kursorem: (sx,sy) -> (ix,iy)
        sx, sy = int(ev.x), int(ev.y)
        ix, iy = self.ctx.to_image_xy(sx, sy)  # koordy w space obrazu dla starego zoom/pan

        # Dla nowego zoomu: wylicz taki pan, by (ix,iy) pozostał pod kursorem (sx,sy).
        # to_screen(ix,iy) = (ix*new_zoom + new_pan_x, iy*new_zoom + new_pan_y)
        new_pan_x = sx - int(round(ix * new_zoom))
        new_pan_y = sy - int(round(iy * new_zoom))

        self._publish_view_changed(new_zoom, (new_pan_x, new_pan_y))

    # ── overlay ───────────────────────────────────────────────────────────────
    def draw_overlay(self, tk_canvas: Any) -> None:
        # ViewTool nie rysuje overlay (opcjonalnie: siatka, crosshair po włączeniu)
        return

    # ── pomocnicze ────────────────────────────────────────────────────────────
    def _publish_view_changed(self, zoom: float, pan_xy: Tuple[int, int]) -> None:
        # viewport (x0,y0,x1,y1) – opcjonalnie: emitujemy None, ImageCanvas może wypełnić
        payload = {
            "zoom": float(zoom),
            "pan": [int(pan_xy[0]), int(pan_xy[1])],
            "viewport": None,
        }
        self.ctx.publish("ui.image.view.changed", payload)
        # prosimy tylko o przerys overlay (bitmapę zwykle przerysuje wyższa warstwa po zmianie stanu)
        self.ctx.invalidate(None)
