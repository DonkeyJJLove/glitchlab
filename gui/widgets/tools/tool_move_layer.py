# glitchlab/gui/widgets/tools/tool_move_layer.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from .base import ToolBase, ToolEventContext


class MoveLayerTool(ToolBase):
    """
    Narzędzie: Move Layer (przesuwanie aktywnej warstwy po obrazie).

    Założenia:
    - Nie dotykamy bezpośrednio warstw w GUI — publikujemy zdarzenia na EventBus.
    - Warstwa docelowa to „aktywna” (obsługiwana przez LayerManager/App).
    - Przesuwamy w układzie współrzędnych OBRAZU (image-space), niezależnie od zoom/pan.

    Zdarzenia:
    - preview (w trakcie przeciągania):
        topic: "ui.layer.move.preview"
        payload: { "target": "active", "dx": int, "dy": int }
    - commit (puszczenie LPM – finalizacja):
        topic: "ui.layer.move.commit"
        payload: { "target": "active", "dx": int, "dy": int }
    - cancel (ESC w trakcie przeciągania – odrzucenie):
        topic: "ui.layer.move.cancel"
        payload: { "target": "active" }

    Wspierane modyfikatory:
    - SHIFT: blokada osi (największa |dx| lub |dy| zostaje, druga => 0).
    """

    name = "move"

    def __init__(self, ctx: ToolEventContext) -> None:
        super().__init__(ctx)
        self._dragging: bool = False
        self._p0_img: Optional[Tuple[int, int]] = None  # punkt startowy (image-space)
        self._p_img: Optional[Tuple[int, int]] = None   # bieżący punkt (image-space)
        self._acc_dxdy: Tuple[int, int] = (0, 0)        # skumulowane delta

    # ───────────────────────── lifecycle ─────────────────────────

    def on_activate(self, opts: Optional[Dict] = None) -> None:
        super().on_activate(opts)
        self._dragging = False
        self._p0_img = None
        self._p_img = None
        self._acc_dxdy = (0, 0)
        self.ctx.invalidate(None)

    def on_deactivate(self) -> None:
        super().on_deactivate()
        # brak auto-commit — jeśli użytkownik porzuci narzędzie, warstwa zostaje bez zmian preview
        self._dragging = False
        self._p0_img = None
        self._p_img = None
        self._acc_dxdy = (0, 0)
        self.ctx.invalidate(None)

    # ───────────────────────── mouse ─────────────────────────

    def on_mouse_down(self, ev: Any) -> None:
        super().on_mouse_down(ev)
        # start przeciągania w koordynatach obrazu
        ix, iy = self.ctx.to_image_xy(int(ev.x), int(ev.y))
        self._p0_img = (ix, iy)
        self._p_img = (ix, iy)
        self._acc_dxdy = (0, 0)
        self._dragging = True
        self.ctx.invalidate(None)

    def on_mouse_move(self, ev: Any) -> None:
        if not (self._active and self._dragging and self._p0_img is not None):
            return

        ix, iy = self.ctx.to_image_xy(int(ev.x), int(ev.y))
        x0, y0 = self._p0_img
        dx = ix - x0
        dy = iy - y0

        # SHIFT => blokada osi (większa składowa zostaje)
        try:
            shift = (getattr(ev, "state", 0) & 0x0001) != 0
        except Exception:
            shift = False
        if shift:
            if abs(dx) >= abs(dy):
                dy = 0
            else:
                dx = 0

        self._p_img = (ix, iy)
        self._acc_dxdy = (dx, dy)

        # preview przesunięcia aktywnej warstwy
        self.ctx.publish("ui.layer.move.preview", {
            "target": "active",
            "dx": int(dx),
            "dy": int(dy),
        })
        self.ctx.invalidate(None)

    def on_mouse_up(self, ev: Any) -> None:
        super().on_mouse_up(ev)
        if not self._dragging:
            return

        dx, dy = self._acc_dxdy
        # commit przesunięcia (finalizacja)
        self.ctx.publish("ui.layer.move.commit", {
            "target": "active",
            "dx": int(dx),
            "dy": int(dy),
        })

        # reset
        self._dragging = False
        self._p0_img = None
        self._p_img = None
        self._acc_dxdy = (0, 0)
        self.ctx.invalidate(None)

    # ───────────────────────── keyboard / wheel ─────────────────────────

    def on_key(self, ev: Any) -> None:
        # ESC podczas przeciągania — anuluj preview
        key = str(getattr(ev, "keysym", "")).lower()
        if key == "escape" and self._dragging:
            self._dragging = False
            self._p0_img = None
            self._p_img = None
            self._acc_dxdy = (0, 0)
            self.ctx.publish("ui.layer.move.cancel", {"target": "active"})
            self.ctx.invalidate(None)

    # ───────────────────────── overlay ─────────────────────────

    def draw_overlay(self, tk_canvas: Any) -> None:
        """
        Rysuje strzałkę wektora przesunięcia (od punktu startowego do bieżącego),
        aby dać użytkownikowi natychmiastowy feedback. Overlay jest w screen-space,
        dlatego trzeba przeliczać z image-space przez get_zoom_pan().
        """
        if not (self._dragging and self._p0_img and self._p_img):
            return

        (zoom, (pan_x, pan_y)) = self.ctx.get_zoom_pan()

        def img_to_screen(ix: int, iy: int) -> Tuple[int, int]:
            sx = int(round(ix * zoom + pan_x))
            sy = int(round(iy * zoom + pan_y))
            return sx, sy

        x0, y0 = self._p0_img
        x1, y1 = self._p_img
        sx0, sy0 = img_to_screen(x0, y0)
        sx1, sy1 = img_to_screen(x1, y1)

        # linia wektora
        tk_canvas.create_line(sx0, sy0, sx1, sy1, fill="#FFCA28", width=2)

        # grot strzałki
        dx = sx1 - sx0
        dy = sy1 - sy0
        L = (dx * dx + dy * dy) ** 0.5
        if L >= 1.0:
            ux, uy = dx / L, dy / L
            # dwa punkty boczne grota
            gx = sx1 - 10 * ux
            gy = sy1 - 10 * uy
            left = (gx - 5 * uy, gy + 5 * ux)
            right = (gx + 5 * uy, gy - 5 * ux)
            tk_canvas.create_polygon(
                (sx1, sy1, int(left[0]), int(left[1]), int(right[0]), int(right[1])),
                fill="#FFCA28", outline="#FFB300"
            )

        # opis delta (px)
        dx_img, dy_img = self._acc_dxdy
        label = f"{dx_img:+d}, {dy_img:+d}px"
        tk_canvas.create_rectangle(sx1 + 8, sy1 - 14, sx1 + 8 + 72, sy1 + 4, fill="#00000080", outline="")
        tk_canvas.create_text(sx1 + 12, sy1 - 6, anchor="nw", text=label, fill="#FFD54F")
