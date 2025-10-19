# glitchlab/app/widgets/tools/tool_brush.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import numpy as np

from .base import ToolBase, ToolEventContext


class BrushTool(ToolBase):
    """
    Pędzel do malowania w aktywnej masce:
      - LPM down/move: nanoszenie stempla (okrąg) z miękkością (hardness)
      - LPM up: publikacja ui.image.mask.updated z bbox i px_changed

    Opcje (opts przy on_activate):
      - size: int (średnica px, domyślnie 24)
      - hardness: float 0..1 (0=miękki, 1=twardy), domyślnie 0.7
      - strength: float 0..1 (siła narastania wartości maski), domyślnie 1.0
      - spacing: float 0..1 (proporcja do rozmiaru pędzla, domyślnie 0.4)

    Założenia maski:
      - dtype uint8 (0/255). Gdy napotkamy float [0,1], przeliczymy do uint8.
    """
    name = "brush"

    def __init__(self, ctx: ToolEventContext) -> None:
        super().__init__(ctx)
        # stan pociągnięcia
        self._p_last: Optional[Tuple[int, int]] = None
        self._dirty_bbox: Optional[Tuple[int, int, int, int]] = None
        self._px_changed_total: int = 0

        # parametry pędzla
        self.size: int = 24
        self.hardness: float = 0.7
        self.strength: float = 1.0
        self.spacing: float = 0.4  # w ułamku średnicy

        # gotowy kernel (odświeżany gdy zmieniają się parametry)
        self._kernel_u8: Optional[np.ndarray] = None  # uint8 0..255
        self._kernel_r: int = 0  # promień (dla bbox)

    # ── cykl życia ────────────────────────────────────────────────────────────
    def on_activate(self, opts: Optional[Dict] = None) -> None:
        super().on_activate(opts)
        if opts:
            self.size = max(1, int(opts.get("size", self.size)))
            self.hardness = float(min(1.0, max(0.0, opts.get("hardness", self.hardness))))
            self.strength = float(min(1.0, max(0.0, opts.get("strength", self.strength))))
            self.spacing = float(min(1.0, max(0.05, opts.get("spacing", self.spacing))))
        self._rebuild_kernel()
        self._p_last = None
        self._dirty_bbox = None
        self._px_changed_total = 0
        self.ctx.invalidate(None)

    def on_deactivate(self) -> None:
        super().on_deactivate()
        self._p_last = None
        self.ctx.invalidate(None)

    # ── zdarzenia myszy ───────────────────────────────────────────────────────
    def on_mouse_down(self, ev: Any) -> None:
        super().on_mouse_down(ev)
        ix, iy = self.ctx.to_image_xy(int(ev.x), int(ev.y))
        self._ensure_mask_alloc()
        self._stamp(ix, iy)
        self._p_last = (ix, iy)
        self.ctx.invalidate(None)

    def on_mouse_move(self, ev: Any) -> None:
        if not self._active or self._p_last is None:
            return
        ix, iy = self.ctx.to_image_xy(int(ev.x), int(ev.y))
        self._stroke_line(self._p_last, (ix, iy))
        self._p_last = (ix, iy)
        self.ctx.invalidate(None)

    def on_mouse_up(self, ev: Any) -> None:
        super().on_mouse_up(ev)
        # Publikacja podsumowania pociągnięcia
        if self._dirty_bbox is not None:
            x0, y0, x1, y1 = self._dirty_bbox
            self.ctx.publish("ui.image.mask.updated", {
                "mask_key": "current",
                "op": "paint",
                "bbox": [x0, y0, x1, y1],
                "stats": {"px_changed": int(self._px_changed_total)},
            })
        self._p_last = None
        self._dirty_bbox = None
        self._px_changed_total = 0
        self.ctx.invalidate(None)

    # ── overlay (podgląd dysku pędzla) ────────────────────────────────────────
    def draw_overlay(self, tk_canvas: Any) -> None:
        try:
            # narysuj okrąg pędzla w miejscu kursora (jeśli wewnątrz okna)
            # Uwaga: ToolBase nie daje hover; rely na ostatnim punkcie drag jako proxy
            # Opcjonalnie można włączyć śledzenie <Motion> także bez drag w ImageCanvas i routować tu.
            if self._p_last is None:
                return
            zoom, (pan_x, pan_y) = self.ctx.get_zoom_pan()
            r = int(round((self._kernel_r) * zoom))
            cx = int(round(self._p_last[0] * zoom + pan_x))
            cy = int(round(self._p_last[1] * zoom + pan_y))
            tk_canvas.create_oval(
                cx - r, cy - r, cx + r, cy + r,
                outline="#FFEE58", width=1
            )
        except Exception:
            # overlay jest niekrytyczny — nie wysadzamy narzędzia
            pass

    # ── logika pędzla ────────────────────────────────────────────────────────
    def _rebuild_kernel(self) -> None:
        """Buduje kernel okrągły z miękkością. Zwraca uint8 0..255."""
        d = max(1, int(self.size))
        r = d // 2
        self._kernel_r = r
        yy, xx = np.mgrid[-r:r+1, -r:r+1]
        rr = np.sqrt(xx * xx + yy * yy)
        # profil: wewnątrz r -> [hardness..1], na brzegu -> 0
        # hardness=1 => twardy brzeg; hardness=0 => pełen gradient
        hard = float(self.hardness)
        inner = hard * r
        k = np.zeros_like(rr, dtype=np.float32)
        if r > 0:
            # część wewnętrzna (pełna siła)
            k[rr <= inner] = 1.0
            # przejście do zera na zewnątrz
            ring = (rr > inner) & (rr <= r + 1e-6)
            k[ring] = np.clip(1.0 - (rr[ring] - inner) / max(1e-6, (r - inner)), 0.0, 1.0)
        # siła pędzla (strength) skaluje wartości
        k *= float(self.strength)
        # do uint8 0..255
        self._kernel_u8 = (np.clip(k, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

    def _ensure_mask_alloc(self) -> None:
        m = self.ctx.get_mask()
        if m is None:
            # utwórz maskę dopasowaną do obrazu
            img = self.ctx.get_image()
            if img is None:
                return
            if hasattr(img, "size"):  # PIL.Image
                w, h = img.size
            else:  # ndarray
                h, w = img.shape[:2]
            m = np.zeros((h, w), dtype=np.uint8)
            self.ctx.set_mask(m)
        else:
            # ewentualna normalizacja dtype
            if m.dtype != np.uint8:
                m = (np.clip(m.astype(np.float32), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
                self.ctx.set_mask(m)

    def _stamp(self, cx: int, cy: int) -> None:
        """Naniesienie jednego stempla kernelu na maskę (clamped add/max)."""
        m = self.ctx.get_mask()
        if m is None or self._kernel_u8 is None:
            return

        h, w = m.shape[:2]
        r = self._kernel_r
        x0 = max(0, cx - r)
        y0 = max(0, cy - r)
        x1 = min(w, cx + r + 1)
        y1 = min(h, cy + r + 1)
        if x1 <= x0 or y1 <= y0:
            return

        kx0 = x0 - (cx - r)
        ky0 = y0 - (cy - r)
        kx1 = kx0 + (x1 - x0)
        ky1 = ky0 + (y1 - y0)

        sub = m[y0:y1, x0:x1]
        ker = self._kernel_u8[ky0:ky1, kx0:kx1]

        # Tryb "paint": podnosimy maskę w kierunku 255 metodą max (działa jak alfa w dół)
        before = sub.copy()
        np.maximum(sub, ker, out=sub)

        # zlicz zmienione piksele (dla telemetrii)
        changed = int((sub != before).sum())
        self._px_changed_total += changed

        # zapis z powrotem (sub jest view, już zmodyfikowany)
        m[y0:y1, x0:x1] = sub
        self.ctx.set_mask(m)

        # rozszerz dirty bbox
        self._grow_dirty_bbox(x0, y0, x1, y1)

    def _stroke_line(self, p0: Tuple[int, int], p1: Tuple[int, int]) -> None:
        """Interpolacja stempli między dwoma punktami."""
        x0, y0 = p0
        x1, y1 = p1
        dx = x1 - x0
        dy = y1 - y0
        dist = max(abs(dx), abs(dy))
        if dist == 0:
            self._stamp(x0, y0)
            return
        # krok co 'spacing * size'
        step = max(1.0, float(self.spacing) * float(self.size))
        n = int(np.ceil(dist / step))
        for i in range(n + 1):
            t = 0.0 if n == 0 else i / float(n)
            xi = int(round(x0 + t * dx))
            yi = int(round(y0 + t * dy))
            self._stamp(xi, yi)

    def _grow_dirty_bbox(self, x0: int, y0: int, x1: int, y1: int) -> None:
        if self._dirty_bbox is None:
            self._dirty_bbox = (x0, y0, x1, y1)
        else:
            a0, b0, a1, b1 = self._dirty_bbox
            self._dirty_bbox = (min(a0, x0), min(b0, y0), max(a1, x1), max(b1, y1))
