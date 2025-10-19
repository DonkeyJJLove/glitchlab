# glitchlab/app/widgets/tools/tool_pipette.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import time
import numpy as np

from .base import ToolBase, ToolEventContext


def _clip_xy(ix: int, iy: int, w: int, h: int) -> Tuple[int, int]:
    return max(0, min(ix, w - 1)), max(0, min(iy, h - 1))


def _rgb_at(img) -> Tuple[int, int, int, int, int]:
    """
    Zwraca: (r,g,b,w,h) dla obrazu kompozytowanego (PIL.Image lub ndarray).
    Uwaga: ta funkcja zwraca tylko W,H (nie odczytuje piksela).
    """
    if img is None:
        return 0, 0, 0, 0, 0
    if hasattr(img, "size"):  # PIL.Image
        w, h = img.size
    else:
        # ndarray (H,W,3)
        h, w = img.shape[:2]
    return 0, 0, 0, w, h


def _read_rgb(img, ix: int, iy: int) -> Tuple[int, int, int]:
    if hasattr(img, "getpixel"):
        r, g, b = img.getpixel((ix, iy))[:3]
        return int(r), int(g), int(b)
    # ndarray
    px = img[iy, ix]
    r, g, b = int(px[0]), int(px[1]), int(px[2])
    return r, g, b


def _read_mask_bit(mask: Optional[np.ndarray], ix: int, iy: int) -> int:
    if mask is None:
        return 0
    v = mask[iy, ix]
    if mask.dtype != np.uint8:
        v = int(np.clip(float(v), 0.0, 1.0) * 255.0 + 0.5)
    return 1 if v >= 128 else 0


def _luma_y(r: int, g: int, b: int) -> float:
    # Rec. 601 approx
    return 0.299 * r + 0.587 * g + 0.114 * b


class PipetteTool(ToolBase):
    """
    Pipeta:
      - hover: emituje ui.image.cursor.hover (throttling ~75ms),
      - klik LPM: emituje ui.image.pixel.probe (pełne dane),
      - overlay: krzyż + próbnik koloru obok kursora.
    """
    name = "pipette"

    def __init__(self, ctx: ToolEventContext) -> None:
        super().__init__(ctx)
        self._last_emit_t: float = 0.0
        self._hover_xy_img: Optional[Tuple[int, int]] = None
        self._hover_rgb: Optional[Tuple[int, int, int]] = None
        self._hover_mask_bit: int = 0
        self._throttle_sec: float = 0.075  # ~13 Hz

    # ── aktywacja / dezaktywacja ─────────────────────────────────────────────
    def on_activate(self, opts: Optional[Dict] = None) -> None:
        super().on_activate(opts)
        self._last_emit_t = 0.0
        self._hover_xy_img = None
        self._hover_rgb = None
        self._hover_mask_bit = 0
        self.ctx.invalidate(None)

    def on_deactivate(self) -> None:
        super().on_deactivate()
        self._hover_xy_img = None
        self._hover_rgb = None
        self.ctx.invalidate(None)

    # ── zdarzenia myszy ───────────────────────────────────────────────────────
    def on_mouse_move(self, ev: Any) -> None:
        # Obsługujemy zarówno hover (bez trzymania LPM), jak i drag (nieistotne dla pipety).
        img = self.ctx.get_image()
        if img is None:
            return
        _, _, _, w, h = _rgb_at(img)
        ix, iy = self.ctx.to_image_xy(int(ev.x), int(ev.y))
        ix, iy = _clip_xy(ix, iy, w, h)

        rgb = _read_rgb(img, ix, iy)
        m = self.ctx.get_mask()
        mask_bit = _read_mask_bit(m, ix, iy)

        self._hover_xy_img = (ix, iy)
        self._hover_rgb = rgb
        self._hover_mask_bit = mask_bit

        now = time.monotonic()
        if (now - self._last_emit_t) >= self._throttle_sec:
            self._last_emit_t = now
            payload = {"xy": [ix, iy], "rgb": list(rgb), "mask": int(mask_bit), "value": float(_luma_y(*rgb))}
            self.ctx.publish("ui.image.cursor.hover", payload)

        # odśwież overlay (krzyżyk/próbnik)
        self.ctx.invalidate(None)

    def on_mouse_down(self, ev: Any) -> None:
        super().on_mouse_down(ev)
        # „klik” = próbka
        img = self.ctx.get_image()
        if img is None:
            return
        _, _, _, w, h = _rgb_at(img)
        ix, iy = self.ctx.to_image_xy(int(ev.x), int(ev.y))
        ix, iy = _clip_xy(ix, iy, w, h)
        rgb = _read_rgb(img, ix, iy)
        mask_bit = _read_mask_bit(self.ctx.get_mask(), ix, iy)
        payload = {
            "xy": [ix, iy],
            "rgb": list(rgb),
            "value": float(_luma_y(*rgb)),
            "mask": int(mask_bit),
        }
        self.ctx.publish("ui.image.pixel.probe", payload)
        # zachowaj też do overlayu
        self._hover_xy_img = (ix, iy)
        self._hover_rgb = rgb
        self._hover_mask_bit = mask_bit
        self.ctx.invalidate(None)

    def on_mouse_up(self, ev: Any) -> None:
        super().on_mouse_up(ev)
        # nic trwałego; overlay zostaje aż do ruchu
        self.ctx.invalidate(None)

    # ── overlay ───────────────────────────────────────────────────────────────
    def draw_overlay(self, tk_canvas: Any) -> None:
        if self._hover_xy_img is None or self._hover_rgb is None:
            return
        zoom, (pan_x, pan_y) = self.ctx.get_zoom_pan()
        ix, iy = self._hover_xy_img
        sx = int(round(ix * zoom + pan_x))
        sy = int(round(iy * zoom + pan_y))

        # krzyż
        s = max(5, int(round(6 * zoom)))
        tk_canvas.create_line(sx - s, sy, sx + s, sy, fill="#FFFFFF")
        tk_canvas.create_line(sx, sy - s, sx, sy + s, fill="#FFFFFF")

        # próbnik koloru
        r, g, b = self._hover_rgb
        side = max(10, int(round(14 * zoom)))
        pad = max(6, int(round(8 * zoom)))
        x0 = sx + pad
        y0 = sy + pad
        x1 = x0 + side
        y1 = y0 + side
        hex_col = f"#{r:02X}{g:02X}{b:02X}"
        tk_canvas.create_rectangle(x0, y0, x1, y1, outline="#FFFFFF", width=1, fill=hex_col)

        # opis: RGB i bit maski
        label = f"{r},{g},{b}  m:{self._hover_mask_bit}"
        tk_canvas.create_text(x1 + max(6, int(6 * zoom)), y0 + side // 2, anchor="w", fill="#EEEEEE", text=label, font=("TkDefaultFont", max(8, int(9 * zoom))))
