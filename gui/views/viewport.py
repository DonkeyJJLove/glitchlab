# glitchlab/gui/views/viewport.py
# -*- coding: utf-8 -*-
"""
Viewport — zunifikowany viewer 2D (Canvas + overlay: crosshair & rulers).

Cechy:
• Jednolite API: set_image(), set_enabled(), get_zoom()/set_zoom()/zoom_fit(), center(),
  screen_to_image(), set_crosshair().
• Obsługa narzędzi przez EventBus: ui.tools.select (pan/zoom/ruler/probe/pick).
• Pan/Zoom na Canvasie (wheel = zoom, LMB drag = pan gdy tool=pan).
• Ruler’y (góra/lewo) reagujące na zoom; crosshair nad obrazem; publikacja ui.cursor.pos.
• Działa z opcjonalnym ImageCanvas (jeśli dostępny) lub w trybie fallback (czysty Canvas).

Uwaga:
To jest lekki kontener UI — nie ma własnej historii obrazu; historię trzyma wyższa warstwa.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import tkinter as tk
from tkinter import ttk

# ─────────────────── opcjonalne zależności (Pillow) ───────────────────
try:
    from PIL import Image, ImageTk  # type: ignore
except Exception:
    Image = None  # type: ignore
    ImageTk = None  # type: ignore

# ─────────────────── opcjonalny ImageCanvas (jeśli istnieje) ──────────
try:
    from glitchlab.gui.widgets.image_canvas import ImageCanvas  # type: ignore
except Exception:
    ImageCanvas = None  # type: ignore


class Viewport(ttk.Frame):
    """
    Viewer 2D z overlayem (crosshair + rulers) i pan/zoom.
    """

    # UI stałe
    RULER_TOP_H = 26
    RULER_LEFT_W = 32
    BG_VIEWPORT = "#101015"
    BG_RULER = "#1b1b1f"
    FG_RULER = "#9a9aa2"
    FG_RULER_MINOR = "#5b5b66"
    FG_CROSS = "#66aaff"

    TOOLS = ("pan", "zoom", "ruler", "probe", "pick")

    def __init__(
        self,
        master: tk.Misc,
        *,
        bus: Optional[Any] = None,
        tool_var: Optional[tk.StringVar] = None,
    ) -> None:
        super().__init__(master)
        self.bus = bus
        self.tool_var = tool_var or tk.StringVar(value="pan")

        # runtime
        self._enabled = True
        self._tool = self.tool_var.get()
        self._img_orig: Optional["Image.Image"] = None
        self._scaled_ref: Optional["Image.Image"] = None  # kopia po skali (PIL)
        self._tk_img: Optional["ImageTk.PhotoImage"] = None  # keepref
        self._scale = 1.0
        self._pan_origin: Optional[Tuple[int, int]] = None

        # siatka 2×2: (mask corner + top ruler) / (left ruler + viewport)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(1, weight=1)

        # narożne maski (wypełnienie)
        self._corner = tk.Canvas(self, width=self.RULER_LEFT_W, height=self.RULER_TOP_H,
                                 bg=self.BG_RULER, highlightthickness=0)
        self._corner.grid(row=0, column=0)

        self._ruler_top = tk.Canvas(self, height=self.RULER_TOP_H,
                                    bg=self.BG_RULER, highlightthickness=0)
        self._ruler_top.grid(row=0, column=1, sticky="ew")

        self._ruler_left = tk.Canvas(self, width=self.RULER_LEFT_W,
                                     bg=self.BG_RULER, highlightthickness=0)
        self._ruler_left.grid(row=1, column=0, sticky="ns")

        # główny viewport
        self._viewport = tk.Canvas(self, bg=self.BG_VIEWPORT, highlightthickness=0)
        self._viewport.grid(row=1, column=1, sticky="nsew")

        # wewnątrz viewportu osadzamy ImageCanvas (jeśli dostępny) lub rysujemy obraz samodzielnie
        if ImageCanvas is not None:
            self._image_widget = ImageCanvas(self._viewport)  # type: ignore[assignment]
            self._image_win = self._viewport.create_window(0, 0, window=self._image_widget, anchor="center")
        else:
            self._image_widget = None
            self._image_item = None  # ID obrazka na Canvasie fallback
            self._image_win = None

        # crosshair (linie)
        self._cross_h = self._viewport.create_line(0, 0, 0, 0, fill=self.FG_CROSS)
        self._cross_v = self._viewport.create_line(0, 0, 0, 0, fill=self.FG_CROSS)
        self._cross_enabled = True

        self._wire_events()
        self._wire_bus()

    # ──────────────────────────── BUS / TOOLS ────────────────────────────

    def _wire_bus(self) -> None:
        if not self.bus or not hasattr(self.bus, "subscribe"):
            return
        try:
            # aktualizacja aktywnego narzędzia (z menu lub innego panelu)
            self.bus.subscribe("ui.tools.select",
                               lambda _t, d: self._set_tool(str((d or {}).get("name", ""))))
        except Exception:
            pass

    # ──────────────────────────── EVENTS ─────────────────────────────────

    def _wire_events(self) -> None:
        vp = self._viewport
        vp.bind("<Configure>", lambda _e: (self._resize_and_center(), self._redraw_rulers()))

        # kółko — zoom
        vp.bind("<MouseWheel>", self._on_wheel)     # Windows/mac
        vp.bind("<Button-4>", self._on_wheel)       # X11 up
        vp.bind("<Button-5>", self._on_wheel)       # X11 down

        # pan (LMB)
        vp.bind("<ButtonPress-1>", self._on_btn1_down)
        vp.bind("<B1-Motion>", self._on_btn1_drag)
        vp.bind("<ButtonRelease-1>", self._on_btn1_up)

        # kursor — crosshair & publikacja pozycji
        vp.bind("<Motion>", self._on_motion)
        vp.bind("<Leave>", self._on_leave)

        # zsynchronizuj narzędzie z tk.StringVar
        self.tool_var.trace_add("write", lambda *_: self._set_tool(self.tool_var.get()))

    # ──────────────────────────── PUBLIC API ─────────────────────────────

    def set_image(self, pil_img: "Image.Image") -> None:
        """Ustaw obraz oryginalny; dopasuj do okna i odrysuj overlay."""
        if Image is None:
            return
        self._img_orig = pil_img.convert("RGB") if getattr(pil_img, "mode", "") != "RGB" else pil_img
        self._fit_to_window()
        self._redraw_rulers()

    def set_enabled(self, flag: bool) -> None:
        """Gdy False — blokuj interakcje (podczas pracy asynchronicznej)."""
        self._enabled = bool(flag)

    # zoom/pan API
    def get_zoom(self) -> float:
        return float(self._scale)

    def set_zoom(self, z: float) -> None:
        self._scale = float(max(1e-3, min(16.0, z)))
        self._render_scaled()
        self._resize_and_center()
        self._redraw_rulers()

    def zoom_fit(self) -> None:
        self._fit_to_window()

    def center(self) -> None:
        self._resize_and_center()

    def screen_to_image(self, x: int, y: int) -> Tuple[int, int]:
        """Konwersja współrzędnych ekranu na współrzędne w obrazie (po skali)."""
        if not self._img_orig:
            return (x, y)
        # pobierz rzeczywistą pozycję okienka z obrazem
        vx, vy = self._viewport.coords(self._image_win) if self._image_win else (0, 0)
        # środek obrazu jest w (vx, vy)
        iw, ih = self._img_orig.size
        w, h = int(iw * self._scale), int(ih * self._scale)
        # lewy-górny róg obrazka:
        ox, oy = int(vx - w / 2), int(vy - h / 2)
        ix = int(round((x - ox) / max(self._scale, 1e-6)))
        iy = int(round((y - oy) / max(self._scale, 1e-6)))
        return ix, iy

    def set_crosshair(self, enabled: bool) -> None:
        self._cross_enabled = bool(enabled)
        if not enabled:
            self._viewport.coords(self._cross_h, 0, 0, 0, 0)
            self._viewport.coords(self._cross_v, 0, 0, 0, 0)

    # ──────────────────────────── INTERNAL: RENDER ───────────────────────

    def _fit_to_window(self) -> None:
        if not self._img_orig:
            return
        vw, vh = max(1, self._viewport.winfo_width()), max(1, self._viewport.winfo_height())
        iw, ih = self._img_orig.size
        self._scale = min(vw / iw, vh / ih, 1.0)
        self._render_scaled()
        self._resize_and_center()

    def _render_scaled(self) -> None:
        if not (self._img_orig and Image and ImageTk):
            return
        iw, ih = self._img_orig.size
        w = max(1, int(round(iw * self._scale)))
        h = max(1, int(round(ih * self._scale)))

        try:
            img_scaled = self._img_orig if (w, h) == self._img_orig.size else self._img_orig.resize(
                (w, h), getattr(Image, "LANCZOS", 1)
            )
            self._scaled_ref = img_scaled
            if self._image_widget is not None:
                # tryb ImageCanvas
                if hasattr(self._image_widget, "set_image") and callable(self._image_widget.set_image):
                    self._image_widget.set_image(img_scaled)
                elif hasattr(self._image_widget, "display") and callable(self._image_widget.display):
                    self._image_widget.display(img_scaled)  # type: ignore[attr-defined]
                # wymuś rozmiar okna widgetu
                self._viewport.itemconfigure(self._image_win, width=w, height=h)
            else:
                # fallback: rysuj na Canvasie
                self._tk_img = ImageTk.PhotoImage(img_scaled)
                if getattr(self, "_image_item", None) is None:
                    self._image_item = self._viewport.create_image(0, 0, image=self._tk_img, anchor="nw")
                else:
                    self._viewport.itemconfigure(self._image_item, image=self._tk_img)
        except Exception:
            pass

    def _resize_and_center(self) -> None:
        if not self._img_orig:
            return
        iw, ih = self._img_orig.size
        w, h = int(round(iw * self._scale)), int(round(ih * self._scale))
        vw, vh = max(1, self._viewport.winfo_width()), max(1, self._viewport.winfo_height())
        if self._image_win:
            self._viewport.itemconfigure(self._image_win, width=w, height=h)
            self._viewport.coords(self._image_win, vw // 2, vh // 2)
        elif getattr(self, "_image_item", None) is not None:
            # fallback na NW — wycentruj przez offset
            ox, oy = max(0, (vw - w) // 2), max(0, (vh - h) // 2)
            self._viewport.coords(self._image_item, ox, oy)

    # ──────────────────────────── INTERNAL: RULERS ───────────────────────

    def _redraw_rulers(self, cursor: Optional[Tuple[int, int]] = None) -> None:
        for c in (self._ruler_top, self._ruler_left):
            c.delete("all")
        if not self._img_orig:
            return

        vw, vh = self._viewport.winfo_width(), self._viewport.winfo_height()
        iw, ih = self._img_orig.size
        sx = sy = self._scale

        # tła
        self._ruler_top.create_rectangle(0, 0, vw, self.RULER_TOP_H, fill=self.BG_RULER, outline="")
        self._ruler_left.create_rectangle(0, 0, self.RULER_LEFT_W, vh, fill=self.BG_RULER, outline="")

        def _step(size: int) -> int:
            if size <= 0:
                return 1
            raw = size / 10.0
            mag = 10 ** int(math.floor(math.log10(raw)))
            for base in (1, 2, 5, 10):
                step = int(base * mag)
                if raw <= step:
                    return max(1, step)
            return max(1, int(raw))

        # X
        step_x = _step(iw)
        for ix in range(0, iw + 1, step_x):
            x = int(ix * sx)
            if x > vw:
                break
            major = (ix % (step_x * 5) == 0)
            y1 = 18 if major else 14
            self._ruler_top.create_line(x, 0, x, y1, fill=self.FG_RULER_MINOR)
            if major:
                self._ruler_top.create_text(x + 3, self.RULER_TOP_H - 12, anchor="nw",
                                            text=str(ix), fill=self.FG_RULER, font=("", 9))

        # Y
        step_y = _step(ih)
        for iy in range(0, ih + 1, step_y):
            y = int(iy * sy)
            if y > vh:
                break
            major = (iy % (step_y * 5) == 0)
            x0 = self.RULER_LEFT_W - (18 if major else 14)
            self._ruler_left.create_line(x0, y, self.RULER_LEFT_W, y, fill=self.FG_RULER_MINOR)
            if major:
                self._ruler_left.create_text(2, y + 2, anchor="nw",
                                             text=str(iy), fill=self.FG_RULER, font=("", 9))

        # kursor
        if cursor and self._cross_enabled:
            cx, cy = cursor
            self._ruler_top.create_line(cx, 0, cx, self.RULER_TOP_H, fill=self.FG_CROSS)
            self._ruler_left.create_line(0, cy, self.RULER_LEFT_W, cy, fill=self.FG_CROSS)

    # ──────────────────────────── EVENT HANDLERS ─────────────────────────

    def _set_tool(self, name: str) -> None:
        if name in self.TOOLS:
            self._tool = name
            try:
                self.tool_var.set(name)
            except Exception:
                pass

    def _on_wheel(self, ev: tk.Event) -> None:  # type: ignore[override]
        if not self._enabled or self._tool != "zoom":
            return
        delta = int(getattr(ev, "delta", 0))
        if getattr(ev, "num", None) in (4, 5):  # X11
            delta = 120 if ev.num == 4 else -120
        factor = 1.1 if delta > 0 else 0.9
        self.set_zoom(self._scale * factor)

    def _on_btn1_down(self, ev: tk.Event) -> None:  # type: ignore[override]
        if not self._enabled:
            return
        if self._tool == "pan":
            self._pan_origin = (int(ev.x), int(ev.y))
            self._viewport.configure(cursor="fleur")

    def _on_btn1_drag(self, ev: tk.Event) -> None:  # type: ignore[override]
        if not self._enabled or self._tool != "pan" or not self._pan_origin:
            return
        x0, y0 = self._pan_origin
        dx, dy = int(ev.x) - x0, int(ev.y) - y0
        vx, vy = self._viewport.coords(self._image_win) if self._image_win else (0, 0)
        self._viewport.coords(self._image_win, vx + dx, vy + dy)
        self._pan_origin = (int(ev.x), int(ev.y))

    def _on_btn1_up(self, _ev=None) -> None:
        self._pan_origin = None
        self._viewport.configure(cursor="")

    def _on_motion(self, ev: tk.Event) -> None:  # type: ignore[override]
        x, y = int(ev.x), int(ev.y)
        if self._cross_enabled:
            vw, vh = self._viewport.winfo_width(), self._viewport.winfo_height()
            self._viewport.coords(self._cross_h, 0, y, vw, y)
            self._viewport.coords(self._cross_v, x, 0, x, vh)
        self._redraw_rulers(cursor=(x, y))
        # publikacja pozycji kursora (ekranowej i obrazowej)
        if self.bus and hasattr(self.bus, "publish"):
            ix, iy = self.screen_to_image(x, y)
            try:
                self.bus.publish("ui.cursor.pos", {"x": ix, "y": iy})
            except Exception:
                pass

    def _on_leave(self, _ev=None) -> None:
        # schowaj crosshair
        self._viewport.coords(self._cross_h, 0, 0, 0, 0)
        self._viewport.coords(self._cross_v, 0, 0, 0, 0)
        self._redraw_rulers()
