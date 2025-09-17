# glitchlab/gui/widgets/canvas_container.py
# -*- coding: utf-8 -*-
"""
CanvasContainer – viewer 2-D + toolbox + linijki + crosshair.

Funkcje:
• Przechowuje oryginalny obraz (self._img_orig) i renderuje skalowaną kopię.
• Prawidłowe centrowanie w viewport + linijki zależne od zoomu.
• Toolbox (pan/zoom/ruler/probe/pick) jako radiobuttony z ikonami.
• Crosshair i publikacja pozycji kursora na busie: topic "ui.cursor.pos".
• Obsługa pan (LPM przeciągnij) oraz zoom (scroll) – gdy wybrane narzędzie.
• Integracja z EventBus: reaguje na "ui.tools.select".
• Blokada UI w trakcie pracy (set_enabled(False)) – wygaszanie zdarzeń.

Uwaga: korzysta z opcjonalnych zależności (Pillow). W razie braku – degraduje się do etykiety.
"""

from __future__ import annotations

import math
from pathlib import Path
import tkinter as tk
from tkinter import ttk
from typing import Any, Optional, Tuple, Dict

# ────────── Pillow (opcjonalnie – miniatury, powiększacz) ──────────
try:
    from PIL import Image, ImageTk
except Exception:  # Pillow nieobecny
    Image = ImageTk = None  # type: ignore

# ────────── fallback ImageCanvas (pojedyncza etykieta) ──────────
try:
    from glitchlab.gui.widgets.image_canvas import ImageCanvas  # type: ignore
except Exception:

    class ImageCanvas(ttk.Label):  # type: ignore
        """Awaryjny viewer – zwykły Label z obrazkiem."""
        def set_image(self, pil_img):  # type: ignore
            if Image and ImageTk:
                try:
                    self._photo = ImageTk.PhotoImage(pil_img)  # type: ignore[arg-type]
                    self.configure(image=self._photo, text="")
                    return
                except Exception:  # pragma: no cover
                    pass
            self.configure(text="(viewer unavailable)")


# ══════════════════════════ CanvasContainer ══════════════════════════
class CanvasContainer(ttk.Frame):
    """Viewer + toolbox + rulers + crosshair / pan / zoom / probe."""

    # ---------- stałe UI ----------
    TOOLS = ("pan", "zoom", "ruler", "probe", "pick")

    TOOLBOX_W, RULER_TOP_H, RULER_LEFT_W = 48, 28, 34
    BG_VIEWPORT = "#101010"
    BG_RULER = "#1b1b1b"
    FG_RULER = "#9a9a9a"
    FG_RULER_MINOR = "#5b5b5b"
    FG_CROSS = "#66aaff"
    FG_DISABLED = "#444444"

    _ICON_DIR = Path(__file__).resolve().parents[2] / "resources" / "icons"
    _ICON_FILES = {
        "pan": "icon_pan.png",
        "zoom": "icon_zoom.png",
        "ruler": "icon_ruler.png",
        "probe": "icon_probe.png",
        "pick": "icon_pick.png",
    }

    # ---------- konstruktor ----------
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
        self._tool = self.tool_var.get()
        self._enabled = True
        self._img_orig: Optional["Image.Image"] = None  # oryginał
        self._scale = 1.0
        self._pan_origin: Optional[Tuple[int, int]] = None

        # siatka 3×2
        self.columnconfigure(2, weight=1)
        self.rowconfigure(1, weight=1)

        self._build_toolbox()
        self._build_rulers()
        self._build_viewport()
        self._wire_events()
        self._wire_bus()

    # ---------- budowanie pod-UI ----------
    def _build_toolbox(self) -> None:
        box = ttk.Frame(self, width=self.TOOLBOX_W)
        box.grid(row=1, column=0, sticky="ns")
        box.grid_propagate(False)

        # ikony (opcjonalnie)
        self._icons: Dict[str, tk.PhotoImage] = {}
        if Image and ImageTk:
            for key, fname in self._ICON_FILES.items():
                p = self._ICON_DIR / fname
                if p.exists():
                    try:
                        img = Image.open(p).resize((32, 32), Image.NEAREST)
                        self._icons[key] = ImageTk.PhotoImage(img)  # type: ignore[arg-type]
                    except Exception:
                        pass

        # radiobuttony
        for r, name in enumerate(self.TOOLS):
            ttk.Radiobutton(
                box,
                image=self._icons.get(name, ""),
                text=name.capitalize() if name not in self._icons else "",
                compound="top",
                padding=1,
                variable=self.tool_var,
                value=name,
                command=lambda n=name: self._set_tool(n),
                style="Toolbutton",
            ).grid(row=r, column=0, pady=4, padx=6)

    def _build_rulers(self) -> None:
        # narożne maski
        tk.Canvas(
            self,
            width=self.TOOLBOX_W,
            height=self.RULER_TOP_H,
            bg=self.BG_RULER,
            highlightthickness=0,
        ).grid(row=0, column=0)
        tk.Canvas(
            self,
            width=self.RULER_LEFT_W,
            height=self.RULER_TOP_H,
            bg=self.BG_RULER,
            highlightthickness=0,
        ).grid(row=0, column=1)

        self._ruler_left = tk.Canvas(
            self, width=self.RULER_LEFT_W, bg=self.BG_RULER, highlightthickness=0
        )
        self._ruler_left.grid(row=1, column=1, sticky="ns")

        self._ruler_top = tk.Canvas(
            self, height=self.RULER_TOP_H, bg=self.BG_RULER, highlightthickness=0
        )
        self._ruler_top.grid(row=0, column=2, sticky="ew")

    def _build_viewport(self) -> None:
        self._viewport = tk.Canvas(self, bg=self.BG_VIEWPORT, highlightthickness=0)
        self._viewport.grid(row=1, column=2, sticky="nsew")

        self._viewer = ImageCanvas(self._viewport)
        self._viewer_win = self._viewport.create_window(
            0, 0, window=self._viewer, anchor="center"
        )

        # crosshair
        self._cross_h = self._viewport.create_line(0, 0, 0, 0, fill=self.FG_CROSS)
        self._cross_v = self._viewport.create_line(0, 0, 0, 0, fill=self.FG_CROSS)

        # półprzezroczysta „kurtyna” przy disabled
        self._disabled_overlay = self._viewport.create_rectangle(
            0, 0, 0, 0, fill="", outline=""
        )

    # ---------- publiczne API ----------
    def set_image(self, pil_img: "Image.Image") -> None:
        """Zapisz oryginał i wyrenderuj w aktualnej skali."""
        self._img_orig = pil_img
        self._render_scaled_image()
        self.fit_to_window()
        self._redraw_rulers()

    def fit_to_window(self) -> None:
        if not self._img_orig:
            return
        vw, vh = (
            max(1, self._viewport.winfo_width()),
            max(1, self._viewport.winfo_height()),
        )
        iw, ih = self._img_orig.size
        self._scale = min(vw / iw, vh / ih, 1.0)
        self._render_scaled_image()
        self._resize_and_center()

    def set_enabled(self, flag: bool) -> None:
        """Blokuj/odblokuj interakcję (używane podczas run.progress)."""
        self._enabled = bool(flag)
        # wizualne przykrycie (półprzezroczyste)
        try:
            vw, vh = self._viewport.winfo_width(), self._viewport.winfo_height()
            if not self._enabled:
                self._viewport.coords(self._disabled_overlay, 0, 0, vw, vh)
                self._viewport.itemconfigure(self._disabled_overlay, fill="#00000060", outline="")
                self._viewport.configure(cursor="watch")
            else:
                self._viewport.itemconfigure(self._disabled_overlay, fill="", outline="")
                self._viewport.configure(cursor="")
        except Exception:
            pass

    # ---------- bus ----------
    def _wire_bus(self) -> None:
        if not (self.bus and hasattr(self.bus, "subscribe")):
            return

        # zmiana narzędzia z menu „Tools”
        def _on_tool(_t: str, d: Dict[str, Any]) -> None:
            name = (d or {}).get("name")
            if isinstance(name, str):
                self._set_tool(name)

        try:
            self.bus.subscribe("ui.tools.select", _on_tool)
        except Exception:
            pass

    # ---------- eventy ----------
    def _wire_events(self) -> None:
        vp = self._viewport
        vp.bind(
            "<Configure>",
            lambda _e: (self._resize_and_center(), self._redraw_rulers()),
        )
        vp.bind("<Motion>", self._on_motion)
        vp.bind("<Leave>", self._on_leave)

        # wheel (zoom)
        vp.bind("<MouseWheel>", self._on_wheel)   # Win/macOS
        vp.bind("<Button-4>", self._on_wheel)     # X11 up
        vp.bind("<Button-5>", self._on_wheel)     # X11 down

        # panning
        vp.bind("<ButtonPress-1>", self._on_btn1_down)
        vp.bind("<B1-Motion>", self._on_btn1_drag)
        vp.bind("<ButtonRelease-1>", self._on_btn1_up)

    # ---------- handlers ----------
    def _on_motion(self, ev: tk.Event) -> None:  # type: ignore[override]
        if not self._enabled:
            return
        x, y = int(ev.x), int(ev.y)
        # crosshair
        vw, vh = self._viewport.winfo_width(), self._viewport.winfo_height()
        self._viewport.coords(self._cross_h, 0, y, vw, y)
        self._viewport.coords(self._cross_v, x, 0, x, vh)
        # rulers highlight
        self._redraw_rulers(cursor=(x, y))
        # publish cursor pos
        if self.bus and hasattr(self.bus, "publish"):
            self.bus.publish("ui.cursor.pos", {"x": x, "y": y})

    def _on_leave(self, _ev=None):
        if not self._enabled:
            return
        self._viewport.coords(self._cross_h, 0, 0, 0, 0)
        self._viewport.coords(self._cross_v, 0, 0, 0, 0)
        self._redraw_rulers()

    def _on_wheel(self, ev: tk.Event) -> None:  # type: ignore[override]
        if not self._enabled or self._tool != "zoom":
            return
        delta = int(getattr(ev, "delta", 0))
        if getattr(ev, "num", None) in (4, 5):  # X11
            delta = 120 if ev.num == 4 else -120
        factor = 1.1 if delta > 0 else 0.9
        self._scale = min(8.0, max(1e-3, self._scale * factor))
        self._render_scaled_image()
        self._resize_and_center()
        self._redraw_rulers()

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
        vx, vy = self._viewport.coords(self._viewer_win)
        self._viewport.coords(self._viewer_win, vx + dx, vy + dy)
        self._pan_origin = (int(ev.x), int(ev.y))

    def _on_btn1_up(self, _ev=None):
        if not self._enabled:
            return
        self._pan_origin = None
        self._viewport.configure(cursor="")

    # ---------- pomocnicze ----------
    def _set_tool(self, name: str) -> None:
        if name in self.TOOLS:
            self._tool = name
            self.tool_var.set(name)

    def _render_scaled_image(self) -> None:
        """Tworzy kopię oryginału w aktualnej skali i przekazuje do ImageCanvas."""
        if not (self._img_orig and Image and ImageTk):
            return
        iw, ih = self._img_orig.size
        w = max(1, int(iw * self._scale))
        h = max(1, int(ih * self._scale))
        try:
            img_scaled = (
                self._img_orig
                if (w, h) == self._img_orig.size
                else self._img_orig.resize((w, h), Image.LANCZOS)
            )
            self._viewer.set_image(img_scaled)
            # zachowaj refę, żeby GC nie zjadł bitmapy
            self._viewer._photo_ref = getattr(self._viewer, "_photo", None)
            self._viewport.itemconfigure(self._viewer_win, width=w, height=h)
        except Exception:  # pragma: no cover
            pass

    def _resize_and_center(self) -> None:
        if not self._img_orig:
            return
        iw, ih = self._img_orig.size
        ww, wh = int(iw * self._scale), int(ih * self._scale)
        vw, vh = self._viewport.winfo_width(), self._viewport.winfo_height()
        self._viewport.itemconfigure(self._viewer_win, width=ww, height=wh)
        self._viewport.coords(self._viewer_win, vw // 2, vh // 2)

    # ---------- rulers ----------
    def _redraw_rulers(self, cursor: Optional[Tuple[int, int]] = None) -> None:
        for c in (self._ruler_top, self._ruler_left):
            c.delete("all")
        if not self._img_orig:
            return

        vw, vh = self._viewport.winfo_width(), self._viewport.winfo_height()
        iw, ih = self._img_orig.size
        sx = sy = self._scale

        # tło
        self._ruler_top.create_rectangle(0, 0, vw, self.RULER_TOP_H, fill=self.BG_RULER, outline="")
        self._ruler_left.create_rectangle(0, 0, self.RULER_LEFT_W, vh, fill=self.BG_RULER, outline="")

        # helper: krok
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

        # skala X
        step_x = _step(iw)
        for ix in range(0, iw + 1, step_x):
            x = int(ix * sx)
            if x > vw:
                break
            major = ix % (step_x * 5) == 0
            y1 = 18 if major else 14
            self._ruler_top.create_line(x, 0, x, y1, fill=self.FG_RULER_MINOR)
            if major:
                self._ruler_top.create_text(
                    x + 3, self.RULER_TOP_H - 12, anchor="nw", text=str(ix), fill=self.FG_RULER, font=("", 9)
                )

        # skala Y
        step_y = _step(ih)
        for iy in range(0, ih + 1, step_y):
            y = int(iy * sy)
            if y > vh:
                break
            major = iy % (step_y * 5) == 0
            x0 = self.RULER_LEFT_W - (18 if major else 14)
            self._ruler_left.create_line(x0, y, self.RULER_LEFT_W, y, fill=self.FG_RULER_MINOR)
            if major:
                self._ruler_left.create_text(2, y + 2, anchor="nw", text=str(iy), fill=self.FG_RULER, font=("", 9))

        # podświetlenie kursora
        if cursor:
            cx, cy = cursor
            self._ruler_top.create_line(cx, 0, cx, self.RULER_TOP_H, fill=self.FG_CROSS)
            self._ruler_left.create_line(0, cy, self.RULER_LEFT_W, cy, fill=self.FG_CROSS)
