# glitchlab/gui/widgets/canvas_container.py
# -*- coding: utf-8 -*-
"""
CanvasContainer – kontener widoku obrazu:
• Toolbox (ikony narzędzi)
• Rulers (góra/lewo)
• LayerCanvas (dwuwarstwowy viewer: obraz + overlay crosshair)
• Pan/Zoom przez LayerCanvas (overlay zawsze nad obrazem)
• Publikacja pozycji kursora w image-space: "ui.cursor.pos"

Wersja DIAG: intensywne logowanie sekwencji przy pierwszym wczytaniu obrazu
oraz bezpieczne odroczenie zoom_fit na first-paint, by uniknąć rozmiaru 0×0.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import tkinter as tk
from tkinter import ttk

# ────────── Pillow (opcjonalnie – miniatury, powiększacz) ──────────
try:
    from PIL import Image, ImageTk
except Exception:  # Pillow nieobecny
    Image = ImageTk = None  # type: ignore

# ────────── nowy dwuwarstwowy viewer ──────────
try:
    from glitchlab.gui.widgets.layer_canvas import LayerCanvas  # type: ignore
except Exception:
    LayerCanvas = None  # type: ignore

# ────────── fallback mini-viewer jeśli LayerCanvas niedostępny ──────────
if LayerCanvas is None:
    class LayerCanvas(ttk.Label):  # type: ignore
        def __init__(self, master: tk.Misc, *, bus: Optional[Any] = None) -> None:
            super().__init__(master, text="(LayerCanvas unavailable)")
            self._photo = None
            self.bus = bus

        def set_layers(self, images, names=None) -> None:
            if Image and ImageTk and images:
                try:
                    pil = images[0]
                    if not hasattr(pil, "mode"):
                        from PIL import Image as _PIL  # type: ignore
                        pil = _PIL.fromarray(pil, "RGB")
                    pil = pil.convert("RGB")
                    self._photo = ImageTk.PhotoImage(pil)  # type: ignore[arg-type]
                    self.configure(image=self._photo, text="")
                except Exception:
                    pass

        # API zgodne, ale nieaktywne w fallbacku
        def set_composite(self, image): self.set_layers([image], names=["Composite"])
        def add_layer(self, image, name="Layer") -> int: return -1
        def remove_layer(self, index: int) -> None: ...
        def reorder_layer(self, src: int, dst: int) -> None: ...
        def set_active_layer(self, index: int) -> None: ...
        def get_active_layer(self) -> Optional[int]: return 0
        def set_layer_visible(self, index: int, visible: bool) -> None: ...
        def set_layer_offset(self, index: int, dx: int, dy: int) -> None: ...
        def get_zoom(self) -> float: return 1.0
        def set_zoom(self, z: float) -> None: ...
        def zoom_fit(self) -> None: ...
        def center(self) -> None: ...
        def set_view_center(self, x_img: float, y_img: float) -> None: ...
        def screen_to_image(self, sx: int, sy: int) -> Tuple[int, int]: return (sx, sy)
        def set_crosshair(self, flag: bool) -> None: ...


class CanvasContainer(ttk.Frame):
    """Viewer + toolbox + rulers + crosshair / pan / zoom / probe (via LayerCanvas)."""

    # ---------- stałe UI ----------
    TOOLS = ("pan", "zoom", "ruler", "probe", "pick")

    TOOLBOX_W, RULER_TOP_H, RULER_LEFT_W = 56, 28, 34
    BG_VIEWPORT = "#101010"
    BG_RULER = "#1b1b1b"
    FG_RULER = "#9a9a9a"
    FG_RULER_MINOR = "#5b5b5b"
    FG_CROSS = "#66aaff"
    FG_DISABLED = "#444444"

    # UWAGA: w repo katalog to „resources/icons”, nie „icons”.
    _ICON_DIR = Path(__file__).resolve().parents[2] / "resources" / "icons"
    _ICON_FILES = {
        "pan": "icon_pan.png",
        "zoom": "icon_zoom.png",
        "ruler": "icon_ruler.png",
        "probe": "icon_probe.png",
        "pick": "icon_pick.png",
    }

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
        self._img_orig: Optional["Image.Image"] = None
        self._scale = 1.0

        # tag do „kurtyny” na płótnie obrazu (rysowanej na _viewer._cv_img)
        self._disabled_tag = "_disabled_overlay"

        # siatka 3×2
        self.columnconfigure(2, weight=1)
        self.rowconfigure(1, weight=1)

        self._build_toolbox()
        self._build_rulers()
        self._build_viewport()
        self._wire_events()
        self._wire_bus()

    # ---------- helpers: diag ----------
    def _diag(self, msg: str) -> None:
        if self.bus and hasattr(self.bus, "publish"):
            try:
                self.bus.publish("diag.log", {"msg": f"[Canvas] {msg}", "level": "DEBUG"})
                return
            except Exception:
                pass
        print(f"[Canvas][DEBUG] {msg}")

    # ---------- budowanie pod-UI ----------
    def _build_toolbox(self) -> None:
        box = ttk.Frame(self, width=self.TOOLBOX_W)
        box.grid(row=1, column=0, sticky="ns")
        box.grid_propagate(False)

        # Ikony – wczytaj gdy dostępne
        self._icons: Dict[str, tk.PhotoImage] = {}
        if Image and ImageTk:
            for key, fname in self._ICON_FILES.items():
                p = self._ICON_DIR / fname
                if p.exists():
                    try:
                        img = Image.open(p).convert("RGBA").resize((28, 28), Image.LANCZOS)
                        self._icons[key] = ImageTk.PhotoImage(img)  # type: ignore[arg-type]
                    except Exception:
                        pass

        def make_radio(name: str, text_fallback: str) -> ttk.Radiobutton:
            return ttk.Radiobutton(
                box,
                image=self._icons.get(name, ""),
                text=text_fallback if name not in self._icons else "",
                compound="top",
                padding=2,
                variable=self.tool_var,
                value=name,
                command=lambda n=name: self._set_tool(n),
                style="Toolbutton",
            )

        radios = [
            ("pan", "🖐"),
            ("zoom", "🔍"),
            ("ruler", "📏"),
            ("probe", "🎯"),
            ("pick", "🎨"),
        ]
        for r, (name, label) in enumerate(radios):
            make_radio(name, label).grid(row=r, column=0, pady=4, padx=10)

        ttk.Separator(box, orient="horizontal").grid(row=len(radios), column=0, sticky="ew", pady=(8, 2))
        self._cross_var = tk.BooleanVar(value=True)
        chk = ttk.Checkbutton(box, text="Cross", variable=self._cross_var, style="Toolbutton",
                              command=lambda: self._viewer.set_crosshair(bool(self._cross_var.get())))
        chk.grid(row=len(radios)+1, column=0, pady=2)

    def _build_rulers(self) -> None:
        # zaślepki rogów
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
        # Obszar widoku (LayerCanvas)
        wrap = ttk.Frame(self)
        wrap.grid(row=1, column=2, sticky="nsew")
        wrap.rowconfigure(0, weight=1)
        wrap.columnconfigure(0, weight=1)

        self._viewer = LayerCanvas(wrap, bus=self.bus)  # overlay crosshair nad obrazem
        self._viewer.grid(row=0, column=0, sticky="nsew")

        # Kurtynę rysujemy jako elementy na _viewer._cv_img (tag self._disabled_tag).

    # ---------- publiczne API ----------
    def set_image(self, pil_img: "Image.Image") -> None:
        """
        Ustaw obraz kompozytu do podglądu.
        DIAG: logujemy geometrię i odraczamy zoom_fit do chwili, gdy Canvas będzie miał realny rozmiar.
        """
        self._diag("set_image: start")
        self._img_orig = pil_img if (not Image or getattr(pil_img, "mode", "") == "RGB") else pil_img.convert("RGB")

        # Preferencja trybu kompozytu (jeśli LayerCanvas ma taką metodę)
        if hasattr(self._viewer, "set_composite"):
            self._viewer.set_composite(self._img_orig)  # type: ignore[attr-defined]
            self._diag("set_image: viewer.set_composite OK")
        else:
            self._viewer.set_layers([self._img_orig], names=["Background"])
            self._diag("set_image: viewer.set_layers([Background]) OK")

        # >>> KLUCZ: odroczyć fit_to_window do czasu aż geometry manager policzy wymiary
        def _after_layout_fit():
            try:
                cw = self._viewer._cv_img.winfo_width()  # type: ignore[attr-defined]
                ch = self._viewer._cv_img.winfo_height()  # type: ignore[attr-defined]
            except Exception:
                cw = ch = 0
            self._diag(f"after_idle: canvas size = {cw}x{ch}")
            if cw <= 1 or ch <= 1:
                # Jeszcze za wcześnie – spróbuj ponownie
                self.after(16, _after_layout_fit)
                return
            self.fit_to_window()
            self._diag("after_idle: fit_to_window() done")

        # Zamiast natychmiastowego fitu — po idle
        self.after_idle(_after_layout_fit)

    def fit_to_window(self) -> None:
        self._viewer.zoom_fit()
        self._redraw_rulers()

    def set_enabled(self, flag: bool) -> None:
        """Blokuj/odblokuj interakcję (używane podczas run.progress)."""
        self._enabled = bool(flag)
        try:
            cv = getattr(self._viewer, "_cv_img", None)
            if not isinstance(cv, tk.Canvas):
                return
            # usuń poprzednią kurtynę
            try:
                cv.delete(self._disabled_tag)
            except Exception:
                pass
            if not self._enabled:
                w = max(1, cv.winfo_width())
                h = max(1, cv.winfo_height())
                cv.create_rectangle(
                    0, 0, w, h,
                    fill="#000000",
                    stipple="gray25",   # 25% coverage
                    width=0,
                    tags=(self._disabled_tag,),
                )
                cv.configure(cursor="watch")
            else:
                cv.configure(cursor="")
        except Exception:
            pass

    # ---------- bus ----------
    def _wire_bus(self) -> None:
        if not (self.bus and hasattr(self.bus, "subscribe")):
            return

        def _on_tool(_t: str, d: Dict[str, Any]) -> None:
            name = (d or {}).get("name")
            if isinstance(name, str):
                self._set_tool(name)

        def _on_cross(_t: str, d: Dict[str, Any]) -> None:
            flag = bool((d or {}).get("enabled", True))
            self._cross_var.set(flag)
            self._viewer.set_crosshair(flag)

        try:
            self.bus.subscribe("ui.tools.select", _on_tool)
            self.bus.subscribe("ui.image.crosshair.set", _on_cross)
        except Exception:
            pass

    # ---------- eventy ----------
    def _wire_events(self) -> None:
        # redraw rulers przy zmianach rozmiaru kontenera
        self.bind("<Configure>", lambda _e: self._redraw_rulers())
        # ruch kursora z overlay (LayerCanvas sam publikuje ui.cursor.pos)
        self._viewer.bind("<Motion>", lambda e: self._on_overlay_motion(e), add="+")

    # ---------- handlers ----------
    def _on_overlay_motion(self, e: tk.Event) -> None:  # type: ignore[override]
        try:
            x, y = int(e.x), int(e.y)
        except Exception:
            return
        self._redraw_rulers(cursor=(x, y))

    # ---------- pomocnicze ----------
    def _set_tool(self, name: str) -> None:
        if name in self.TOOLS:
            self._tool = name
            self.tool_var.set(name)

    # ---------- rulers ----------
    def _redraw_rulers(self, cursor: Optional[Tuple[int, int]] = None) -> None:
        for c in (self._ruler_top, self._ruler_left):
            c.delete("all")
        base_img = getattr(self._viewer, "_get_base_image", None)
        pil_img = base_img() if callable(base_img) else None
        if pil_img is None:
            vw = self._ruler_top.winfo_width()
            vh = self._ruler_left.winfo_height()
            self._ruler_top.create_rectangle(0, 0, max(1, vw), self.RULER_TOP_H, fill=self.BG_RULER, outline="")
            self._ruler_left.create_rectangle(0, 0, self.RULER_LEFT_W, max(1, vh), fill=self.BG_RULER, outline="")
            return

        vw, vh = self._ruler_top.winfo_width(), self._ruler_left.winfo_height()
        iw, ih = pil_img.size
        sx = sy = float(self._viewer.get_zoom())

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

        if cursor:
            cx, cy = cursor
            self._ruler_top.create_line(cx, 0, cx, self.RULER_TOP_H, fill=self.FG_CROSS)
            self._ruler_left.create_line(0, cy, self.RULER_LEFT_W, cy, fill=self.FG_CROSS)
