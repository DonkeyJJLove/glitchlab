# glitchlab/gui/widgets/image_canvas.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from typing import Callable, Optional, Tuple, Any

import tkinter as tk
from tkinter import ttk

try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore

try:
    from PIL import Image, ImageTk  # type: ignore
except Exception:
    Image = ImageTk = None  # type: ignore

__all__ = ["ImageCanvas", "OverviewMini", "MaskPreview"]


# ═════════════════════════════════ ImageCanvas ═════════════════════════════════
class ImageCanvas(ttk.Frame):
    """
    Lekki viewer 2D z płynnym zoom/pan i krzyżem celowniczym.
    Transformacja:  screen_xy = img_xy * zoom + offset

    Public API:
      - set_image(img_u8 | PIL.Image)
      - get_zoom() -> float
      - set_zoom(z: float)
      - zoom_to(z: float) (alias)
      - fit()
      - center()
      - set_crosshair(flag: bool)
      - get_viewport_pixels() -> (x0,y0,x1,y1)  (w pikselach obrazu)
      - set_view_center(x_img: float, y_img: float)
      - on_view_changed: Optional[Callable[[], None]]
      - set_enabled(flag: bool)  # no-op w tym widoku (zgodność z resztą GUI)
      - screen_to_image(x: int, y: int) -> (ix, iy)  (pomocnicze)
      - get_origin() -> (ox, oy)  (offset w przestrzeni canvas)

    Uwaga:
    - Dbamy o to, by „crosshair” był renderowany **nad** obrazem.
    - Skalowanie jakościowe: LANCZOS dla powiększeń/zmniejszeń.
    """

    BG = "#2b2b2b"

    def __init__(self, parent: tk.Misc, **kwargs: Any) -> None:
        super().__init__(parent, **kwargs)

        self.cv = tk.Canvas(self, bg=self.BG, highlightthickness=0, bd=0)
        self.cv.pack(fill="both", expand=True)

        # Stan obrazu/renderu
        self._img_np: Optional["np.ndarray"] = None
        self._img_pil: Optional["Image.Image"] = None
        self._photo: Optional["ImageTk.PhotoImage"] = None
        self._img_item: Optional[int] = None

        # Transformacja
        self._zoom: float = 1.0
        self._min_zoom: float = 0.05
        self._max_zoom: float = 16.0
        self._offset: Tuple[float, float] = (0.0, 0.0)  # lewy-górny obrazka w canvas

        # Interakcje
        self._is_panning: bool = False
        self._pan_anchor: Tuple[int, int] = (0, 0)

        # Overlays
        self._crosshair: bool = False
        self._cross_ids: Tuple[Optional[int], Optional[int]] = (None, None)

        # Callback na zmianę widoku (np. OverviewMini)
        self.on_view_changed: Optional[Callable[[], None]] = None

        # Zdarzenia
        self.cv.bind("<Configure>", lambda _e: self._redraw())
        self.cv.bind("<Button-1>", self._on_btn1)
        self.cv.bind("<B1-Motion>", self._on_drag)
        self.cv.bind("<ButtonRelease-1>", self._on_release)

        # Scroll (platformy różnie raportują wheel)
        self.cv.bind("<MouseWheel>", self._on_wheel)   # Windows/macOS
        self.cv.bind("<Button-4>", self._on_wheel_up)  # X11 up
        self.cv.bind("<Button-5>", self._on_wheel_down)  # X11 down

    # ───────────────────────────── Public API ─────────────────────────────

    def set_enabled(self, _flag: bool) -> None:
        """Zgodność z resztą GUI – tutaj interfejs zawsze aktywny (no-op)."""
        # Można dodać przyciemnienie/kurtynę, jeśli kiedyś będzie potrzeba.
        return

    def set_image(self, img: Any) -> None:
        """
        Przyjmuje:
          - ndarray (HxW, HxWx1, HxWx3, HxWx4) – dowolny dtype (normalizujemy do uint8)
          - PIL.Image
        """
        # Normalizacja do PIL RGB + ndarray uint8
        pil: Optional["Image.Image"] = None
        arr_u8: Optional["np.ndarray"] = None

        try:
            if Image is not None and hasattr(Image, "Image") and isinstance(img, Image.Image):  # PIL.Image
                pil = img.convert("RGB")
                if np is not None:
                    arr_u8 = np.array(pil, dtype=np.uint8)
            elif np is not None and isinstance(img, np.ndarray):
                a = img
                if a.ndim == 2:  # gray → RGB
                    a = (a.astype("float32"))
                    if a.max() <= 1.001:
                        a = a * 255.0
                    a = a.clip(0, 255).astype("uint8")
                    a = np.stack([a, a, a], axis=-1)
                elif a.ndim == 3 and a.shape[-1] in (1, 3, 4):
                    a = a.astype("float32")
                    if a.max() <= 1.001:
                        a = a * 255.0
                    a = a.clip(0, 255).astype("uint8")
                    if a.shape[-1] == 1:
                        a = np.repeat(a, 3, axis=-1)
                    if a.shape[-1] == 4:
                        # premultiply z czarnym tłem deterministycznie
                        rgb = a[..., :3].astype("float32")
                        alpha = (a[..., 3:4].astype("float32") / 255.0)
                        a = (rgb * alpha).clip(0, 255).astype("uint8")
                else:
                    # nieobsługiwany kształt
                    a = None  # type: ignore
                if a is not None and Image is not None:
                    pil = Image.fromarray(a, "RGB")
                    arr_u8 = a
            else:
                pil = None
        except Exception:
            pil = None
            arr_u8 = None

        # Ustaw stan
        self._img_pil = pil
        self._img_np = arr_u8
        self._ensure_photo()

        # Dopasuj do okna (z zachowaniem zdrowego marginesu)
        if self._img_pil is None:
            self._img_item = None
            self._redraw()
            return

        # reset zoom tylko przy pierwszym obrazie w sesji
        if self._img_item is None:
            self._zoom = 1.0
        self.fit()
        self._emit_view_changed()

    def get_zoom(self) -> float:
        return float(self._zoom)

    def set_zoom(self, z: float) -> None:
        z = float(z)
        z = max(self._min_zoom, min(self._max_zoom, z))
        if abs(z - self._zoom) < 1e-9:
            return
        # Zoom wokół środka canvasa
        cx = self.cv.winfo_width() / 2.0
        cy = self.cv.winfo_height() / 2.0
        self._zoom_around_point(z / self._zoom, (cx, cy))
        self._emit_view_changed()

    def zoom_to(self, z: float) -> None:
        self.set_zoom(z)

    def fit(self) -> None:
        """Dopasowanie całego obrazu do rozmiaru okna (z niewielkim marginesem)."""
        if self._img_pil is None:
            return
        cw = max(1, self.cv.winfo_width())
        ch = max(1, self.cv.winfo_height())
        iw, ih = self._img_pil.size
        if iw <= 0 or ih <= 0:
            return
        scale = min(cw / iw, ch / ih) * 0.985  # delikatny margines
        scale = max(self._min_zoom, min(self._max_zoom, scale))
        self._zoom = scale
        self.center()
        self._emit_view_changed()

    def center(self) -> None:
        """Wyśrodkuj obraz w oknie (dla aktualnego zoomu)."""
        if self._img_pil is None:
            return
        cw = self.cv.winfo_width()
        ch = self.cv.winfo_height()
        iw, ih = self._img_pil.size
        self._offset = ((cw - iw * self._zoom) / 2.0, (ch - ih * self._zoom) / 2.0)
        self._redraw()

    def set_view_center(self, x_img: float, y_img: float) -> None:
        """Ustaw środek widoku na wskazany punkt obrazu (w pikselach obrazu)."""
        if self._img_pil is None:
            return
        cw = self.cv.winfo_width()
        ch = self.cv.winfo_height()
        iw, ih = self._img_pil.size
        x_img = max(0.0, min(float(iw), float(x_img)))
        y_img = max(0.0, min(float(ih), float(y_img)))
        self._offset = (cw / 2.0 - x_img * self._zoom, ch / 2.0 - y_img * self._zoom)
        self._redraw()
        self._emit_view_changed()

    def get_viewport_pixels(self) -> Tuple[int, int, int, int]:
        """Aktualny prostokąt widzialnej części (x0,y0,x1,y1) w pikselach obrazu."""
        if self._img_pil is None or self._zoom <= 0:
            return (0, 0, 0, 0)
        cw = self.cv.winfo_width()
        ch = self.cv.winfo_height()
        iw, ih = self._img_pil.size
        ox, oy = self._offset

        x0 = int(max(0, math.floor((0 - ox) / self._zoom)))
        y0 = int(max(0, math.floor((0 - oy) / self._zoom)))
        x1 = int(min(iw, math.ceil((cw - ox) / self._zoom)))
        y1 = int(min(ih, math.ceil((ch - oy) / self._zoom)))

        x0 = max(0, min(iw, x0))
        y0 = max(0, min(ih, y0))
        x1 = max(0, min(iw, x1))
        y1 = max(0, min(ih, y1))
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        return (x0, y0, x1, y1)

    def screen_to_image(self, x: int, y: int) -> Tuple[int, int]:
        """Konwersja współrzędnych ekranu (canvas) na piksele obrazu."""
        if self._img_pil is None or self._zoom <= 0:
            return (0, 0)
        ox, oy = self._offset
        ix = int((x - ox) / self._zoom)
        iy = int((y - oy) / self._zoom)
        iw, ih = self._img_pil.size
        ix = max(0, min(iw - 1, ix))
        iy = max(0, min(ih - 1, iy))
        return ix, iy

    def get_origin(self) -> Tuple[float, float]:
        """Zwraca bieżący offset (ox, oy) w przestrzeni canvas."""
        return self._offset

    def set_crosshair(self, flag: bool) -> None:
        self._crosshair = bool(flag)
        self._redraw()

    # ───────────────────────────── Events ─────────────────────────────

    def _on_btn1(self, e: tk.Event) -> None:  # type: ignore[override]
        self._is_panning = True
        self._pan_anchor = (int(e.x), int(e.y))

    def _on_drag(self, e: tk.Event) -> None:  # type: ignore[override]
        if not self._is_panning:
            return
        ax, ay = self._pan_anchor
        dx, dy = int(e.x) - ax, int(e.y) - ay
        ox, oy = self._offset
        self._offset = (ox + dx, oy + dy)
        self._pan_anchor = (int(e.x), int(e.y))
        self._redraw()
        self._emit_view_changed()

    def _on_release(self, _e: tk.Event) -> None:  # type: ignore[override]
        self._is_panning = False

    def _on_wheel(self, e: tk.Event) -> None:  # type: ignore[override]
        # Windows/macOS: delta wielokrotność 120; direction = sign(delta)
        d = 1 if int(getattr(e, "delta", 0)) > 0 else -1
        self._zoom_around_point(1.1 if d > 0 else (1 / 1.1), (e.x, e.y))
        self._emit_view_changed()

    def _on_wheel_up(self, e: tk.Event) -> None:  # X11
        self._zoom_around_point(1.1, (e.x, e.y))
        self._emit_view_changed()

    def _on_wheel_down(self, e: tk.Event) -> None:  # X11
        self._zoom_around_point(1 / 1.1, (e.x, e.y))
        self._emit_view_changed()

    # ───────────────────────────── Internals ─────────────────────────────

    def _zoom_around_point(self, factor: float, pivot_xy: Tuple[float, float]) -> None:
        if self._img_pil is None:
            return
        new_zoom = max(self._min_zoom, min(self._max_zoom, self._zoom * float(factor)))
        factor = new_zoom / (self._zoom if self._zoom != 0 else 1.0)
        px, py = pivot_xy
        ox, oy = self._offset
        # Utrzymaj punkt (px,py) na ekranie w tym samym miejscu
        self._offset = (px - (px - ox) * factor, py - (py - oy) * factor)
        self._zoom = new_zoom
        self._redraw()

    def _ensure_photo(self) -> None:
        if self._img_pil is None or ImageTk is None:
            self._photo = None
            return
        try:
            self._photo = ImageTk.PhotoImage(self._img_pil)
            # keep ref optional (czasem inne widgety podmieniają atrybuty)
            self._photo_ref = self._photo
        except Exception:
            self._photo = None

    def _redraw(self) -> None:
        """Rysuje obraz oraz overlay’e (crosshair) w poprawnej kolejności."""
        self.cv.delete("all")
        if self._img_pil is None or Image is None or ImageTk is None:
            return

        iw, ih = self._img_pil.size
        disp_w = max(1, int(round(iw * self._zoom)))
        disp_h = max(1, int(round(ih * self._zoom)))

        # Render w dobrej jakości: LANCZOS dla wszystkich zmian skali
        try:
            pil = self._img_pil if (disp_w == iw and disp_h == ih) else self._img_pil.resize((disp_w, disp_h), Image.LANCZOS)
        except Exception:
            pil = self._img_pil  # awaryjnie bez resize

        try:
            self._photo = ImageTk.PhotoImage(pil)
            self._photo_ref = self._photo  # trzymaj referencję
        except Exception:
            self._photo = None

        if self._photo is None:
            return

        ox, oy = self._offset
        self._img_item = self.cv.create_image(ox, oy, anchor="nw", image=self._photo)

        # Crosshair na wierzchu
        if self._crosshair:
            cw = self.cv.winfo_width()
            ch = self.cv.winfo_height()
            cx = cw // 2
            cy = ch // 2
            # półprzezroczystość imitujemy przerywaniem/stipple
            self.cv.create_line(0, cy, cw, cy, fill="#6fa2ff")
            self.cv.create_line(cx, 0, cx, ch, fill="#6fa2ff")

    def _emit_view_changed(self) -> None:
        cb = self.on_view_changed
        if callable(cb):
            try:
                cb()
            except Exception:
                pass


# ═════════════════════════════════ OverviewMini ═══════════════════════════════
class OverviewMini(ttk.Frame):
    """
    Minimap całego obrazu + prostokąt aktualnego viewportu.
    Klik/drag w minimapie przewija główny obraz.
    """
    def __init__(self, parent: tk.Misc, canvas: ImageCanvas, width: int = 220, height: int = 160) -> None:
        super().__init__(parent)
        self.canvas_ref = canvas
        self.width = width
        self.height = height

        self.cv = tk.Canvas(self, width=width, height=height, bg="#1e1e1e",
                            highlightthickness=1, highlightbackground="#444")
        self.cv.pack(fill="both", expand=False)

        self._thumb: Optional["Image.Image"] = None
        self._photo: Optional["ImageTk.PhotoImage"] = None

        # Powiązania
        self.canvas_ref.on_view_changed = self._refresh
        self.cv.bind("<Configure>", lambda _e: self._rebuild_thumb())
        self.cv.bind("<Button-1>", self._on_click)
        self.cv.bind("<B1-Motion>", self._on_click)

    def _rebuild_thumb(self) -> None:
        img = getattr(self.canvas_ref, "_img_pil", None)
        if img is None or Image is None or ImageTk is None:
            self.cv.delete("all")
            self._thumb = None
            self._photo = None
            return
        W = max(1, self.cv.winfo_width())
        H = max(1, self.cv.winfo_height())
        iw, ih = img.size
        scale = min(W / iw, H / ih)
        tw = max(1, int(iw * scale))
        th = max(1, int(ih * scale))
        try:
            self._thumb = img.resize((tw, th), Image.BILINEAR)
            self._photo = ImageTk.PhotoImage(self._thumb)
        except Exception:
            self._thumb = None
            self._photo = None
        self._refresh()

    def _refresh(self) -> None:
        self.cv.delete("all")
        if self._photo is None:
            self._rebuild_thumb()
            if self._photo is None:
                return
        W = self.cv.winfo_width()
        H = self.cv.winfo_height()
        tw = self._photo.width()
        th = self._photo.height()
        x0 = (W - tw) // 2
        y0 = (H - th) // 2
        self.cv.create_image(x0, y0, anchor="nw", image=self._photo)

        x_img0, y_img0, x_img1, y_img1 = self.canvas_ref.get_viewport_pixels()
        img = getattr(self.canvas_ref, "_img_pil", None)
        if img is None or x_img1 <= x_img0 or y_img1 <= y_img0:
            return
        iw, ih = img.size
        scale = min(tw / iw, th / ih)
        rx0 = x0 + int(x_img0 * scale)
        ry0 = y0 + int(y_img0 * scale)
        rx1 = x0 + int(x_img1 * scale)
        ry1 = y0 + int(y_img1 * scale)
        self.cv.create_rectangle(rx0, ry0, rx1, ry1, outline="#80ff80")

    def _on_click(self, e: tk.Event) -> None:  # type: ignore[override]
        img = getattr(self.canvas_ref, "_img_pil", None)
        if img is None or self._photo is None:
            return
        W = self.cv.winfo_width()
        H = self.cv.winfo_height()
        tw = self._photo.width()
        th = self._photo.height()
        x0 = (W - tw) // 2
        y0 = (H - th) // 2
        sx = int(e.x) - x0
        sy = int(e.y) - y0
        if sx < 0 or sy < 0 or sx > tw or sy > th:
            return
        iw, ih = img.size
        scale = min(tw / iw, th / ih)
        x_img = sx / scale
        y_img = sy / scale
        self.canvas_ref.set_view_center(x_img, y_img)


# ═════════════════════════════════ MaskPreview ════════════════════════════════
class MaskPreview(ttk.Frame):
    """
    Podgląd maski 2D (ndarray) w małym okienku.
    API:
      - set_array(arr: np.ndarray|None)
    """
    def __init__(self, parent: tk.Misc, width: int = 220, height: int = 120) -> None:
        super().__init__(parent)
        self.cv = tk.Canvas(self, width=width, height=height, bg="#1e1e1e",
                            highlightthickness=1, highlightbackground="#444")
        self.cv.pack(fill="both", expand=False)
        self._photo: Optional["ImageTk.PhotoImage"] = None
        self._arr: Optional["np.ndarray"] = None
        self.cv.bind("<Configure>", lambda _e: self._resize_to_canvas())

    def set_array(self, arr: Optional["np.ndarray"]) -> None:
        if np is None:
            self._arr = None
        else:
            self._arr = None if arr is None else np.array(arr, copy=False)
        self._resize_to_canvas()

    def _resize_to_canvas(self) -> None:
        self.cv.delete("all")
        if self._arr is None or Image is None or ImageTk is None or np is None:
            return
        a = self._arr
        if a.ndim == 3:
            a = a[..., 0]

        # Normalizacja do 0..255
        try:
            mn = float(np.nanmin(a))
            mx = float(np.nanmax(a))
            den = (mx - mn) if (mx - mn) > 1e-12 else 1.0
            u8 = np.clip((a - mn) * (255.0 / den), 0, 255).astype("uint8")
            pil = Image.fromarray(u8, "L").convert("RGB")
        except Exception:
            return

        W = max(1, self.cv.winfo_width())
        H = max(1, self.cv.winfo_height())
        iw, ih = pil.size
        scale = min(W / iw, H / ih)
        tw = max(1, int(iw * scale))
        th = max(1, int(ih * scale))
        try:
            pil = pil.resize((tw, th), Image.NEAREST)
            self._photo = ImageTk.PhotoImage(pil)
        except Exception:
            self._photo = None
            return
        x0 = (W - tw) // 2
        y0 = (H - th) // 2
        self.cv.create_image(x0, y0, anchor="nw", image=self._photo)
