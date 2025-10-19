# glitchlab/app/widgets/image_layers.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Any

import numpy as np
from PIL import Image, ImageTk

# Typy pomocnicze
ToScreen = Callable[[int, int], Tuple[int, int]]     # (ix,iy)->(sx,sy)
GetZoomPan = Callable[[], Tuple[float, Tuple[int, int]]]  # -> (zoom, (panx,pany))
GetBaseSize = Callable[[], Tuple[int, int]]          # -> (W,H) obrazu bazowego


# ──────────────────────────────────────────────────────────────────────────────
# Baza warstwy
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LayerBase:
    visible: bool = True
    alpha: float = 1.0   # 0..1

    def set_visible(self, v: bool) -> None:
        self.visible = bool(v)

    def set_alpha(self, a: float) -> None:
        self.alpha = float(max(0.0, min(1.0, a)))

    def draw(self, tk_canvas: Any, to_screen: ToScreen, get_zoom_pan: GetZoomPan, get_base_size: GetBaseSize) -> None:
        """Rysuje warstwę na tk.Canvas. Implementacje powinny respektować visible/alpha."""
        raise NotImplementedError


# ──────────────────────────────────────────────────────────────────────────────
# Warstwa obrazu RGB (bitmapa) z prostym cache PhotoImage per skala
# ──────────────────────────────────────────────────────────────────────────────

class ImageLayer(LayerBase):
    """
    Warstwa obrazu (np.uint8 RGB lub PIL.Image).
    Uwaga: przechowujemy oryginał; na potrzeby renderu tworzymy przeskalowany cache PhotoImage.
    """
    def __init__(self, image: np.ndarray | Image.Image | None = None) -> None:
        super().__init__(visible=True, alpha=1.0)
        self._src_img: Optional[Image.Image] = None   # PIL RGB
        self._tk_img: Optional[ImageTk.PhotoImage] = None
        self._tk_key: Optional[Tuple[int, int]] = None  # (scaled_w, scaled_h)
        if image is not None:
            self.set_image(image)

    # API
    def set_image(self, image: np.ndarray | Image.Image) -> None:
        if isinstance(image, Image.Image):
            self._src_img = image.convert("RGB")
        else:
            # np.uint8 RGB
            assert image.ndim == 3 and image.shape[2] == 3, "ImageLayer expects (H,W,3) uint8"
            self._src_img = Image.fromarray(image.astype(np.uint8), mode="RGB")
        # Inwaliduj cache
        self._tk_img = None
        self._tk_key = None

    def _get_scaled_photo(self, zoom: float) -> Optional[ImageTk.PhotoImage]:
        if self._src_img is None:
            return None
        # docelowy rozmiar po skali
        w, h = self._src_img.size
        sw = max(1, int(round(w * zoom)))
        sh = max(1, int(round(h * zoom)))
        key = (sw, sh)
        if self._tk_img is not None and self._tk_key == key:
            return self._tk_img
        # rekalkulacja
        if zoom == 1.0:
            im = self._src_img
        else:
            # szybki, dobry wizualnie resampling do podglądu
            im = self._src_img.resize(key, resample=Image.BILINEAR if zoom < 1.0 else Image.BICUBIC)
        self._tk_img = ImageTk.PhotoImage(im)
        self._tk_key = key
        return self._tk_img

    def draw(self, tk_canvas: Any, to_screen: ToScreen, get_zoom_pan: GetZoomPan, get_base_size: GetBaseSize) -> None:
        if not self.visible or self.alpha <= 0.0:
            return
        photo = self._get_scaled_photo(get_zoom_pan()[0])
        if photo is None:
            return
        # Umieszczamy obraz w (0,0) przestrzeni obrazu → na ekran wg pan/zoom.
        sx, sy = to_screen(0, 0)
        # Tkinter: anchor="nw" by lewy górny róg był w sx,sy
        tk_canvas.create_image(sx, sy, image=photo, anchor="nw")


# ──────────────────────────────────────────────────────────────────────────────
# Warstwa maski (półprzezroczysta nakładka)
# ──────────────────────────────────────────────────────────────────────────────

class MaskLayer(LayerBase):
    """
    Render półprzezroczystej maski (H,W) nad obrazem.
    - get_mask: callable -> ndarray (H,W), dtype uint8 {0,255} lub float [0,1]
    - color: kolor maski (R,G,B), np. (255, 0, 0)
    - alpha: globalna przezroczystość warstwy (0..1)
    """
    def __init__(self, get_mask: Callable[[], Optional[np.ndarray]], color: Tuple[int, int, int] = (255, 0, 0), alpha: float = 0.3) -> None:
        super().__init__(visible=True, alpha=alpha)
        self._get_mask = get_mask
        self._color = tuple(int(c) for c in color)
        self._tk_img: Optional[ImageTk.PhotoImage] = None
        self._tk_key: Optional[Tuple[int, int]] = None

    def set_color(self, rgb: Tuple[int, int, int]) -> None:
        self._color = tuple(int(c) for c in rgb)
        self._tk_img = None
        self._tk_key = None

    def _normalize_mask(self, m: np.ndarray) -> np.ndarray:
        m = np.asarray(m)
        if m.ndim == 3:
            m = m[..., 0]
        if m.dtype != np.float32 and m.dtype != np.float64:
            # traktuj jako 0..255
            m = m.astype(np.float32) / 255.0
        return np.clip(m, 0.0, 1.0)

    def _make_overlay_rgba(self, zoom: float) -> Optional[ImageTk.PhotoImage]:
        mask = self._get_mask()
        if mask is None:
            return None
        m = self._normalize_mask(mask)  # f32 [0,1], HxW
        h, w = m.shape[:2]
        # z koloru + alpha global częściowo wypełniona przez maskę
        a = (m * (self.alpha * 255.0)).astype(np.uint8)
        r, g, b = self._color
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[..., 0] = r
        rgba[..., 1] = g
        rgba[..., 2] = b
        rgba[..., 3] = a
        im = Image.fromarray(rgba, mode="RGBA")

        # skala
        if zoom != 1.0:
            sw = max(1, int(round(w * zoom)))
            sh = max(1, int(round(h * zoom)))
            im = im.resize((sw, sh), resample=Image.NEAREST)

        return ImageTk.PhotoImage(im)

    def draw(self, tk_canvas: Any, to_screen: ToScreen, get_zoom_pan: GetZoomPan, get_base_size: GetBaseSize) -> None:
        if not self.visible or self.alpha <= 0.0:
            return
        zoom = get_zoom_pan()[0]
        # cache per rozdzielczość
        key = get_base_size()
        if self._tk_img is None or self._tk_key != (key, zoom):
            self._tk_img = self._make_overlay_rgba(zoom)
            self._tk_key = (key, zoom)
        if self._tk_img is None:
            return
        sx, sy = to_screen(0, 0)
        tk_canvas.create_image(sx, sy, image=self._tk_img, anchor="nw")


# ──────────────────────────────────────────────────────────────────────────────
# Warstwa overlay (ramki/uchwyty/wskaźniki)
# ──────────────────────────────────────────────────────────────────────────────

class OverlayLayer(LayerBase):
    """
    Warstwa do rysowania dynamicznych nakładek przez narzędzia.
    draw_func: callable(tk_canvas, to_screen, get_zoom_pan, get_base_size) -> None
    """
    def __init__(self, draw_func: Callable[[Any, ToScreen, GetZoomPan, GetBaseSize], None]) -> None:
        super().__init__(visible=True, alpha=1.0)
        self._draw_func = draw_func

    def draw(self, tk_canvas: Any, to_screen: ToScreen, get_zoom_pan: GetZoomPan, get_base_size: GetBaseSize) -> None:
        if not self.visible:
            return
        # Uwaga: overlay nie modyfikuje obrazu/maski — tylko prymitywy Canvas
        self._draw_func(tk_canvas, to_screen, get_zoom_pan, get_base_size)
