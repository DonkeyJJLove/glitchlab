# glitchlab/gui/widgets/layer_canvas.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import tkinter as tk
from tkinter import ttk

try:
    from PIL import Image, ImageTk
except Exception:
    Image = ImageTk = None  # type: ignore

try:
    import numpy as np
except Exception:
    np = None  # type: ignore


@dataclass
class _Layer:
    name: str
    image: "Image.Image"                       # PIL RGB
    photo: Optional["ImageTk.PhotoImage"] = None  # cache do Canvas.create_image
    visible: bool = True
    offset: Tuple[int, int] = (0, 0)


class LayerCanvas(ttk.Frame):
    """
    Viewer jednopłótnowy:
      • jeden Canvas dla obrazów i overlay (krzyż),
      • brak publikacji zdarzeń warstw (źródło prawdy = App/LayerManager),
      • emitujemy tylko:
          - ui.image.crosshair.moved
          - ui.cursor.pos
    """

    BG = "#101010"
    CROSS_FG = "#66aaff"

    def __init__(self, master: tk.Misc, *, bus: Optional[Any] = None) -> None:
        super().__init__(master)
        self.bus = bus

        # pojedynczy Canvas – nic go nie przykrywa
        self._cv = tk.Canvas(self, bg=self.BG, highlightthickness=0, bd=0)
        self._cv.pack(fill="both", expand=True)

        # stan warstw
        self._layers: List[_Layer] = []
        self._active: Optional[int] = None

        # transformacja widoku
        self._zoom: float = 1.0
        self._min_zoom: float = 0.05
        self._max_zoom: float = 16.0
        self._pan: Tuple[float, float] = (0.0, 0.0)

        # interakcje
        self._panning: bool = False
        self._pan_anchor: Tuple[int, int] = (0, 0)

        # crosshair (rysowany na tym samym canvasie, ale wyżej)
        self._cross_enabled: bool = True
        self._cross_h = self._cv.create_line(0, 0, 0, 0, fill=self.CROSS_FG, width=1, tags=("__cross__",))
        self._cross_v = self._cv.create_line(0, 0, 0, 0, fill=self.CROSS_FG, width=1, tags=("__cross__",))

        self._did_first_fit: bool = False

        self._wire_events()

    # ─────────────────────────── public API: kompozyt/warstwy ───────────────────────────

    def set_composite(self, image: Any) -> None:
        """Ustaw kompozyt sceny jako jedną warstwę do renderu."""
        pil = self._to_pil(image)
        if pil is None:
            return
        self.set_layers([pil], names=["Composite"])

    def set_layers(self, images: List[Any], names: Optional[List[str]] = None) -> None:
        """Ustaw komplet warstw (PIL/ndarray) i odrysuj."""
        self._layers.clear()
        names = names or [f"Layer {i+1}" for i in range(len(images))]
        for img, nm in zip(images, names):
            pil = self._to_pil(img)
            if pil is None:
                continue
            self._layers.append(_Layer(name=nm, image=pil))
        self._active = 0 if self._layers else None
        self._rebuild_photos()
        self._render_all()
        if not self._did_first_fit and self._layers:
            self.after_idle(self.zoom_fit)
            self.after_idle(self.center)
            self._did_first_fit = True

    def add_layer(self, image: Any, name: str = "Layer") -> int:
        pil = self._to_pil(image)
        if pil is None:
            return -1
        self._layers.append(_Layer(name=name, image=pil))
        idx = len(self._layers) - 1
        self._active = idx
        self._rebuild_photos([idx])
        self._render_all()
        return idx

    def remove_layer(self, index: int) -> None:
        if index < 0 or index >= len(self._layers):
            return
        del self._layers[index]
        if not self._layers:
            self._active = None
            self._cv.delete("all")
            # odtwórz crosshair placeholdery (mogły zostać skasowane przez delete("all"))
            self._cross_h = self._cv.create_line(0, 0, 0, 0, fill=self.CROSS_FG, width=1, tags=("__cross__",))
            self._cross_v = self._cv.create_line(0, 0, 0, 0, fill=self.CROSS_FG, width=1, tags=("__cross__",))
        else:
            if self._active is None or self._active >= len(self._layers):
                self._active = max(0, len(self._layers) - 1)
        self._render_all()

    def reorder_layer(self, src: int, dst: int) -> None:
        if src == dst or min(src, dst) < 0 or max(src, dst) >= len(self._layers):
            return
        lyr = self._layers.pop(src)
        self._layers.insert(dst, lyr)
        if self._active == src:
            self._active = dst
        elif self._active is not None:
            if src < self._active <= dst:
                self._active -= 1
            elif dst <= self._active < src:
                self._active += 1
        self._render_all()

    def set_active_layer(self, index: int) -> None:
        if 0 <= index < len(self._layers):
            self._active = index

    def get_active_layer(self) -> Optional[int]:
        return self._active

    def set_layer_visible(self, index: int, visible: bool) -> None:
        if 0 <= index < len(self._layers):
            self._layers[index].visible = bool(visible)
            self._render_all()

    def set_layer_offset(self, index: int, dx: int, dy: int) -> None:
        if 0 <= index < len(self._layers):
            self._layers[index].offset = (int(dx), int(dy))
            self._render_all()

    # ─────────────────────────── public API: widok ─────────────────────────────

    def get_zoom(self) -> float:
        return float(self._zoom)

    def set_zoom(self, z: float) -> None:
        z = max(self._min_zoom, min(self._max_zoom, float(z)))
        if abs(z - self._zoom) < 1e-12:
            return
        cw, ch = self._cv.winfo_width(), self._cv.winfo_height()
        self._zoom_around((cw / 2.0, ch / 2.0), z / (self._zoom if self._zoom else 1.0))

    def zoom_fit(self) -> None:
        base = self._get_base_image()
        if base is None:
            return
        iw, ih = base.size
        cw = max(1, self._cv.winfo_width())
        ch = max(1, self._cv.winfo_height())
        scale = min(cw / iw, ch / ih) * 0.985
        scale = max(self._min_zoom, min(self._max_zoom, scale))
        self._zoom = scale
        self.center()

    def center(self) -> None:
        base = self._get_base_image()
        if base is None:
            return
        iw, ih = base.size
        cw, ch = self._cv.winfo_width(), self._cv.winfo_height()
        self._pan = ((cw - iw * self._zoom) / 2.0, (ch - ih * self._zoom) / 2.0)
        self._render_all()

    def set_view_center(self, x_img: float, y_img: float) -> None:
        base = self._get_base_image()
        if base is None:
            return
        iw, ih = base.size
        x_img = max(0.0, min(float(iw), float(x_img)))
        y_img = max(0.0, min(float(ih), float(y_img)))
        cw, ch = self._cv.winfo_width(), self._cv.winfo_height()
        self._pan = (cw / 2.0 - x_img * self._zoom, ch / 2.0 - y_img * self._zoom)
        self._render_all()

    def screen_to_image(self, sx: int, sy: int) -> Tuple[int, int]:
        base = self._get_base_image()
        if base is None or self._zoom <= 0:
            return (0, 0)
        ox, oy = self._pan
        ix = int((sx - ox) / self._zoom)
        iy = int((sy - oy) / self._zoom)
        iw, ih = base.size
        ix = max(0, min(iw - 1, ix))
        iy = max(0, min(ih - 1, iy))
        return ix, iy

    def set_crosshair(self, flag: bool) -> None:
        self._cross_enabled = bool(flag)
        if not flag:
            self._cv.coords(self._cross_h, 0, 0, 0, 0)
            self._cv.coords(self._cross_v, 0, 0, 0, 0)

    # ─────────────────────────── internals: eventy ─────────────────────────────

    def _wire_events(self) -> None:
        # rozmiar
        self.bind("<Configure>", lambda _e: self._render_all())
        # mysz
        self._cv.bind("<Motion>", self._on_motion)
        self._cv.bind("<Leave>", self._on_leave)
        # pan
        self._cv.bind("<ButtonPress-1>", self._on_btn1)
        self._cv.bind("<B1-Motion>", self._on_drag)
        self._cv.bind("<ButtonRelease-1>", self._on_release)
        # wheel zoom
        self._cv.bind("<MouseWheel>", self._on_wheel)     # Win/mac
        self._cv.bind("<Button-4>", self._on_wheel_up)    # X11 up
        self._cv.bind("<Button-5>", self._on_wheel_down)  # X11 down

    def _on_motion(self, e: tk.Event) -> None:  # type: ignore[override]
        sx, sy = int(e.x), int(e.y)
        if self._cross_enabled:
            cw, ch = self._cv.winfo_width(), self._cv.winfo_height()
            self._cv.coords(self._cross_h, 0, sy, cw, sy)
            self._cv.coords(self._cross_v, sx, 0, sx, ch)
            # upewnij się, że krzyż jest nad obrazami
            try:
                self._cv.tag_raise("__cross__")
            except Exception:
                pass
        self._emit("ui.image.crosshair.moved", {"sx": sx, "sy": sy})
        ix, iy = self.screen_to_image(sx, sy)
        self._emit("ui.cursor.pos", {"x": ix, "y": iy})

    def _on_leave(self, _e=None) -> None:
        if self._cross_enabled:
            self._cv.coords(self._cross_h, 0, 0, 0, 0)
            self._cv.coords(self._cross_v, 0, 0, 0, 0)

    def _on_btn1(self, e: tk.Event) -> None:  # type: ignore[override]
        self._panning = True
        self._pan_anchor = (int(e.x), int(e.y))

    def _on_drag(self, e: tk.Event) -> None:  # type: ignore[override]
        if not self._panning:
            return
        ax, ay = self._pan_anchor
        dx, dy = int(e.x) - ax, int(e.y) - ay
        px, py = self._pan
        self._pan = (px + dx, py + dy)
        self._pan_anchor = (int(e.x), int(e.y))
        self._render_all()

    def _on_release(self, _e: tk.Event) -> None:  # type: ignore[override]
        self._panning = False

    def _on_wheel(self, e: tk.Event) -> None:  # type: ignore[override]
        delta = int(getattr(e, "delta", 0))
        factor = 1.1 if delta > 0 else (1 / 1.1)
        self._zoom_around((e.x, e.y), factor)

    def _on_wheel_up(self, e: tk.Event) -> None:  # X11
        self._zoom_around((e.x, e.y), 1.1)

    def _on_wheel_down(self, e: tk.Event) -> None:  # X11
        self._zoom_around((e.x, e.y), 1 / 1.1)

    # ─────────────────────────── internals: render ─────────────────────────────

    def _render_all(self) -> None:
        # nie czyść __cross__ przed pobraniem jego geometrii – odtworzymy po rysowaniu
        # Zapamiętaj istniejące współrzędne krzyża
        ch_coords = (self._cv.coords(self._cross_h), self._cv.coords(self._cross_v))
        self._cv.delete("all")

        # odtwórz crosshair linie (po delete("all") też znikają)
        self._cross_h = self._cv.create_line(*((ch_coords[0] + [0, 0, 0, 0])[:4]), fill=self.CROSS_FG, width=1, tags=("__cross__",))
        self._cross_v = self._cv.create_line(*((ch_coords[1] + [0, 0, 0, 0])[:4]), fill=self.CROSS_FG, width=1, tags=("__cross__",))

        if not self._layers:
            return

        ox, oy = self._pan
        for layer in self._layers:
            if not layer.visible:
                continue
            iw, ih = layer.image.size
            disp_w = max(1, int(round(iw * self._zoom)))
            disp_h = max(1, int(round(ih * self._zoom)))
            try:
                if Image is not None and ImageTk is not None:
                    pil = layer.image if (disp_w, disp_h) == layer.image.size else layer.image.resize((disp_w, disp_h), Image.LANCZOS)
                    layer.photo = ImageTk.PhotoImage(pil)
                else:
                    layer.photo = None
            except Exception:
                layer.photo = None
                continue
            dx, dy = layer.offset
            sx = int(round(ox + dx * self._zoom))
            sy = int(round(oy + dy * self._zoom))
            if layer.photo is not None:
                self._cv.create_image(sx, sy, anchor="nw", image=layer.photo)

        # crosshair zawsze na wierzchu
        try:
            self._cv.tag_raise("__cross__")
        except Exception:
            pass

    def _rebuild_photos(self, indexes: Optional[List[int]] = None) -> None:
        if ImageTk is None:
            for i in (indexes or range(len(self._layers))):
                self._layers[int(i)].photo = None
            return
        targets = [int(i) for i in (indexes or range(len(self._layers)))]
        for i in targets:
            lyr = self._layers[i]
            try:
                lyr.photo = ImageTk.PhotoImage(lyr.image)
            except Exception:
                lyr.photo = None

    def _get_base_image(self) -> Optional["Image.Image"]:
        if not self._layers:
            return None
        idx = self._active if self._active is not None else 0
        idx = max(0, min(len(self._layers) - 1, idx))
        return self._layers[idx].image

    def _zoom_around(self, pivot: Tuple[float, float], factor: float) -> None:
        new_zoom = max(self._min_zoom, min(self._max_zoom, self._zoom * float(factor)))
        factor = new_zoom / (self._zoom if self._zoom else 1.0)
        px, py = pivot
        ox, oy = self._pan
        self._pan = (px - (px - ox) * factor, py - (py - oy) * factor)
        self._zoom = new_zoom
        self._render_all()

    # ─────────────────────────── helpers ─────────────────────────────

    def _emit(self, topic: str, payload: dict) -> None:
        if self.bus and hasattr(self.bus, "publish"):
            try:
                self.bus.publish(topic, payload)
            except Exception:
                pass

    @staticmethod
    def _to_pil(obj: Any) -> Optional["Image.Image"]:
        if Image is None:
            return None
        try:
            if hasattr(Image, "Image") and isinstance(obj, Image.Image):
                return obj.convert("RGB") if getattr(obj, "mode", "") != "RGB" else obj
            if np is not None and isinstance(obj, np.ndarray):
                arr = obj
                if arr.ndim == 2:  # gray -> RGB
                    a = arr.astype("float32")
                    if a.max() <= 1.001:
                        a *= 255.0
                    a = a.clip(0, 255).astype("uint8")
                    arr = np.stack([a, a, a], axis=-1)
                elif arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
                    a = arr.astype("float32")
                    if a.max() <= 1.001:
                        a *= 255.0
                    a = a.clip(0, 255).astype("uint8")
                    if a.shape[-1] == 1:
                        a = np.repeat(a, 3, axis=-1)
                    if a.shape[-1] == 4:
                        rgb = a[..., :3].astype("float32")
                        alpha = a[..., 3:4].astype("float32") / 255.0
                        a = (rgb * alpha).clip(0, 255).astype("uint8")
                    arr = a
                else:
                    return None
                return Image.fromarray(arr, "RGB")
        except Exception:
            return None
        return None
