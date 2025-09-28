# -*- coding: utf-8 -*-
"""
RĘCZNY TEST widoku: LayerCanvas (+ opcjonalnie LayersPanel).
Uruchomienie:
  python -m glitchlab.tests.test_layer_viewer_manual [ścieżka_do_obrazka]

Co sprawdza:
  • Czy obraz jest widoczny na LayerCanvas (pojedynczy Canvas).
  • Pan/zoom (LPM-drag + kółko myszy).
  • Crosshair (linia h/v za kursorem).
  • Kilka warstw (podstawowa + nakładka przesunięta).
  • Integracja z LayersPanel (jeśli zainstalowany): add/remove/reorder/visible/opacity/blend.
  • EventBus – loguje emitowane eventy do konsoli.

Skróty:
  F  – Fit to window
  C  – Center
  +  – Zoom in (x1.1)
  -  – Zoom out (x/1.1)
  1/2 – Ustaw warstwę aktywną (0..n-1)
  V  – Toggle visibility aktywnej warstwy
  Strzałki – Nudge offset aktywnej warstwy (±5 px)
"""

from __future__ import annotations

import os
import sys
import tkinter as tk
from tkinter import ttk
from typing import Any, Dict, Optional

# Pillow do generowania prostych obrazów testowych
try:
    from PIL import Image, ImageDraw, ImageOps
except Exception:
    Image = ImageDraw = ImageOps = None  # type: ignore

# Spróbuj pobrać LayerCanvas i opcjonalny LayersPanel
try:
    from glitchlab.gui.widgets.layer_canvas import LayerCanvas  # type: ignore
except Exception as e:
    print("[FATAL] Brak LayerCanvas:", e)
    raise

try:
    from glitchlab.gui.views.layer_panel import LayersPanel  # type: ignore
except Exception:
    LayersPanel = None  # type: ignore


# ───────────────────────────── prosty EventBus do logowania ──────────────────
class _Bus:
    def __init__(self, root: tk.Misc) -> None:
        self._handlers: Dict[str, list] = {}
        self.root = root

    def publish(self, topic: str, payload: Dict[str, Any]) -> None:
        print(f"[BUS] {topic} {payload}")
        for fn in self._handlers.get(topic, []):
            try:
                fn(topic, payload)
            except Exception as e:
                print("[BUS handler error]", e)

    def subscribe(self, topic: str, fn) -> None:
        self._handlers.setdefault(topic, []).append(fn)


# ───────────────────────────── obrazy testowe ─────────────────────────────────
def _make_checker(size=(1024, 768), cell=64):
    assert Image is not None
    w, h = size
    im = Image.new("RGB", size, (235, 235, 235))
    draw = ImageDraw.Draw(im)
    c1 = (210, 210, 210); c2 = (245, 245, 245)
    for y in range(0, h, cell):
        for x in range(0, w, cell):
            c = c1 if ((x // cell + y // cell) % 2 == 0) else c2
            draw.rectangle([x, y, x + cell - 1, y + cell - 1], fill=c)
    # ramka
    draw.rectangle([0, 0, w - 1, h - 1], outline=(60, 60, 60), width=3)
    return im

def _make_overlay(size=(1024, 768)):
    assert Image is not None
    w, h = size
    im = Image.new("RGB", size, (0, 0, 0))
    draw = ImageDraw.Draw(im)
    # kilka kolorowych kształtów do łatwego rozróżnienia
    draw.ellipse([w//4, h//4, w//4 + 240, h//4 + 240], outline=(255, 64, 64), width=8)
    draw.rectangle([w//2 - 160, h//2 - 100, w//2 + 160, h//2 + 100], outline=(64, 160, 255), width=8)
    draw.line([0, 0, w, h], fill=(64, 200, 64), width=6)
    draw.text((20, 20), "Overlay", fill=(255, 128, 0))
    return im


# ───────────────────────────── main UI ────────────────────────────────────────
def main():
    root = tk.Tk()
    root.title("GlitchLab – manualny test LayerCanvas")

    # layout: viewer po lewej, (opcjonalnie) panel warstw po prawej
    pw = ttk.Panedwindow(root, orient="horizontal")
    pw.pack(fill="both", expand=True)

    bus = _Bus(root)
    left = ttk.Frame(pw); left.rowconfigure(0, weight=1); left.columnconfigure(0, weight=1)
    viewer = LayerCanvas(left, bus=bus)
    viewer.grid(row=0, column=0, sticky="nsew")
    pw.add(left, weight=70)

    right = ttk.Frame(pw)
    panel = None
    if LayersPanel is not None:
        panel = LayersPanel(right, bus=bus)
        panel.pack(fill="both", expand=True)
        pw.add(right, weight=30)

    # wejście: ścieżka lub generowany checker
    img_path = sys.argv[1] if len(sys.argv) > 1 else None
    if Image is None:
        raise RuntimeError("Pillow jest wymagany dla testu.")

    if img_path and os.path.exists(img_path):
        base = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")
    else:
        base = _make_checker((1024, 768))

    overlay = _make_overlay(base.size)

    # ustaw 2 warstwy (druga przesunięta)
    viewer.set_layers([base, overlay], names=["Background", "Overlay"])
    viewer.set_layer_offset(1, 120, 80)
    viewer.zoom_fit()

    # jeśli mamy panel warstw, karm go snapshotem (symulacja App)
    def _push_snapshot(_topic=None, _payload=None):
        if panel is None:
            return
        # minimalny snapshot dla panelu
        layers = []
        for i, nm in enumerate(["Background", "Overlay"]):
            layers.append({
                "id": str(i),
                "name": nm,
                "visible": True,
                "opacity": 1.0,
                "blend": "normal",
            })
        panel.set_snapshot(layers, active="0")

    _push_snapshot()

    # Klawisze testowe
    def zoom_in(_e=None):
        viewer.set_zoom(viewer.get_zoom() * 1.1)

    def zoom_out(_e=None):
        viewer.set_zoom(viewer.get_zoom() / 1.1)

    def fit(_e=None):
        viewer.zoom_fit()

    def center(_e=None):
        viewer.center()

    def set_active(idx: int):
        def _inner(_e=None):
            viewer.set_active_layer(idx)
            print("[TEST] active layer =", viewer.get_active_layer())
        return _inner

    def toggle_vis(_e=None):
        idx = viewer.get_active_layer()
        if idx is None:
            return
        # odczyt widoczności z wewnętrznej listy (tylko do testu)
        visible = True
        try:
            visible = bool(viewer._LayerCanvas__dict__)  # trigger NameError – zostawiamy fallback niżej
        except Exception:
            pass
        try:
            visible = viewer._LayerCanvas__layers[idx].visible  # pyright: ignore
        except Exception:
            # nie ma dostępu – zrobimy heurystykę: spróbuj przełączyć zawsze na odwrotność
            pass
        # bezpiecznie toggluj
        try:
            viewer.set_layer_visible(idx, not visible)
        except Exception:
            viewer.set_layer_visible(idx, False)

    def nudge(dx: int, dy: int):
        def _inner(_e=None):
            idx = viewer.get_active_layer()
            if idx is None:
                return
            try:
                # Pobierz aktualny offset przez „screen_to_image” na (0,0)? Nie – wewnętrzny stan mamy przez API
                # więc róbmy inkrementalnie: najpierw nic nie wiemy – zatem przesuń o +/− i zostaw rysunek
                # (dla demo wystarczy).
                viewer.set_layer_offset(idx, dx + 0, dy + 0)  # inkrementy – jeżeli chcesz kumulować, rozbuduj API
            except Exception:
                pass
        return _inner

    # mapowanie klawiszy
    root.bind("+", zoom_in)
    root.bind("=", zoom_in)
    root.bind("-", zoom_out)
    root.bind("_", zoom_out)
    root.bind("f", fit)
    root.bind("F", fit)
    root.bind("c", center)
    root.bind("C", center)
    root.bind("1", set_active(0))
    root.bind("2", set_active(1))
    root.bind("v", toggle_vis)
    root.bind("V", toggle_vis)

    # strzałki – drobne przesunięcia (5 px w układzie obrazu → przeskalowane automatycznie)
    def _arrow(dx, dy):
        def _h(_e=None):
            idx = viewer.get_active_layer()
            if idx is None:
                return
            # odczytaj stary offset (prywatny – do testu)
            old = (0, 0)
            try:
                old = viewer._LayerCanvas__layers[idx].offset  # type: ignore
            except Exception:
                try:
                    old = viewer._layers[idx].offset  # type: ignore
                except Exception:
                    pass
            viewer.set_layer_offset(idx, old[0] + dx, old[1] + dy)
        return _h

    root.bind("<Left>", _arrow(-5, 0))
    root.bind("<Right>", _arrow(5, 0))
    root.bind("<Up>", _arrow(0, -5))
    root.bind("<Down>", _arrow(0, 5))

    print("== Manualny test LayerCanvas ==")
    print("Klawisze: F=Fit, C=Center, +/-=Zoom, 1/2=aktywna warstwa, V=toggle visible, Strzałki=nudge")
    print("Jeśli nie widzisz obrazu: sprawdź, czy masz Pillow; odpal bez ścieżki aby użyć obrazów testowych.")

    root.geometry("1200x800")
    root.mainloop()


if __name__ == "__main__":
    main()
