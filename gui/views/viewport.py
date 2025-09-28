# glitchlab/gui/views/viewport.py
# -*- coding: utf-8 -*-
"""
Viewport — kontener modułu obrazu (toolbar + ImageCanvas).

Rola:
- Składa w jedną całość:
    • ImageToolbar (UI narzędzi canvasa)  [jeśli dostępny]
    • ImageCanvas (render + interakcje)
- Łączy się z LayerManagerem i EventBusem.
- Przekazuje dalej zdarzenia ui.image.* / ui.layers.* i reaguje na nie lokalnie.

Uwaga:
- Ten plik NIE implementuje własnego zoom/pan/render — robi to ImageCanvas.
- Jeśli ImageToolbar nie jest jeszcze dostępny, tworzy prosty placeholder,
  tak aby moduł działał od razu (plik 2 dostarczy właściwy toolbar).

Publiczne API (delegaty do canvasa):
    set_image(image)
    set_tool(name: str, opts: dict | None = None)
    fit()
    reset()
    set_mask_visibility(visible: bool, alpha: float | None = None)
    apply_view(zoom: float | None = None, pan: tuple[int,int] | None = None)
    refresh_from_layers()
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Callable

import tkinter as tk
from tkinter import ttk

# ── zależności opcjonalne ────────────────────────────────────────────────────
try:
    from PIL import Image
except Exception:  # pragma: no cover - środowisko bez Pillow
    Image = None  # type: ignore

try:
    from glitchlab.gui.widgets.image_canvas import ImageCanvas  # nasz canvas z narzędziami
except Exception:  # pragma: no cover
    ImageCanvas = None  # type: ignore

try:
    from glitchlab.gui.widgets.tools.image_toolbar import ImageToolbar  # właściwy toolbar (plik 2)
except Exception:  # pragma: no cover
    ImageToolbar = None  # type: ignore

try:
    from gui.services.layer_manager import LayerManager
except Exception:  # pragma: no cover
    LayerManager = None  # type: ignore


class _ToolbarPlaceholder(ttk.Frame):
    """
    Minimalny placeholder, gdy ImageToolbar nie jest jeszcze dostępny.
    Publikuje podstawowe zdarzenia tak jak ImageToolbar:
      - ui.image.tool.change (View/Rect/Brush/Pipette/Measure)
      - ui.image.view.fit / ui.image.view.reset
      - ui.mask.visibility
    """
    def __init__(self, master: tk.Misc, publish: Callable[[str, Dict[str, Any]], None]) -> None:
        super().__init__(master)
        self.publish = publish

        self.columnconfigure(10, weight=1)
        ttk.Label(self, text="Image Toolbar (placeholder)", foreground="#888").grid(row=0, column=0, padx=6, pady=4)

        def _btn(txt: str, tool: str | None = None, cmd: Optional[Callable[[], None]] = None, col: int = 0):
            if tool is not None:
                b = ttk.Button(self, text=txt, command=lambda: self.publish("ui.image.tool.change", {"tool": tool, "opts": {}}))
            else:
                b = ttk.Button(self, text=txt, command=cmd)
            b.grid(row=1, column=col, padx=2, pady=(0, 6))
            return b

        _btn("View", "view", col=0)
        _btn("Rect", "rect", col=1)
        _btn("Brush", "brush", col=2)
        _btn("Pipette", "pipette", col=3)
        _btn("Measure", "measure", col=4)
        ttk.Separator(self, orient="vertical").grid(row=1, column=5, sticky="ns", padx=6)
        _btn("Fit", None, cmd=lambda: self.publish("ui.image.view.fit", {}), col=6)
        _btn("Reset", None, cmd=lambda: self.publish("ui.image.view.reset", {}), col=7)
        ttk.Separator(self, orient="vertical").grid(row=1, column=8, sticky="ns", padx=6)
        chk_var = tk.BooleanVar(value=True)
        def _mask_toggle():
            self.publish("ui.mask.visibility", {"mask_key": "current", "visible": bool(chk_var.get()), "alpha": 0.3})
        ttk.Checkbutton(self, text="Mask overlay", variable=chk_var, command=_mask_toggle).grid(row=1, column=9, padx=6)


class Viewport(ttk.Frame):
    """
    Kontener: [Toolbar] nad [ImageCanvas].
    Dba o minimalny wiring z EventBusem i LayerManagerem.
    """

    def __init__(
        self,
        master: tk.Misc,
        *,
        bus: Optional[Any] = None,
        layer_manager: Optional[LayerManager] = None,
        canvas_kwargs: Optional[Dict[str, Any]] = None,
        toolbar_top: bool = True,
    ) -> None:
        super().__init__(master)
        self.bus = bus
        self._lm: Optional[LayerManager] = layer_manager
        self._publish = (getattr(bus, "publish", None) or (lambda *_: None))  # Callable(topic, payload)

        # Layout
        self.rowconfigure(1 if toolbar_top else 0, weight=1)
        self.columnconfigure(0, weight=1)

        # Toolbar (docelowo: ImageToolbar)
        if ImageToolbar is not None:
            self.toolbar = ImageToolbar(self, publish=self._publish)
        else:
            self.toolbar = _ToolbarPlaceholder(self, publish=self._publish)

        # ImageCanvas
        canvas_kwargs = dict(canvas_kwargs or {})
        if ImageCanvas is not None:
            self.canvas = ImageCanvas(self, publish=self._publish, layer_manager=self._lm, **canvas_kwargs)
        else:
            # awaryjny placeholder Canvas, aby nie kruszyć UI (bez interakcji)
            self.canvas = tk.Canvas(self, bg="black", highlightthickness=0)  # type: ignore[assignment]

        # Pack
        if toolbar_top:
            self.toolbar.grid(row=0, column=0, sticky="ew")
            self.canvas.grid(row=1, column=0, sticky="nsew")
        else:
            self.canvas.grid(row=0, column=0, sticky="nsew")
            self.toolbar.grid(row=1, column=0, sticky="ew")

        # Wiring bus
        self._wire_bus()

    # ── BUS wiring ───────────────────────────────────────────────────────────
    def _wire_bus(self) -> None:
        sub = getattr(self.bus, "subscribe", None)
        if not callable(sub):
            return

        # Przekładamy zdarzenia widoku/warstw na metody canvasa.
        def on_layers_changed(_payload: Dict[str, Any]):
            self.refresh_from_layers()

        def on_tool_change(payload: Dict[str, Any]):
            self.set_tool(payload.get("tool"), payload.get("opts"))

        def on_mask_visibility(payload: Dict[str, Any]):
            vis = bool(payload.get("visible", True))
            alpha = payload.get("alpha", None)
            self.set_mask_visibility(vis, alpha)

        def on_view_changed(payload: Dict[str, Any]):
            self.apply_view(payload.get("zoom", None), payload.get("pan", None))

        def on_view_fit(_payload: Dict[str, Any]):
            self.fit()

        def on_view_reset(_payload: Dict[str, Any]):
            self.reset()

        sub("ui.layers.changed", on_layers_changed)
        sub("ui.image.tool.change", on_tool_change)
        sub("ui.mask.visibility", on_mask_visibility)
        sub("ui.image.view.changed", on_view_changed)
        sub("ui.image.view.fit", on_view_fit)
        sub("ui.image.view.reset", on_view_reset)

    # ── Publiczne API (delegaty do ImageCanvas) ──────────────────────────────
    def set_image(self, image: Any) -> None:
        """
        Ustaw obraz wejściowy.
        Jeżeli korzystamy z LayerManagera, obraz trafi do warstwy i zostanie skomponowany.
        """
        if hasattr(self.canvas, "set_image"):
            self.canvas.set_image(image)  # type: ignore[attr-defined]

    def set_tool(self, name: Optional[str], opts: Optional[Dict[str, Any]] = None) -> None:
        if not name or not hasattr(self.canvas, "set_tool"):
            return
        try:
            self.canvas.set_tool(name, opts or {})  # type: ignore[attr-defined]
        except Exception:
            pass

    def fit(self) -> None:
        if hasattr(self.canvas, "fit_to_window"):
            self.canvas.fit_to_window()  # type: ignore[attr-defined]

    def reset(self) -> None:
        if hasattr(self.canvas, "reset_view"):
            self.canvas.reset_view()  # type: ignore[attr-defined]

    def set_mask_visibility(self, visible: bool, alpha: float | None = None) -> None:
        if hasattr(self.canvas, "set_mask_visibility"):
            self.canvas.set_mask_visibility(visible, alpha)  # type: ignore[attr-defined]

    def apply_view(self, zoom: float | None = None, pan: Tuple[int, int] | None = None) -> None:
        if hasattr(self.canvas, "apply_view"):
            # normalizacja pan
            if isinstance(pan, list) and len(pan) == 2:
                pan = (int(pan[0]), int(pan[1]))
            self.canvas.apply_view(zoom, pan)  # type: ignore[attr-defined]

    def refresh_from_layers(self) -> None:
        if hasattr(self.canvas, "refresh_from_layers"):
            self.canvas.refresh_from_layers()  # type: ignore[attr-defined]
        elif hasattr(self.canvas, "update_idletasks"):
            self.canvas.update_idletasks()

    # ── Dostęp do wewnętrznych komponentów (np. do dalszego wireingu) ───────
    def get_canvas(self) -> Any:
        return self.canvas

    def get_toolbar(self) -> Any:
        return self.toolbar
