# glitchlab/gui/views/viewport.py
# -*- coding: utf-8 -*-
"""
Viewport — kontener modułu obrazu (Toolbox + ImageCanvas).

Rola:
- Składa:   [Toolbox] nad [ImageCanvas]
- Integruje EventBus i (opcjonalnie) LayerManager
- Nie duplikuje logiki renderu/zoom/pan (deleguje do ImageCanvas)

Publikowane/obsługiwane zdarzenia (EventBus):
  pub: ui.image.tool.change {tool, opts}
       ui.image.view.fit    {}
       ui.image.view.reset  {}
       ui.image.rulers      {visible: bool}
  sub: ui.layers.changed    {}
       ui.image.tool.change {tool, opts}
       ui.mask.visibility   {mask_key, visible, alpha?}
       ui.image.view.changed{zoom?, pan?}
       ui.image.view.fit    {}
       ui.image.view.reset  {}

Zależności opcjonalne: PIL; Toolbox; ImageCanvas; LayerManager
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Callable

import tkinter as tk
from tkinter import ttk

# ── opcjonalne zależności ─────────────────────────────────────────────────────
try:
    from PIL import Image  # noqa: F401
except Exception:  # pragma: no cover
    Image = None  # type: ignore

try:
    from glitchlab.gui.widgets.image_canvas import ImageCanvas
except Exception:  # pragma: no cover
    ImageCanvas = None  # type: ignore

try:
    from glitchlab.gui.widgets.toolbox import Toolbox
except Exception:  # pragma: no cover
    Toolbox = None  # type: ignore

try:
    from gui.services.layer_manager import LayerManager
except Exception:  # pragma: no cover
    LayerManager = None  # type: ignore


def _map_tool(name: str) -> str:
    """Mapowanie nazw z Toolbox → kanoniczne nazwy narzędzi canvasa."""
    return {"hand": "view", "zoom": "zoom", "pick": "pipette", "measure": "measure"}.get(name, "view")


class _ToolbarFallback(ttk.Frame):
    """Awaryjny pasek narzędzi, gdy Toolbox nie jest dostępny."""
    def __init__(self, master: tk.Misc, publish: Callable[[str, Dict[str, Any]], None]) -> None:
        super().__init__(master)
        self.publish = publish
        self.columnconfigure(10, weight=1)
        ttk.Label(self, text="(Toolbox placeholder)", foreground="#888").grid(row=0, column=0, padx=6, pady=4)

        def btn(txt: str, cmd):  # prosty helper
            b = ttk.Button(self, text=txt, command=cmd)
            b.grid(row=1, column=btn.col, padx=2, pady=(0, 6))
            btn.col += 1
        btn.col = 0  # type: ignore[attr-defined]

        btn("View",    lambda: self.publish("ui.image.tool.change", {"tool": "view"}))
        btn("Rect",    lambda: self.publish("ui.image.tool.change", {"tool": "rect"}))
        btn("Brush",   lambda: self.publish("ui.image.tool.change", {"tool": "brush"}))
        btn("Pipette", lambda: self.publish("ui.image.tool.change", {"tool": "pipette"}))
        btn("Measure", lambda: self.publish("ui.image.tool.change", {"tool": "measure"}))
        ttk.Separator(self, orient="vertical").grid(row=1, column=btn.col, sticky="ns", padx=6); btn.col += 1
        btn("Fit",     lambda: self.publish("ui.image.view.fit", {}))
        btn("Reset",   lambda: self.publish("ui.image.view.reset", {}))


class Viewport(ttk.Frame):
    """
    Kontener: [Toolbox] nad [ImageCanvas].
    Dba o wiring EventBus i (opcjonalnie) reagowanie na zmiany warstw.
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
        self._publish: Callable[[str, Dict[str, Any]], None] = (
            getattr(bus, "publish", None) or (lambda *_: None)
        )

        # Layout
        self.rowconfigure(1 if toolbar_top else 0, weight=1)
        self.columnconfigure(0, weight=1)

        # Toolbar: preferuj istniejący Toolbox
        if Toolbox is not None:
            self.toolbar = Toolbox(
                self,
                on_tool_changed=lambda t: self._publish("ui.image.tool.change", {"tool": _map_tool(t), "opts": {}}),
                on_toggle_crosshair=lambda s: self._set_crosshair(bool(s)),
                on_toggle_rulers=lambda s: self._publish("ui.image.rulers", {"visible": bool(s)}),
            )
        else:
            self.toolbar = _ToolbarFallback(self, publish=self._publish)

        # Canvas
        canvas_kwargs = dict(canvas_kwargs or {})
        if ImageCanvas is not None:
            # Przekaż publish i (opcjonalnie) LayerManager, jeżeli ImageCanvas to obsługuje.
            try:
                self.canvas = ImageCanvas(self, publish=self._publish, layer_manager=self._lm, **canvas_kwargs)  # type: ignore[arg-type]
            except TypeError:
                self.canvas = ImageCanvas(self, **canvas_kwargs)  # type: ignore[call-arg]
        else:
            self.canvas = tk.Canvas(self, bg="black", highlightthickness=0)  # type: ignore[assignment]

        # Pack
        if toolbar_top:
            self.toolbar.grid(row=0, column=0, sticky="ew")
            self.canvas.grid(row=1, column=0, sticky="nsew")
        else:
            self.canvas.grid(row=0, column=0, sticky="nsew")
            self.toolbar.grid(row=1, column=0, sticky="ew")

        self._wire_bus()

    # ── EventBus: subskrypcje ────────────────────────────────────────────────
    def _wire_bus(self) -> None:
        sub = getattr(self.bus, "subscribe", None)
        if not callable(sub):
            return

        # Warstwy → odśwież widok
        sub("ui.layers.changed", lambda _p: self.refresh_from_layers())

        # Zmiana narzędzia (może przyjść z innego miejsca niż Toolbox)
        sub("ui.image.tool.change", lambda p: self.set_tool(p.get("tool"), p.get("opts")))

        # Widoczność/alpha maski
        sub("ui.mask.visibility", lambda p: self.set_mask_visibility(bool(p.get("visible", True)), p.get("alpha", None)))

        # Zmiany param. widoku (np. sync z innym komponentem)
        sub("ui.image.view.changed", lambda p: self.apply_view(p.get("zoom"), p.get("pan")))

        # Fit/Reset z zewnątrz
        sub("ui.image.view.fit",   lambda _p: self.fit())
        sub("ui.image.view.reset", lambda _p: self.reset())

    # ── Delegaty do canvasa ──────────────────────────────────────────────────
    def set_image(self, image: Any) -> None:
        if hasattr(self.canvas, "set_image"):
            self.canvas.set_image(image)  # type: ignore[attr-defined]

    def set_tool(self, name: Optional[str], opts: Optional[Dict[str, Any]] = None) -> None:
        if not name:
            return
        # Jeżeli ImageCanvas ma własne API narzędzi, użyjemy; w innym razie ignorujemy.
        if hasattr(self.canvas, "set_tool"):
            try:
                self.canvas.set_tool(str(name), dict(opts or {}))  # type: ignore[attr-defined]
            except Exception:
                pass  # brak obsługi — doimplementujemy po stronie canvasa

    def fit(self) -> None:
        if hasattr(self.canvas, "fit"):
            self.canvas.fit()  # type: ignore[attr-defined]
        elif hasattr(self.canvas, "fit_to_window"):
            self.canvas.fit_to_window()  # type: ignore[attr-defined]

    def reset(self) -> None:
        # Proste: fit + (opcjonalnie) center
        self.fit()
        if hasattr(self.canvas, "center"):
            self.canvas.center()  # type: ignore[attr-defined]

    def set_mask_visibility(self, visible: bool, alpha: float | None = None) -> None:
        if hasattr(self.canvas, "set_mask_visibility"):
            self.canvas.set_mask_visibility(visible, alpha)  # type: ignore[attr-defined]

    def apply_view(self, zoom: float | None = None, pan: Tuple[int, int] | None = None) -> None:
        if hasattr(self.canvas, "apply_view"):
            if isinstance(pan, list) and len(pan) == 2:
                pan = (int(pan[0]), int(pan[1]))
            self.canvas.apply_view(zoom, pan)  # type: ignore[attr-defined]
        else:
            if zoom is not None and hasattr(self.canvas, "set_zoom"):
                self.canvas.set_zoom(float(zoom))  # type: ignore[attr-defined]
            # pan pomijamy bez natywnego wsparcia

    def refresh_from_layers(self) -> None:
        if hasattr(self.canvas, "refresh_from_layers"):
            self.canvas.refresh_from_layers()  # type: ignore[attr-defined]
        elif hasattr(self.canvas, "update_idletasks"):
            self.canvas.update_idletasks()

    # ── Drobne wygody ────────────────────────────────────────────────────────
    def get_canvas(self) -> Any:
        return self.canvas

    def get_toolbar(self) -> Any:
        return self.toolbar

    def _set_crosshair(self, state: bool) -> None:
        if hasattr(self.canvas, "set_crosshair"):
            try:
                self.canvas.set_crosshair(bool(state))  # type: ignore[attr-defined]
            except Exception:
                pass
