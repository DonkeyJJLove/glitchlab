# glitchlab/gui/views/left_toolbar.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any, Dict, Optional


class LeftToolBar(ttk.Frame):
    """
    Pionowy pasek narzędzi (ikony/teksty) po lewej stronie.
    Publikuje zdarzenia na EventBus:
      - ui.tools.select {name: str}      # wybór aktywnego narzędzia
    Reaguje na zdarzenia run.* (best-effort), aby blokować interakcje podczas wykonywania:
      - run.progress {value: 0..1}
      - run.done / run.error

    Minimalny kontrakt busa: bus.publish(topic: str, payload: dict)
    (subskrypcja run.* jest opcjonalna — jeśli bus ma .subscribe, to się podpinamy)
    """

    # Kolejność i definicja narzędzi (name -> label)
    TOOLS: Dict[str, str] = {
        "pan": "Pan",
        "zoom": "Zoom",
        "ruler": "Ruler",
        "probe": "Probe",
        "pick": "Pick Color",
        # 3D – na przyszłość, pokazywane tylko, jeśli caller tak zdecyduje
        "orbit3d": "3D Orbit",
        "move3d": "3D Move",
    }

    def __init__(
        self,
        master: tk.Misc,
        *,
        bus: Optional[Any] = None,
        show_3d: bool = False,
        initial: str = "pan",
    ):
        super().__init__(master)
        self.bus = bus
        self._current = tk.StringVar(value=initial if initial in self.TOOLS else "pan")
        self._buttons: Dict[str, ttk.Radiobutton] = {}

        self.columnconfigure(0, weight=1)

        # Sekcja: 2D
        ttk.Label(self, text="2D Tools", anchor="w").grid(row=0, column=0, sticky="ew", padx=6, pady=(6, 2))

        row = 1
        for key in ("pan", "zoom", "ruler", "probe", "pick"):
            self._add_tool_button(key, text=self.TOOLS[key], row=row)
            row += 1

        # Separator
        ttk.Separator(self, orient="horizontal").grid(row=row, column=0, sticky="ew", padx=6, pady=6)
        row += 1

        # Sekcja: 3D (opcjonalna)
        if show_3d:
            ttk.Label(self, text="3D (experimental)", anchor="w").grid(row=row, column=0, sticky="ew", padx=6, pady=(0, 2))
            row += 1
            for key in ("orbit3d", "move3d"):
                self._add_tool_button(key, text=self.TOOLS[key], row=row)
                row += 1

        # Wstępny publish aktywnego narzędzia
        self.after(0, lambda: self._publish("ui.tools.select", {"name": self._current.get()}))

        # Opcjonalne subskrypcje run.* (jeśli EventBus ma .subscribe)
        self._wire_run_signals()

    # --------------------------------------------------------------------- UI

    def _add_tool_button(self, name: str, *, text: str, row: int) -> None:
        b = ttk.Radiobutton(
            self,
            text=text,
            value=name,
            variable=self._current,
            style="Toolbutton",
            command=lambda n=name: self._on_select(n),
        )
        b.grid(row=row, column=0, sticky="ew", padx=6, pady=2)
        self._buttons[name] = b

    def _on_select(self, name: str) -> None:
        self._current.set(name)
        self._publish("ui.tools.select", {"name": name})

    # -------------------------------------------------------------- BUS / RUN

    def _wire_run_signals(self) -> None:
        """Podłącz się do run.* jeśli bus exposes subscribe()."""
        if self.bus is None or not hasattr(self.bus, "subscribe"):
            return

        def on_progress(_t: str, d: Dict[str, Any]) -> None:
            # Jeżeli 0 < value < 1 blokujemy; przy 0 lub >=1 odblokowujemy
            try:
                v = float((d or {}).get("value") or 0.0)
            except Exception:
                v = 0.0
            self.set_enabled(not (0.0 < v < 1.0))

        def on_done(_t: str, _d: Dict[str, Any]) -> None:
            self.set_enabled(True)

        def on_error(_t: str, _d: Dict[str, Any]) -> None:
            self.set_enabled(True)

        try:
            self.bus.subscribe("run.progress", on_progress)
            self.bus.subscribe("run.done", on_done)
            self.bus.subscribe("run.error", on_error)
        except Exception:
            pass

    # --------------------------------------------------------------- PUBLIC API

    def set_enabled(self, enabled: bool) -> None:
        """Włącza/wyłącza interaktywność wszystkich przycisków."""
        try:
            for b in self._buttons.values():
                b.state(("!disabled",) if enabled else ("disabled",))
        except Exception:
            for b in self._buttons.values():
                try:
                    b.configure(state=("normal" if enabled else "disabled"))
                except Exception:
                    pass

    def get_selected(self) -> str:
        return self._current.get()

    # -------------------------------------------------------------- UTILITIES

    def _publish(self, topic: str, payload: Dict[str, Any]) -> None:
        if self.bus is not None and hasattr(self.bus, "publish"):
            try:
                self.bus.publish(topic, dict(payload))
            except Exception:
                pass
