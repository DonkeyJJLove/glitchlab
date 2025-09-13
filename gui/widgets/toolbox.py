# glitchlab/gui/widgets/toolbox.py
"""
Backwards-compatible Toolbox widget for GlitchLab's GUI.

Zgodność wstecz:
- Obsługuje stare nazwy argumentów z app.py: `on_mode_change`, `on_toggle`.
- Wywołuje `on_toggle("cross", bool)` / `on_toggle("rulers", bool)` przy zmianach
  checkboxów (z bezpiecznym fallbackiem do 1-argumentowej sygnatury).
- Udostępnia metody: set_rulers(bool), set_cross(bool), set_crosshair(bool),
  set_mode(str), set_tool(str), current_tool (property).

Nowe API (opcjonalne):
- on_tool_changed(tool: str)
- on_toggle_crosshair(enabled: bool)
- on_toggle_rulers(enabled: bool)
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional


def _safe_toggle_call(cb: Callable, name: str, enabled: bool) -> None:
    """Wywołaj cb(name, enabled); jeśli nie przyjmuje 2 arg., spróbuj cb(enabled)."""
    try:
        cb(name, enabled)
    except TypeError:
        cb(enabled)


class Toolbox(ttk.Frame):
    """
    Parameters
    ----------
    parent : tk.Misc
        Kontener-rodzic.
    on_tool_changed : Callable[[str], None], optional
        Callback przy zmianie narzędzia.
    on_toggle_crosshair : Callable[[bool], None], optional
        Callback przy zmianie przełącznika 'crosshair'.
    on_toggle_rulers : Callable[[bool], None], optional
        Callback przy zmianie przełącznika 'rulers'.

    Zgodność wstecz (legacy app.py):
    on_mode_change : Callable[[str], None], optional
        Alias dla `on_tool_changed`.
    on_toggle : Callable[..., None], optional
        Wspólny callback dla przełączników; wywoływany jako (name, enabled)
        z bezpiecznym fallbackiem do sygnatury 1-argumentowej.
    **kwargs : dowolne
        Ignorowane — żeby starsze wywołania nie rzucały TypeError.
    """

    def __init__(
        self,
        parent: tk.Misc,
        *,
        on_tool_changed: Optional[Callable[[str], None]] = None,
        on_toggle_crosshair: Optional[Callable[[bool], None]] = None,
        on_toggle_rulers: Optional[Callable[[bool], None]] = None,
        # legacy aliases
        on_mode_change: Optional[Callable[[str], None]] = None,
        on_toggle: Optional[Callable[..., None]] = None,
        **kwargs,
    ) -> None:
        super().__init__(parent)

        # Normalizacja aliasów
        self._on_rulers_toggle = None
        self.on_tool_changed = on_tool_changed or on_mode_change
        self.on_toggle_crosshair = on_toggle_crosshair
        self.on_toggle_rulers = on_toggle_rulers
        self.on_toggle = on_toggle

        # Stan
        self.tool_var = tk.StringVar(value="hand")
        self.crosshair_var = tk.BooleanVar(value=False)
        self.rulers_var = tk.BooleanVar(value=True)

        # UI
        self._build_ui()

    # ---------------- UI ----------------

    def _build_ui(self) -> None:
        tools_box = ttk.LabelFrame(self, text="Tools")
        tools_box.pack(fill="x", padx=4, pady=(4, 2))

        tools = [
            ("hand", "Pan"),
            ("zoom", "Zoom"),
            ("pick", "Pick"),
            ("measure", "Measure"),
        ]
        for value, label in tools:
            ttk.Radiobutton(
                tools_box,
                text=label,
                value=value,
                variable=self.tool_var,
                command=self._on_tool_change,
            ).pack(anchor="w", padx=4, pady=1)

        ov_box = ttk.LabelFrame(self, text="Overlays")
        ov_box.pack(fill="x", padx=4, pady=(2, 4))

        self._ch_box = ttk.Checkbutton(
            ov_box,
            text="Crosshair",
            variable=self.crosshair_var,
            command=self._on_crosshair_toggle,
        )
        self._ch_box.pack(anchor="w", padx=4, pady=1)

        self._ruler_box = ttk.Checkbutton(
            ov_box,
            text="Rulers",
            variable=self.rulers_var,
            command=self._on_rulers_toggle,
        )
        self._ruler_box.pack(anchor="w", padx=4, pady=1)

    # ------------- Handlers -------------

    def _on_tool_change(self) -> None:
        if self.on_tool_changed:
            self.on_tool_changed(self.tool_var.get())

    def _on_crosshair_toggle(self) -> None:
        val = bool(self.crosshair_var.get())
        if self.on_toggle_crosshair:
            self.on_toggle_crosshair(val)

