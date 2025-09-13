# glitchlab/gui/widgets/panel_toolbox.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional, Dict


class Toolbox(ttk.Frame):
    """
    Lekki widget paska narzędzi do pracy z widokiem/obrazem.

    Zakres:
      - Wybór trybu narzędzia: hand / zoom / pick / measure
      - Przełączniki: crosshair, rulers
      - API do synchronizacji stanu z aplikacją (setters + snapshot)

    Integracja (app.py):
      toolbox = Toolbox(parent,
                        on_mode_change=lambda m: ...,
                        on_toggle=lambda key, state: ...)
      toolbox.pack(...)
      # aktualizacja z app:
      toolbox.set_crosshair(True/False)
      toolbox.set_rulers(True/False)
      toolbox.set_mode("hand")

    Zdarzenia:
      - callback on_mode_change(mode: str) wywoływany przy zmianie trybu
      - callback on_toggle(key: str, state: bool) dla "crosshair" | "rulers"

    Styl:
      - Używa przycisków typu Toolbutton (ttk) dla spójnego look&feel
    """

    MODES = (
        ("🖐", "hand",    "Pan / drag"),
        ("🔍", "zoom",    "Zoom"),
        ("🎯", "pick",    "Color picker"),
        ("📏", "measure", "Measure"),
    )

    TOGGLES = (
        ("⊕", "crosshair", "Show crosshair"),
        ("📐", "rulers",    "Show rulers"),
    )

    def __init__(
        self,
        master: tk.Misc,
        on_mode_change: Optional[Callable[[str], None]] = None,
        on_toggle: Optional[Callable[[str, bool], None]] = None,
        **kw,
    ) -> None:
        super().__init__(master, **kw)

        self.on_mode_change = on_mode_change
        self.on_toggle = on_toggle

        # --- style (subtelne, aby przypominały toolbar) ---
        self._ensure_styles()

        # --- stan ---
        self.var_mode = tk.StringVar(value="hand")
        self.var_crosshair = tk.BooleanVar(value=True)
        self.var_rulers = tk.BooleanVar(value=True)

        # --- układ: lewa sekcja = tryby, prawa = przełączniki ---
        left = ttk.Frame(self)
        right = ttk.Frame(self)
        left.pack(side="left")
        ttk.Separator(self, orient="vertical").pack(side="left", fill="y", padx=8)
        right.pack(side="left")

        # tryby (radiobuttony w stylu przycisków)
        self._mode_buttons: Dict[str, ttk.Radiobutton] = {}
        for i, (label, value, _tip) in enumerate(self.MODES):
            b = ttk.Radiobutton(
                left,
                text=label,
                value=value,
                variable=self.var_mode,
                style="Toolbox.Toolbutton",
                command=self._emit_mode_change,
            )
            b.grid(row=0, column=i, padx=(0 if i == 0 else 4), pady=0)
            self._mode_buttons[value] = b

        # przełączniki (checkbuttony w stylu przycisków)
        self._toggle_buttons: Dict[str, ttk.Checkbutton] = {}
        for i, (label, key, _tip) in enumerate(self.TOGGLES):
            var = self._var_for_key(key)
            b = ttk.Checkbutton(
                right,
                text=label,
                variable=var,
                style="Toolbox.Toolbutton",
                command=lambda k=key, v=var: self._emit_toggle(k, bool(v.get())),
            )
            b.grid(row=0, column=i, padx=(0 if i == 0 else 4), pady=0)
            self._toggle_buttons[key] = b

        # skróty klawiaturowe (opcjonalnie)
        self.bind_all("<KeyPress-h>", lambda _e: self.set_mode("hand"))
        self.bind_all("<KeyPress-z>", lambda _e: self.set_mode("zoom"))
        self.bind_all("<KeyPress-p>", lambda _e: self.set_mode("pick"))
        self.bind_all("<KeyPress-m>", lambda _e: self.set_mode("measure"))

    # ------------------- API: settery / snapshot -------------------

    def set_mode(self, mode: str) -> None:
        """Ustaw tryb (hand/zoom/pick/measure) bez emisji callbacka."""
        if mode not in (m[1] for m in self.MODES):
            return
        try:
            self.var_mode.set(mode)
        except Exception:
            pass

    def set_crosshair(self, state: bool) -> None:
        try:
            self.var_crosshair.set(bool(state))
        except Exception:
            pass

    def set_rulers(self, state: bool) -> None:
        try:
            self.var_rulers.set(bool(state))
        except Exception:
            pass

    def snapshot(self) -> dict:
        """Zwraca aktualny stan widgetu (do diagnostyki/logów)."""
        return {
            "mode": self.var_mode.get(),
            "crosshair": bool(self.var_crosshair.get()),
            "rulers": bool(self.var_rulers.get()),
        }

    # ------------------- wewnętrzne emisje -------------------

    def _emit_mode_change(self) -> None:
        if callable(self.on_mode_change):
            try:
                self.on_mode_change(self.var_mode.get())
            except Exception:
                pass
        # Dla chętnych: event Tk
        try:
            self.event_generate("<<ToolboxModeChanged>>")
        except Exception:
            pass

    def _emit_toggle(self, key: str, state: bool) -> None:
        if callable(self.on_toggle):
            try:
                self.on_toggle(key, state)
            except Exception:
                pass
        # Dla chętnych: event Tk (z kluczem w data)
        try:
            self.event_generate("<<ToolboxToggle>>", data=key)
        except Exception:
            pass

    # ------------------- narzędzia -------------------

    def _var_for_key(self, key: str) -> tk.BooleanVar:
        if key == "crosshair":
            return self.var_crosshair
        if key == "rulers":
            return self.var_rulers
        # fallback (nie powinno się zdarzyć)
        return tk.BooleanVar(value=False)

    @staticmethod
    def _ensure_styles() -> None:
        """Definiuje delikatny styl przycisku narzędziowego."""
        try:
            style = ttk.Style()
            # Na wypadek różnych motywów – modyfikujemy minimalnie
            style.configure(
                "Toolbox.Toolbutton",
                padding=(6, 2),
            )
            style.map(
                "Toolbox.Toolbutton",
                relief=[("pressed", "sunken"), ("!pressed", "flat")],
            )
        except Exception:
            pass


# Loader hook opcjonalny — dla wygody importu jako `Panel = Toolbox`
Panel = Toolbox
