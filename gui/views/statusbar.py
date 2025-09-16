# glitchlab/gui/views/statusbar.py
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Optional, Tuple


class StatusBar(ttk.Frame):
    """
    Lekki pasek stanu z panelem:
        [ Progress | Tekst ].....................................[ x=…, y=…, z? ]

    Public API:
        set_text(str)                  – główny komunikat (po lewej)
        set_coords((x, y [, z]))       – aktualizacja współrzędnych (po prawej)
        start() / stop()               – animowany progress
        log(msg)                       – (na razie no-op)
        bind_bus(bus)                  – opcjonalne podpięcie EventBus
    """

    # ------- inicjalizacja -------------------------------------------------
    def __init__(self, master: tk.Misc, *, show_progress: bool = True) -> None:
        super().__init__(master)
        self._show_progress = bool(show_progress)

        self._prog = ttk.Progressbar(self, mode="indeterminate", length=120)
        self._lbl_left = ttk.Label(self, text="Ready", anchor="w")
        self._lbl_right = ttk.Label(self, text="", anchor="e")

        if self._show_progress:
            self._prog.pack(side="left", padx=(6, 4), pady=2)
        self._lbl_left.pack(side="left", fill="x", expand=True, padx=4)
        self._lbl_right.pack(side="right", padx=6)

    # ------- tekst & progress ---------------------------------------------
    def set_text(self, text: str) -> None:
        """Ustaw/lewy komunikat."""
        try:
            self._lbl_left.config(text=text)
        except Exception:
            pass

    def start(self) -> None:
        if self._show_progress:
            try:
                self._prog.start(80)
            except Exception:
                pass

    def stop(self) -> None:
        if self._show_progress:
            try:
                self._prog.stop()
            except Exception:
                pass

    # ------- współrzędne kursora ------------------------------------------
    def set_coords(self, coords: Tuple[int, int] | Tuple[int, int, int] | None) -> None:
        """
        Ustala prawy panel na „x=…, y=… [, z=…]”.
        Przekazuj None lub pusty tuple, aby wyczyścić.
        """
        try:
            if not coords:
                self._lbl_right.config(text="")
                return
            if len(coords) == 2:
                x, y = coords
                s = f"x={x}  y={y}"
            else:
                x, y, z = coords  # type: ignore[misc]
                s = f"x={x}  y={y}  z={z}"
            self._lbl_right.config(text=s)
        except Exception:
            pass

    # alias dla wygody w EventBus
    set_right = set_coords

    # ------- prosty log ----------------------------------------------------
    def log(self, _msg: str) -> None:  # placeholder
        pass

    # ------- EventBus helper ----------------------------------------------
    def bind_bus(self, bus) -> None:
        """
        * run.progress  →  start + set_text
        * run.done      →  stop  + “Done”
        * run.error     →  stop  + “Error: …”
        * ui.status.set →  set_text
        * ui.cursor.pos →  set_coords   (payload: {x, y, z?})
        """
        try:
            bus.subscribe("run.progress", self._ev_run_progress)
            bus.subscribe("run.done",     lambda *_: (self.stop(), self.set_text("Done")))
            bus.subscribe("run.error",    lambda _t, d: (self.stop(),
                                                         self.set_text(f"Error: {d.get('exc','')}")))
            bus.subscribe("ui.status.set",
                          lambda _t, d: self.set_text(str(d.get("text", ""))))
            bus.subscribe("ui.cursor.pos",
                          lambda _t, d: self._ev_cursor(d))
        except Exception:
            pass

    # ------- internal callbacks -------------------------------------------
    def _ev_run_progress(self, _t: str, d: dict) -> None:  # type: ignore[override]
        v = float(d.get("percent", 0.0))
        msg = d.get("message") or f"Working… {int(v * 100)}%"
        if v == 0.0:
            self.start()
        if v >= 1.0:
            self.stop()
        self.set_text(msg)

    def _ev_cursor(self, d: dict) -> None:
        if "z" in d:
            self.set_coords((int(d["x"]), int(d["y"]), int(d["z"])))
        else:
            self.set_coords((int(d["x"]), int(d["y"])))
