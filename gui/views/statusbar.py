# glitchlab/gui/views/statusbar.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Optional, Tuple, Any


class StatusBar(ttk.Frame):
    """
    Lekki pasek stanu:

        [ ▓▓▓   progress  ]  Tekst stanu  ..................................  x=…, y=…

    Public API:
      - set_text(text: str) -> None
      - set_right(coords: tuple[int,int]|tuple[int,int,int]|str|None) -> None
      - set_progress(value: float|None, text: str|None = None) -> None
      - start()/stop()  (skrót do animowanego progressu)
      - bind_bus(bus)   (nasłuch: run.start/run.progress/run.done/run.error/ui.status.set/ui.cursor.pos)

    Zgodność:
      Obsługuje dwie konwencje payloadu „run.progress”:
        • v4: {"value": 0..1, "text": "msg"}           (zalecana)
        • legacy: {"percent": 0..1|0..100, "message": "msg"} (auto-wykrywane)
    """

    def __init__(self, master: tk.Misc, *, show_progress: bool = True) -> None:
        super().__init__(master)

        self._show_progress = bool(show_progress)

        self.columnconfigure(1, weight=1)

        # progress
        self._prog = ttk.Progressbar(self, mode="determinate", length=140, maximum=100)

        # labels
        self._lbl_left = ttk.Label(self, text="Ready", anchor="w")
        self._lbl_right = ttk.Label(self, text="", anchor="e")

        # layout
        c = 0
        if self._show_progress:
            self._prog.grid(row=0, column=c, padx=(6, 4), pady=2, sticky="w")
            c += 1
        self._lbl_left.grid(row=0, column=c, padx=4, pady=1, sticky="we")
        self._lbl_right.grid(row=0, column=c + 1, padx=6, pady=1, sticky="e")

        # state
        self._indeterminate = False

    # ───────────── Tekst po lewej ─────────────
    def set_text(self, text: str) -> None:
        try:
            self._lbl_left.config(text=text)
        except Exception:
            pass

    # ───────────── Prawa strona ─────────────
    def set_right(self, coords_or_text: Tuple[int, int] | Tuple[int, int, int] | str | None) -> None:
        """Ustaw prawy panel. Przyjmuje (x,y[,z]) lub zwykły string albo None (czyść)."""
        try:
            if coords_or_text is None:
                self._lbl_right.config(text="")
                return
            if isinstance(coords_or_text, str):
                self._lbl_right.config(text=coords_or_text)
                return
            # tuple współrzędnych
            if len(coords_or_text) == 2:
                x, y = coords_or_text  # type: ignore[misc]
                s = f"x={int(x)}  y={int(y)}"
            else:
                x, y, z = coords_or_text  # type: ignore[misc]
                s = f"x={int(x)}  y={int(y)}  z={int(z)}"
            self._lbl_right.config(text=s)
        except Exception:
            pass

    # alias zgodny z wcześniejszym kodem
    set_coords = set_right

    # ───────────── Progress ─────────────
    def _ensure_mode(self, indeterminate: bool) -> None:
        if indeterminate != self._indeterminate:
            self._indeterminate = indeterminate
            try:
                self._prog.config(mode="indeterminate" if indeterminate else "determinate")
            except Exception:
                pass

    def set_progress(self, value: Optional[float], text: Optional[str] = None) -> None:
        """
        value:
          - None      → brak paska / zatrzymaj animację
          - 0..1      → tryb determinate (0–100%)
          - inne      → tryb indeterminate (animacja)
        """
        if not self._show_progress:
            if text is not None:
                self.set_text(text)
            return

        try:
            if value is None:
                self._ensure_mode(False)
                self._prog.stop()
                self._prog["value"] = 0
            else:
                # if it's finite in [0,1] → determinate
                try:
                    v = float(value)
                    if 0.0 <= v <= 1.0:
                        self._ensure_mode(False)
                        self._prog.stop()
                        self._prog["value"] = v * 100.0
                    else:
                        self._ensure_mode(True)
                        self._prog.start(80)
                except Exception:
                    self._ensure_mode(True)
                    self._prog.start(80)

            if text is not None:
                self.set_text(text)
        except Exception:
            pass

    # krótkie aliasy
    def start(self) -> None:
        if self._show_progress:
            self._ensure_mode(True)
            try:
                self._prog.start(80)
            except Exception:
                pass

    def stop(self) -> None:
        if self._show_progress:
            self._ensure_mode(False)
            try:
                self._prog.stop()
                self._prog["value"] = 0
            except Exception:
                pass

    # ───────────── EventBus ─────────────
    def bind_bus(self, bus: Any) -> None:
        """
        Mapowanie zdarzeń:
          • run.start:    start + opcjonalny tekst {"text": "..."}
          • run.progress:
                v4:      {"value":0..1, "text":"..."}
                legacy:  {"percent":0..1|0..100, "message":"..."}
          • run.done:     stop + "Done"
          • run.error:    stop + "Error: …"
          • ui.status.set: {"text": "..."} → set_text
          • ui.cursor.pos: {"x":..,"y":..,"z"?:..} → set_right
        """
        if not (bus and hasattr(bus, "subscribe")):
            return
        try:
            bus.subscribe("run.start", lambda _t, d: self._on_run_start(d))
            bus.subscribe("run.progress", self._on_run_progress)
            bus.subscribe("run.done", lambda *_: (self.stop(), self.set_text("Done")))
            bus.subscribe("run.error", lambda _t, d: (self.stop(), self.set_text(f"Error: {d.get('error','')}")))
            bus.subscribe("ui.status.set", lambda _t, d: self.set_text(str(d.get("text", ""))))
            bus.subscribe("ui.cursor.pos", lambda _t, d: self._on_cursor(d))
        except Exception:
            pass

    # ───────────── Handlers ─────────────
    def _on_run_start(self, d: dict) -> None:
        try:
            self.start()
            txt = d.get("text")
            if txt:
                self.set_text(str(txt))
        except Exception:
            pass

    def _on_run_progress(self, _t: str, d: dict) -> None:  # type: ignore[override]
        # v4 schema
        text = d.get("text")
        value = d.get("value")

        # legacy compatibility
        if value is None and "percent" in d:
            p = float(d.get("percent", 0.0))
            value = p / 100.0 if p > 1.0 else p
        if text is None and "message" in d:
            text = d.get("message")

        # fallbackowy napis
        if text is None and isinstance(value, (int, float)):
            pct = max(0, min(100, int(float(value) * 100.0)))
            text = f"Working… {pct}%"

        self.set_progress(value if isinstance(value, (int, float)) else None, text=text)

    def _on_cursor(self, d: dict) -> None:
        try:
            x, y = int(d.get("x")), int(d.get("y"))
            if "z" in d:
                self.set_right((x, y, int(d.get("z"))))
            else:
                self.set_right((x, y))
        except Exception:
            self.set_right(None)
