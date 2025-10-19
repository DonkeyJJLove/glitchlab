# glitchlab/app/widgets/diag_console.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Any, Callable, Dict, Optional
from datetime import datetime


_LogCb = Callable[[str, Dict[str, Any]], None]


class DiagConsole(ttk.Frame):
    """
    Lekka konsola diagnostyczna do wbudowania w GUI (np. w BottomPanel -> zakładka 'Tech/Diagnostics').

    Cechy:
      • Własny bufor Text z monospaced fontem, opcjonalny filtr poziomu logów.
      • Metody API:
          - attach_bus(services) / detach_bus() – subskrypcja EventBus (diag.log, run.*),
          - log(level, msg) – dopisz linię,
          - clear() – wyczyść,
          - get_text() – pobierz cały tekst.
      • Bezpieczne dla wątku UI – wpisy trafiają przez .after().

    Oczekiwane zdarzenia na Bus:
      - diag.log {level, msg}
      - run.progress {stage, percent?}
      - run.done {…}
      - run.error {error}
      - (opcjonalnie) ui.status.set {text}
    """

    LEVELS = ("DEBUG", "OK", "WARN", "ERROR")

    def __init__(self, master: tk.Misc, *, height: int = 10) -> None:
        super().__init__(master)
        self._bus: Optional[Any] = None
        self._subscribed: Dict[str, _LogCb] = {}
        self._level = tk.StringVar(value="DEBUG")

        # Toolbar
        bar = ttk.Frame(self)
        bar.pack(side="top", fill="x", padx=6, pady=(6, 2))

        ttk.Label(bar, text="Level:").pack(side="left")
        self._level_combo = ttk.Combobox(
            bar, width=8, state="readonly", values=self.LEVELS, textvariable=self._level
        )
        self._level_combo.pack(side="left", padx=(4, 8))

        ttk.Button(bar, text="Clear", command=self.clear).pack(side="left")
        ttk.Button(bar, text="Copy", command=self._copy_all).pack(side="left", padx=(4, 0))
        ttk.Button(bar, text="Save…", command=self._save_to_file).pack(side="left", padx=(4, 0))

        # Text area
        self._text = tk.Text(
            self,
            height=height,
            wrap="word",
            bg="#141414",
            fg="#e6e6e6",
            insertbackground="#e6e6e6",
            undo=False,
        )
        self._text.pack(side="top", fill="both", expand=True, padx=6, pady=(0, 6))

        # Tagowanie poziomów
        self._text.tag_config("DEBUG", foreground="#8aa2ff")
        self._text.tag_config("OK", foreground="#7fd07f")
        self._text.tag_config("WARN", foreground="#e4c26b")
        self._text.tag_config("ERROR", foreground="#ff6b6b")

        # Monospace font (jeśli dostępny)
        try:
            self._text.configure(font=("Consolas", 10))
        except Exception:
            pass

    # --------------------------------------------------------------------- API

    def attach_bus(self, bus: Any) -> None:
        """Podepnij EventBus i zasubskrybuj standardowe tematy."""
        self.detach_bus()
        self._bus = bus

        def sub(topic: str, cb: _LogCb) -> None:
            try:
                bus.subscribe(topic, cb)
                self._subscribed[topic] = cb
            except Exception:
                # Bus soft-fallback (np. atrapa)
                setattr(bus, f"_cb_{topic.replace('.', '_')}", cb)
                self._subscribed[topic] = cb

        # Standardowe tematy
        sub("diag.log", self._on_diag_log)
        sub("run.progress", self._on_run_progress)
        sub("run.done", self._on_run_done)
        sub("run.error", self._on_run_error)
        sub("ui.status.set", self._on_ui_status)

        self.log("OK", "DiagConsole attached to services")

    def detach_bus(self) -> None:
        """Odłącz EventBus (jeśli wspiera odsubskrybowanie)."""
        if self._bus is None:
            return
        # Jeśli EventBus ma własne API do odsubskrybowania – użyj go tutaj.
        # Obecny EventBus w projekcie nie eksponuje unsubscribe, więc tylko czyścimy mapę.
        self._subscribed.clear()
        self._bus = None

    def set_level(self, level: str) -> None:
        level = (level or "").upper()
        if level in self.LEVELS:
            self._level.set(level)

    def log(self, level: str, msg: str) -> None:
        """Dopisz linię do konsoli (bezpośrednio, z pominięciem Bus)."""
        self._append_line(level.upper(), msg)

    def clear(self) -> None:
        try:
            self._text.delete("1.0", "end")
        except Exception:
            pass

    def get_text(self) -> str:
        try:
            return self._text.get("1.0", "end-1c")
        except Exception:
            return ""

    # --------------------------------------------------------------- Bus Cbs --

    def _on_diag_log(self, _topic: str, payload: Dict[str, Any]) -> None:
        lvl = str(payload.get("level", "DEBUG")).upper()
        msg = str(payload.get("msg", ""))
        self._append_line(lvl, msg)

    def _on_run_progress(self, _topic: str, payload: Dict[str, Any]) -> None:
        stage = payload.get("stage")
        percent = payload.get("percent")
        if percent is None and "value" in payload:
            percent = payload.get("value")
        if percent is None:
            text = f"progress: stage={stage}"
        else:
            try:
                pct = float(percent) * (100.0 if float(percent) <= 1.0 else 1.0)
                text = f"progress: stage={stage} {pct:.1f}%"
            except Exception:
                text = f"progress: stage={stage} {percent}"
        self._append_line("DEBUG", text)

    def _on_run_done(self, _topic: str, _payload: Dict[str, Any]) -> None:
        self._append_line("OK", "run.done")

    def _on_run_error(self, _topic: str, payload: Dict[str, Any]) -> None:
        err = payload.get("error")
        self._append_line("ERROR", f"run.error: {err}")

    def _on_ui_status(self, _topic: str, payload: Dict[str, Any]) -> None:
        txt = payload.get("text", "")
        if txt:
            self._append_line("DEBUG", f"status: {txt}")

    # ---------------------------------------------------------- UI helpers ----

    def _append_line(self, level: str, msg: str) -> None:
        """Wstaw linię do Text w bezpieczny dla wątku sposób."""
        level = (level or "DEBUG").upper()
        if level not in self.LEVELS:
            level = "DEBUG"

        # Filtr poziomu
        if self._should_filter_out(level):
            return

        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {level:5s} | {msg}\n"

        def _do():
            try:
                self._text.insert("end", line, level)
                self._text.see("end")
            except Exception:
                pass

        # Always marshal to UI thread
        try:
            self.after(0, _do)
        except Exception:
            # W razie braku pętli głównej – próbuj bezpośrednio
            _do()

    def _should_filter_out(self, level: str) -> bool:
        """Zwróć True jeśli dany level ma być ukryty zgodnie z bieżącym filtrem."""
        # Priorytet: DEBUG < OK < WARN < ERROR
        order = {lv: i for i, lv in enumerate(self.LEVELS)}
        try:
            current = self._level.get()
        except Exception:
            current = "DEBUG"
        return order.get(level, 0) < order.get(current, 0)

    def _copy_all(self) -> None:
        try:
            text = self.get_text()
            if not text:
                return
            self.clipboard_clear()
            self.clipboard_append(text)
        except Exception:
            pass

    def _save_to_file(self) -> None:
        try:
            path = filedialog.asksaveasfilename(
                title="Save diagnostics log",
                defaultextension=".log",
                filetypes=[("Log file", "*.log"), ("Text", "*.txt"), ("All files", "*.*")],
            )
            if not path:
                return
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.get_text())
        except Exception as ex:
            try:
                messagebox.showerror("Save log", str(ex))
            except Exception:
                pass
