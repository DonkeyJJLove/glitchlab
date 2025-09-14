# glitchlab/gui/widgets/diag_console.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import tkinter as tk
from tkinter import ttk, filedialog
from typing import Any, Dict, Optional
import datetime as _dt

_LEVEL_COLORS = {
    "INFO":  ("#e0e0e0", "#202020"),
    "OK":    ("#d9fdd3", "#103010"),
    "WARN":  ("#fff4cc", "#403000"),
    "ERROR": ("#ffd7d7", "#401010"),
    "DEBUG": ("#e8f0fe", "#102040"),
}

class DiagConsole(ttk.Frame):
    """
    Prosta konsola diagnostyczna.
    API:
      - log(msg, level="INFO")
      - attach_bus(bus) -> subskrybuje 'diag.log' (payload: {msg, level})
    """
    def __init__(self, master: tk.Misc, **kw) -> None:
        super().__init__(master, **kw)
        self._build()

    def _build(self) -> None:
        bar = ttk.Frame(self); bar.pack(fill="x", padx=6, pady=6)
        ttk.Label(bar, text="Diagnostics log", font=("", 10, "bold")).pack(side="left")
        ttk.Button(bar, text="Clear", command=self.clear).pack(side="right")
        ttk.Button(bar, text="Save…", command=self._save).pack(side="right", padx=(0,6))
        ttk.Button(bar, text="Copy", command=self._copy).pack(side="right", padx=(0,6))

        box = ttk.Frame(self); box.pack(fill="both", expand=True, padx=6, pady=(0,6))
        self.txt = tk.Text(box, height=12, wrap="none")
        sx = ttk.Scrollbar(box, orient="horizontal", command=self.txt.xview)
        sy = ttk.Scrollbar(box, orient="vertical",   command=self.txt.yview)
        self.txt.configure(xscrollcommand=sx.set, yscrollcommand=sy.set)
        self.txt.grid(row=0, column=0, sticky="nsew")
        sy.grid(row=0, column=1, sticky="ns")
        sx.grid(row=1, column=0, sticky="ew")
        box.rowconfigure(0, weight=1)
        box.columnconfigure(0, weight=1)

        # tagi kolorów
        for level, (bg, fg) in _LEVEL_COLORS.items():
            self.txt.tag_configure(level, background=bg, foreground=fg)

    # --- public ---
    def attach_bus(self, bus: Any) -> None:
        try:
            bus.subscribe("diag.log", lambda _t, p: self.log(str((p or {}).get("msg","")),
                                                             str((p or {}).get("level","INFO")).upper()))
        except Exception:
            pass

    def log(self, msg: str, level: str = "INFO") -> None:
        ts = _dt.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        lvl = (level or "INFO").upper()
        if lvl not in _LEVEL_COLORS:
            lvl = "INFO"
        line = f"[{ts}] {lvl}: {msg}\n"
        try:
            self.txt.insert("end", line, (lvl,))
            self.txt.see("end")
        except Exception:
            pass

    def clear(self) -> None:
        try:
            self.txt.delete("1.0", "end")
        except Exception:
            pass

    def _copy(self) -> None:
        try:
            data = self.txt.get("1.0", "end-1c")
            self.clipboard_clear()
            self.clipboard_append(data)
        except Exception:
            pass

    def _save(self) -> None:
        fn = filedialog.asksaveasfilename(
            title="Save diagnostics",
            defaultextension=".log",
            filetypes=[("Log files", "*.log;*.txt"), ("All files", "*.*")],
        )
        if not fn:
            return
        try:
            data = self.txt.get("1.0", "end-1c")
            with open(fn, "w", encoding="utf-8") as f:
                f.write(data)
        except Exception:
            pass
