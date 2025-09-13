"""
---
version: 3
kind: module
id: "gui-views-statusbar"
created_at: "2025-09-13"
name: "glitchlab.gui.views.statusbar"
author: "GlitchLab v3"
role: "Statusbar: tekst stanu + indykator postępu + prosty log (Tk/ttk)"
description: >
  Lekki pasek statusu do aplikacji GlitchLab. Zapewnia metody ustawiania tekstu,
  start/stop animowanego progressbara oraz opcjonalny log liniowy. Może (opcjonalnie)
  subskrybować zdarzenia run.* z EventBus, ale nie zawiera twardej zależności.

inputs:
  set_text(text):   {type: "str"}
  start():          {type: "None", note: "uruchamia indeterminate progress"}
  stop():           {type: "None", note: "zatrzymuje progress"}
  log(message):     {type: "str", optional: true}

events_optional:
  run.progress: {payload: {percent: float, message?: str}}
  run.done:     {payload: {result?: any}}
  run.error:    {payload: {exc: str}}

outputs:
  ui_feedback: {type: "visual", note: "uaktualniony status w GUI"}

interfaces:
  exports:
    - "StatusBar(master)"
    - "StatusBar.set_text(text:str) -> None"
    - "StatusBar.start() / StatusBar.stop() -> None"
    - "StatusBar.log(msg:str) -> None"
    - "StatusBar.bind_bus(bus) -> None  # opcjonalne podpięcie run.*"

depends_on: ["tkinter/ttk","typing","time? (opcjonalnie)"]
used_by: ["glitchlab.gui.app","glitchlab.gui.app_shell","glitchlab.gui.views.viewport","glitchlab.gui.views.notebook"]

policy:
  deterministic: true
  ui_thread_only: true
constraints:
  - "brak I/O; brak zależności na core; opcjonalna integracja z EventBus"
license: "Proprietary"
---
"""
# glitchlab/gui/views/statusbar.py
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Optional


class StatusBar(ttk.Frame):
    """
    Prosty pasek statusu z indeterminate Progressbar.
    Public API używane przez AppShell:
      - set_text(str)
      - start() / stop()
      - log(msg)  (no-op albo później: otwórz panel logów)
    """
    def __init__(self, master, *, show_progress: bool = True) -> None:
        super().__init__(master)
        self._show_progress = bool(show_progress)
        self._prog = ttk.Progressbar(self, mode="indeterminate", length=140)
        self._lbl = ttk.Label(self, text="Ready")

        if self._show_progress:
            self._prog.pack(side="left", padx=(6, 4), pady=2)
        self._lbl.pack(side="left", padx=4)

        # miejsce na drobne telemetry (opcjonalnie)
        self._right = ttk.Label(self, text="")
        self._right.pack(side="right", padx=6)

    def set_text(self, s: str) -> None:
        try:
            self._lbl.config(text=s)
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

    def set_right(self, text: str) -> None:
        try:
            self._right.config(text=text)
        except Exception:
            pass

    def log(self, _msg: str) -> None:
        # docelowo: integracja z panelem logów lub osobnym oknem
        pass
