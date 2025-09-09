# glitchlab/gui/controls.py
# -*- coding: utf-8 -*-
"""
Małe, wielokrotnego użytku kontrolki (Tkinter/ttk) do paneli filtrów.
Każda zwraca ramkę (Frame) i tk.Variable; automatycznie podpina on_change.
"""

from __future__ import annotations
from typing import Callable, Optional, Sequence, Tuple
import tkinter as tk
from tkinter import ttk

BG = "#16181c"
FG = "#e6e6e6"

def labeled_entry(parent: tk.Widget, label: str, textvar: tk.Variable,
                  on_change: Optional[Callable[[], None]] = None,
                  width: int = 12) -> tk.Frame:
    fr = tk.Frame(parent, bg=BG)
    tk.Label(fr, text=label, width=16, anchor="w", bg=BG, fg=FG).pack(side=tk.LEFT)
    e = ttk.Entry(fr, textvariable=textvar, width=width)
    e.pack(side=tk.LEFT, fill="x", expand=True)
    if on_change is not None:
        textvar.trace_add("write", lambda *a: on_change())
    return fr

def labeled_checkbox(parent: tk.Widget, label: str, boolvar: tk.BooleanVar,
                     on_change: Optional[Callable[[], None]] = None) -> tk.Frame:
    fr = tk.Frame(parent, bg=BG)
    tk.Label(fr, text=label, width=16, anchor="w", bg=BG, fg=FG).pack(side=tk.LEFT)
    cb = ttk.Checkbutton(fr, variable=boolvar)
    cb.pack(side=tk.LEFT)
    if on_change is not None:
        boolvar.trace_add("write", lambda *a: on_change())
    return fr

def labeled_combo(parent: tk.Widget, label: str, strvar: tk.StringVar,
                  values: Sequence[str],
                  on_change: Optional[Callable[[], None]] = None,
                  width: int = 12, readonly: bool = True) -> tk.Frame:
    fr = tk.Frame(parent, bg=BG)
    tk.Label(fr, text=label, width=16, anchor="w", bg=BG, fg=FG).pack(side=tk.LEFT)
    state = "readonly" if readonly else "normal"
    cb = ttk.Combobox(fr, textvariable=strvar, values=list(values), state=state, width=width)
    cb.pack(side=tk.LEFT, fill="x", expand=True)
    if on_change is not None:
        cb.bind("<<ComboboxSelected>>", lambda e: on_change())
        strvar.trace_add("write", lambda *a: on_change())
    return fr

def section(parent: tk.Widget, title: str) -> tk.Frame:
    fr = tk.LabelFrame(parent, text=title)
    fr.configure(bg=BG, fg=FG)
    return fr

def two_cols(parent: tk.Widget) -> Tuple[tk.Frame, tk.Frame]:
    left = tk.Frame(parent, bg=BG)
    right = tk.Frame(parent, bg=BG)
    left.pack(side=tk.LEFT, fill="both", expand=True, padx=(0, 6))
    right.pack(side=tk.LEFT, fill="both", expand=True, padx=(6, 0))
    return left, right
