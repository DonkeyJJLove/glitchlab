# -*- coding: utf-8 -*-
from __future__ import annotations
import tkinter as tk
from tkinter import ttk

class MaskPicker(ttk.Frame):
    """
    Wyszukiwalny wybór maski do parametru `mask_key`.
    API:
        MaskPicker(parent, get_keys: ()->list[str], on_change: (key:str)->None, value_var: tk.StringVar|None=None)
        .refresh()  # ponownie pobiera listę kluczy z get_keys()
    """
    def __init__(self, parent, get_keys, on_change, value_var: tk.StringVar | None = None):
        super().__init__(parent)
        self.get_keys = get_keys
        self.on_change = on_change
        self.value = value_var or tk.StringVar(value="")
        self._all: list[str] = []

        # UI
        row = ttk.Frame(self); row.pack(fill="x")
        ttk.Label(row, text="Mask:").pack(side="left")
        self.entry = ttk.Entry(row)
        self.entry.pack(side="left", fill="x", expand=True, padx=6)
        self.entry.bind("<KeyRelease>", self._on_filter)

        self.cmb = ttk.Combobox(row, state="readonly", values=[], textvariable=self.value)
        self.cmb.pack(side="left", fill="x", expand=True, padx=6)
        self.cmb.bind("<<ComboboxSelected>>", lambda e: self.on_change(self.value.get()))

        btns = ttk.Frame(self); btns.pack(fill="x", pady=(4,0))
        ttk.Button(btns, text="None", width=7, command=lambda: self._set_and_emit("")).pack(side="left")
        ttk.Button(btns, text="Edge", width=7, command=lambda: self._set_and_emit("edge")).pack(side="left", padx=(6,0))
        ttk.Button(btns, text="Reload", width=7, command=self.refresh).pack(side="left", padx=(6,0))

        self.refresh()

    def _set_and_emit(self, key: str) -> None:
        self.value.set(key)
        try:
            self.on_change(key)
        except Exception:
            pass

    def _on_filter(self, _e=None):
        q = self.entry.get().strip().lower()
        if not q:
            vals = self._all
        else:
            vals = [k for k in self._all if q in k.lower()]
        self.cmb["values"] = vals
        # nie zmieniaj wyboru gdy filtrujesz — użytkownik może wybrać strzałkami

    def refresh(self):
        try:
            self._all = list(sorted(set(self.get_keys() or [])))
        except Exception:
            self._all = []
        if "" not in self._all:
            self._all.insert(0, "")
        if "edge" not in self._all:
            self._all.insert(1, "edge")
        self.cmb["values"] = self._all
