# glitchlab/gui/log_window.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import tkinter as tk
from tkinter import ttk


class LogWindow(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("GlitchLab — Log")
        self.geometry("720x240")
        self.text = tk.Text(self, wrap="none", bg="#111", fg="#ddd")
        self.text.pack(fill="both", expand=True)
        self.protocol("WM_DELETE_WINDOW", self.withdraw)

    def log(self, msg: str):
        self.text.insert("end", msg.rstrip() + "\n")
        self.text.see("end")
