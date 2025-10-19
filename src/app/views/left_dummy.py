# glitchlab/app/views/left_dummy.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class LeftDummy(ttk.Frame):
    """
    Wąski, zewnętrzny lewy dock – wyłącznie placeholder z pionowym napisem „in project”.
    Nie publikuje żadnych zdarzeń; służy do zajęcia miejsca po lewej stronie.
    """

    def __init__(self, master: tk.Misc, *, text: str = "│ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ ") -> None:
        super().__init__(master)
        # Użyj Canvas, żeby ładnie wycentrować napis (pionowo)
        self._canvas = tk.Canvas(self, width=28, highlightthickness=0, bg=self._bg())
        self._canvas.pack(fill="y", expand=False, side="left")
        self._draw_text(text)

        # dopasowanie tła przy zmianie motywu
        try:
            self.bind("<Configure>", lambda _e: self._sync_bg())
        except Exception:
            pass

    def _bg(self) -> str:
        try:
            style = ttk.Style()
            base = style.lookup("TFrame", "background")
            return base or "#1a1a1a"
        except Exception:
            return "#1a1a1a"

    def _fg(self) -> str:
        try:
            style = ttk.Style()
            # użyj domyślnego koloru tekstu etykiety
            fg = style.lookup("TLabel", "foreground")
            return fg or "#808080"
        except Exception:
            return "#808080"

    def _sync_bg(self) -> None:
        bg = self._bg()
        try:
            self.configure(style="LeftDummy.TFrame")
            self._canvas.configure(bg=bg)
        except Exception:
            pass

    def _draw_text(self, text: str) -> None:
        self._canvas.delete("all")
        bg, fg = self._bg(), self._fg()
        self._canvas.configure(bg=bg)
        # „pionowy” napis za pomocą \n
        vert = "\n".join(list(text))
        self._canvas.create_text(
            14, 10,  # środek „kolumny”
            text=vert,
            fill=fg,
            anchor="n",
            font=("TkDefaultFont", 9, "bold"),
            justify="center",
        )
