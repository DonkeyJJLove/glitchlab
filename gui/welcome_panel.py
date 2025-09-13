# glitchlab/gui/welcome_panel.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from tkinter import ttk
from .panel_base import BasicPanel, PanelContext

class WelcomePanel(BasicPanel):
    def __init__(self, master, ctx: PanelContext | None = None, **kw):
        super().__init__(master, ctx=ctx, **kw)
        msg = (
            "Witaj w GlitchLab GUI\n\n"
            "- Otwórz obraz (File → Open image…)\n"
            "- Wybierz filtr i parametry, kliknij Apply\n"
            "- Dodaj krok do presetów w Preset Manager\n"
            "- HUD na dole pokazuje diagnostykę"
        )
        ttk.Label(self, text=msg, justify="left").pack(padx=8, pady=8, anchor="w")
