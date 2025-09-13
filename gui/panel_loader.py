# glitchlab/gui/panel_loader.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import importlib, inspect, sys
from tkinter import ttk


def get_panel_class(filter_name: str) -> type[ttk.Frame]:
    """
    Próbuje znaleźć panel dedykowany: glitchlab.gui.panels.panel_<filter_name>.
    W braku — zwraca GenericFormPanel.
    """
    name = (filter_name or "").strip().lower()
    if not name:
        try:
            from .welcome_panel import WelcomePanel
            return WelcomePanel  # type: ignore
        except Exception:
            return ttk.Frame  # awaryjnie pusta ramka

    # 1) dedykowany panel
    modname = f"glitchlab.gui.panels.panel_{name}"
    try:
        mod = importlib.import_module(modname)
        for k in dir(mod):
            obj = getattr(mod, k, None)
            if inspect.isclass(obj) and issubclass(obj, ttk.Frame):
                return obj  # type: ignore
    except Exception as e:
        print(f"[panel_loader] fallback for '{name}': {e}", file=sys.stderr)

    # 2) generic
    try:
        from .generic_form_panel import GenericFormPanel
        return GenericFormPanel  # type: ignore
    except Exception:
        return ttk.Frame  # last resort


def list_panels() -> list[str]:
    return ["generic_form", "welcome"]
