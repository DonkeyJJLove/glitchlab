# glitchlab/gui/panel_loader.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib
import inspect
from typing import Optional, Type

try:
    import tkinter as tk
    from tkinter import ttk
except Exception:  # pragma: no cover
    tk = None
    ttk = None

# Opcjonalnie używamy baz, jeśli są dostępne (nie są wymagane do detekcji)
try:
    from .panel_base import PanelBase, PanelContext  # type: ignore
except Exception:  # pragma: no cover
    PanelBase = object  # type: ignore
    class PanelContext:  # type: ignore
        def __init__(self, **kw): self.__dict__.update(kw)


def _is_panel_class(obj) -> bool:
    """Czy to wygląda na klasę panelu Tk? (subklasa ttk.Frame lub PanelBase)."""
    try:
        if inspect.isclass(obj):
            if ttk is not None and issubclass(obj, getattr(ttk, "Frame")):
                return True
            if PanelBase is not object and issubclass(obj, PanelBase):  # type: ignore[arg-type]
                return True
    except Exception:
        pass
    return False


def get_panel_class(filter_name: str) -> Optional[Type]:
    """
    Znajdź klasę panelu dla filtra:
      1) próbuje zaimportować moduł 'glitchlab.gui.panels.panel_<filter_name>'
      2) jeśli brak, próbuje '<filter_name>_panel'
      3) jeśli w module jest symbol 'Panel' będący klasą, zwraca go,
      4) w przeciwnym razie szuka klasy kończącej się na 'Panel',
      5) jeśli brak – zwraca None (caller użyje GenericFormPanel/ParamForm).
    """
    candidates = [
        f"glitchlab.gui.panels.panel_{filter_name}",   # nowa konwencja
        f"glitchlab.gui.panels.{filter_name}_panel",   # stara konwencja
    ]

    for mod_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue

        # Priorytet: symbol eksportowany jawnie jako `Panel`
        cand = getattr(mod, "Panel", None)
        if cand and _is_panel_class(cand):
            return cand

        # Heurystyka: pierwsza klasa *Panel w module
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if obj.__module__ == mod.__name__ and obj.__name__.lower().endswith("panel") and _is_panel_class(obj):
                return obj

    return None


def instantiate_panel(parent, filter_name: str, ctx: Optional[PanelContext] = None):
    """
    Zwraca *instancję* panelu lub None, jeśli nie znaleziono dedykowanego.
    """
    Cls = get_panel_class(filter_name)
    if Cls is None:
        return None
    try:
        return Cls(parent, ctx=ctx)
    except TypeError:
        # część starszych paneli mogła nie przyjmować ctx
        return Cls(parent)

