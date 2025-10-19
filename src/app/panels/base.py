# glitchlab/app/panels/base.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any, Dict, List, Optional, Callable

try:
    from glitchlab.gui.panel_base import PanelBase, PanelContext  # type: ignore
except Exception:  # awaryjny fallback
    from dataclasses import dataclass, field

    @dataclass
    class PanelContext:
        filter_name: str = ""
        defaults: Dict[str, Any] = field(default_factory=dict)
        params: Dict[str, Any] = field(default_factory=dict)
        on_change: Optional[Callable[[Dict[str, Any]], None]] = None
        cache_ref: Optional[Dict[str, Any]] = None
        get_mask_keys: Optional[Callable[[], List[str]]] = None

        def mask_keys(self) -> List[str]:
            if callable(self.get_mask_keys):
                try:
                    return list(self.get_mask_keys())
                except Exception:
                    pass
            if isinstance(self.cache_ref, dict):
                v = self.cache_ref.get("cfg/masks/keys")
                if isinstance(v, list):
                    return list(v)
            return []

        def emit(self, params: Optional[Dict[str, Any]] = None) -> None:
            if callable(self.on_change):
                try:
                    self.on_change(dict(params or {}))
                except Exception:
                    pass

    class PanelBase(ttk.Frame):
        def __init__(self, parent: tk.Widget, ctx: PanelContext | None = None, **kw):
            super().__init__(parent, **kw)
            self.ctx = ctx or PanelContext()


# Alias zgodnościowy
BasicPanel = PanelBase


# ---------------- API Loader ----------------
def register_panel(_name: str, _cls: type) -> None:
    """Zachowany dla kompatybilności – obecny loader paneli nie wymaga rejestracji."""
    return None


def get_panel(name: str):
    """Zwraca klasę panelu dla filtra."""
    try:
        from glitchlab.app.panel_loader import get_panel_class
        return get_panel_class(name)
    except Exception:
        return None


def list_registered_panels():
    pass


def list_panels() -> List[str]:
    """Lista zarejestrowanych paneli."""
    try:
        from glitchlab.app.panel_loader import list_available_panels
        return list_registered_panels()
    except Exception:
        return []
