# glitchlab/gui/panels/__init__.py
# -*- coding: utf-8 -*-
"""
Pakiet paneli GUI — auto-importuje wszystkie moduły paneli:
  • panel_<name>.py
  • <name>_panel.py
Dzięki temu stare i nowe nazwy są dostępne bez ręcznej edycji.
"""
from __future__ import annotations
import importlib, pkgutil

# re-eksport bazowych typów (opcjonalnie)
try:
    from .base import PanelBase, PanelContext  # type: ignore
except Exception:
    PanelBase = object  # type: ignore
    class PanelContext:  # type: ignore
        def __init__(self, **kw): self.__dict__.update(kw)

__all__ = ["PanelBase", "PanelContext"]

# auto-import
def _auto_import_panels() -> None:
    try:
        pkg = importlib.import_module(__name__)
        if hasattr(pkg, "__path__"):
            for _, mod_name, ispkg in pkgutil.iter_modules(pkg.__path__):
                if ispkg:
                    continue
                if mod_name.startswith("panel_") or mod_name.endswith("_panel"):
                    importlib.import_module(f"{__name__}.{mod_name}")
    except Exception:
        # miękko; brak paniki przy środowiskowych różnicach
        pass

_auto_import_panels()