# glitchlab/gui/panel_base.py
# -*- coding: utf-8 -*-
"""
Bazowy kontrakt mini-UI dla filtrów + rejestr paneli.
App będzie wołać fabrykę (panel_loader.get_panel_for_filter), która:
- zwróci panel dedykowany (jeśli zarejestrowany),
- albo panel generyczny zbudowany na bazie schema/sygnatury filtra.

Każdy panel:
- implementuje build(parent) -> Frame
- implementuje get_params() -> dict
- opcjonalnie set_params(dict), validate()
- ma właściwość .on_change (callable), wywoływaną na zmianę któregokolwiek pola
"""

from __future__ import annotations
from typing import Dict, Any, Optional, Callable
import tkinter as tk
from dataclasses import dataclass


# ———————————————————————————————————————————————————————————————
# Lekki kontekst dla paneli (opcjonalny; App może wstrzyknąć)
# ———————————————————————————————————————————————————————————————
@dataclass
class PanelContext:
    """
    Dane dynamiczne, które panele mogą potrzebować do wypełnienia pól:
    - listy masek dostępnych w ctx (np. do dropdownu mask_key)
    Wstrzykiwane przez App: panel.set_context_provider(lambda: PanelContext(...))
    """
    mask_keys: list[str]


# ———————————————————————————————————————————————————————————————
# Kontrakt panelu
# ———————————————————————————————————————————————————————————————
class FilterPanel:
    FILTER_NAME: str = ""  # nazwa filtra z registry

    def __init__(self) -> None:
        self._root: Optional[tk.Frame] = None
        self.on_change: Optional[Callable[[], None]] = None
        self._context_provider: Optional[Callable[[], PanelContext]] = None

    # App może wstrzyknąć provider kontekstu (np. aktualne maski)
    def set_context_provider(self, provider: Callable[[], PanelContext]) -> None:
        self._context_provider = provider

    def get_context(self) -> PanelContext:
        if self._context_provider is None:
            return PanelContext(mask_keys=[])
        return self._context_provider()

    def build(self, parent: tk.Widget) -> tk.Frame:
        raise NotImplementedError

    def get_params(self) -> Dict[str, Any]:
        return {}

    def set_params(self, params: Dict[str, Any]) -> None:
        pass

    def validate(self) -> None:
        pass


# ———————————————————————————————————————————————————————————————
# Rejestr paneli
# ———————————————————————————————————————————————————————————————
_PANELS: Dict[str, type[FilterPanel]] = {}


def register_panel(panel_cls: type[FilterPanel]) -> None:
    name = getattr(panel_cls, "FILTER_NAME", "")
    if not name:
        raise ValueError("Panel class must define FILTER_NAME")
    if name in _PANELS:
        raise KeyError(f"Panel '{name}' already registered")
    _PANELS[name] = panel_cls


def get_panel_class(filter_name: str) -> Optional[type[FilterPanel]]:
    return _PANELS.get(filter_name)


# ———————————————————————————————————————————————————————————————
# Koercja wartości (z form na typy filtra) + clamp
# ———————————————————————————————————————————————————————————————
def coerce_value(val: Any, ty: str | None = None, vmin: Any = None, vmax: Any = None) -> Any:
    """
    Koercja prostych typów wg schema: 'int' | 'float' | 'bool' | 'str'.
    Jeśli ty=None → heurystyka: spróbuj int→float→bool→str.
    """

    def _clamp_num(x):
        if vmin is not None:
            x = max(x, vmin)
        if vmax is not None:
            x = min(x, vmax)
        return x

    if ty == "int":
        x = int(float(val))
        return _clamp_num(x)
    if ty == "float":
        x = float(val)
        return _clamp_num(x)
    if ty == "bool":
        if isinstance(val, str):
            v = val.strip().lower()
            x = v in ("1", "true", "yes", "on")
        else:
            x = bool(val)
        return x
    if ty == "str":
        return str(val)

    # heurystyka
    s = str(val).strip()
    try:
        x = int(s)
        return _clamp_num(x)
    except Exception:
        pass
    try:
        x = float(s)
        return _clamp_num(x)
    except Exception:
        pass
    if s.lower() in ("1", "true", "yes", "on", "0", "false", "no", "off"):
        return (s.lower() in ("1", "true", "yes", "on"))
    return s
