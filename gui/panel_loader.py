from __future__ import annotations
from typing import Type
from .panel_base import PanelBase
# Try registry from panels
try:
    from .panels.base import get_panel as _get, list_panels as _list
except Exception:  # pragma: no cover
    def _get(name): return None
    def _list(): return []

from .generic_form_panel import GenericFormPanel

def get_panel_class(name: str) -> Type[PanelBase]:
    cls = _get(name) if name else None
    return cls or GenericFormPanel

def list_panels() -> list[str]:
    return _list()
