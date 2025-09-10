# gui/panels/registry.py
from __future__ import annotations
from typing import Dict, Type
from .base import PanelBase

PANELS: Dict[str, Type[PanelBase]] = {}

def register_panel(filter_name: str, panel_cls: Type[PanelBase]):
    PANELS[filter_name] = panel_cls

def get_panel(filter_name: str) -> Type[PanelBase] | None:
    return PANELS.get(filter_name)
