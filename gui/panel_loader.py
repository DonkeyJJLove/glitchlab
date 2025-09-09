# glitchlab/gui/panel_loader.py
# -*- coding: utf-8 -*-
"""
Fabryka paneli: dedykowany panel dla filtra (jeśli zarejestrowany),
inaczej GenericFormPanel zbudowany na bazie schema/sygnatury filtra.
"""

from __future__ import annotations
from typing import Optional
from glitchlab.gui.panel_base import get_panel_class, FilterPanel, PanelContext
from glitchlab.gui.generic_form_panel import GenericFormPanel

# WAŻNE: import pakietu z panelami, aby wywołać ich rejestrację.
# W kolejnych krokach dołożymy moduły paneli dedykowanych i zaktualizujemy __init__.
try:
    import glitchlab.gui.panels  # noqa: F401
except Exception:
    # brak pakietu panels → OK, będziemy używać GenericFormPanel
    pass


def get_panel_for_filter(filter_name: str) -> FilterPanel:
    PanelCls = get_panel_class(filter_name)
    if PanelCls is not None:
        return PanelCls()
    return GenericFormPanel(filter_name)
