# glitchlab/gui/views/__init__.py
# -*- coding: utf-8 -*-
"""
GlitchLab GUI Views (v4)

Pakiet `views` zawiera główne komponenty widokowe aplikacji:
  • BottomPanel – zintegrowany dolny panel (HUD/Graph/Mosaic/Diagnostics)
  • HUDView – kanoniczny 3-slotowy HUD oparty o MosaicSpec/Router

W tym pliku udostępniamy wygodne aliasy dla importów najwyższego poziomu:

    from glitchlab.gui.views import BottomPanel, HUDView
"""

from __future__ import annotations

from glitchlab.gui.views.bottom_panel import BottomPanel
from glitchlab.gui.views.hud import HUDView

__all__ = [
    "BottomPanel",
    "HUDView",
]
