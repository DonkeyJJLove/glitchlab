# glitchlab/app/views/__init__.py
# -*- coding: utf-8 -*-
"""
GlitchLab GTX Views (v4)

Pakiet `views` zawiera główne komponenty widokowe aplikacji:
  • BottomPanel – zintegrowany dolny panel (HUD/Graph/Mosaic/Diagnostics)
  • HUDView – kanoniczny 3-slotowy HUD oparty o MosaicSpec/Router

W tym pliku udostępniamy wygodne aliasy dla importów najwyższego poziomu:

    from glitchlab.app.views import BottomPanel, HUDView
"""

from __future__ import annotations

from glitchlab.app.views.bottom_panel import BottomPanel
from glitchlab.app.views.hud import HUDView

__all__ = [
    "BottomPanel",
    "HUDView",
]
