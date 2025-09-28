# glitchlab/gui/widgets/hud.py
# -*- coding: utf-8 -*-
"""
DEPRECATED shim (v4):

Ten moduł jest utrzymany wyłącznie dla zgodności wstecznej.
Docelowym, kanonicznym HUD-em w GlitchLab v4 jest:
    `glitchlab.gui.views.hud.HUDView`

Tutaj przekierowujemy symbol `Hud` -> `HUDView`, aby nie łamać istniejących importów:
    from glitchlab.gui.widgets.hud import Hud

W nowych miejscach importuj bezpośrednio:
    from glitchlab.gui.views.hud import HUDView
"""

from __future__ import annotations

import warnings

__all__ = ["Hud"]

try:  # preferowana ścieżka (v4)
    from glitchlab.gui.views.hud import HUDView as Hud  # type: ignore[F401]
    warnings.warn(
        "glitchlab.gui.widgets.hud is deprecated; use glitchlab.gui.views.hud.HUDView",
        DeprecationWarning,
        stacklevel=2,
    )
except Exception:
    # Awaryjny, minimalny stub — aby nie wywracać aplikacji, gdy views.hud nie jest dostępny.
    import tkinter as tk
    from tkinter import ttk
    from typing import Any, Dict

    class Hud(ttk.Frame):  # type: ignore[override]
        """
        Minimalny fallback HUD (pusty). Nie renderuje miniatur, jedynie pokazuje informację.
        API kompatybilne „best-effort”:
          - render_from_cache(ctx_or_cache)
          - set_cache(cache)
        """
        def __init__(self, master: tk.Misc):
            super().__init__(master)
            ttk.Label(
                self,
                text="(HUD unavailable — fallback stub)\n"
                     "Install/enable glitchlab.gui.views.hud.HUDView",
                foreground="#777",
                justify="center",
            ).pack(fill="both", expand=True, padx=12, pady=12)

        # Stare wywołania czasem przekazują ctx, czasem dict cache
        def render_from_cache(self, ctx_or_cache: Any) -> None:  # pragma: no cover
            # no-op
            pass

        def set_cache(self, _cache: Dict[str, Any]) -> None:  # pragma: no cover
            # no-op
            pass
2