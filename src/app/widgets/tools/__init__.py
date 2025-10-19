# glitchlab/app/widgets/tools/__init__.py
# -*- coding: utf-8 -*-
"""
Registry narzędzi canvasa (bez twardych zależności na wszystkie moduły).

Użycie:
    from glitchlab.app.widgets.tools import get_tool, available, DEFAULT_TOOL
    ToolCls = get_tool("rect")
    tool = ToolCls(ctx)

Konwencja nazw:
    "view"     — przeglądanie (pan/zoom)      [opcjonalne; no-op jeśli viewer robi to sam]
    "rect"     — zaznaczenie prostokątne
    "ellipse"  — zaznaczenie eliptyczne
    "brush"    — malowanie maski              [opcjonalne]
    "eraser"   — kasowanie maski              [opcjonalne]
    "pipette"  — pipeta/probe                 [opcjonalne]
    "measure"  — linia pomiarowa              [opcjonalne]
    "move"     — przesuwanie aktywnej warstwy [opcjonalne]
"""

from __future__ import annotations

from typing import Dict, Type, Any

# ── opcjonalny kontrakt bazowy (na wypadek braku base.py) ────────────────────
try:
    from .base import ToolBase, ToolEventContext  # type: ignore
except Exception:  # pragma: no cover
    class ToolBase:  # type: ignore
        name: str = "base"
        def __init__(self, ctx: Any): self.ctx = ctx
        def on_activate(self, opts=None): ...
        def on_deactivate(self): ...
        def on_mouse_down(self, e): ...
        def on_mouse_move(self, e): ...
        def on_mouse_up(self, e): ...
        def on_key(self, e): ...
        def on_wheel(self, e): ...
        def draw_overlay(self, canvas): ...

    class ToolEventContext:  # type: ignore
        pass


# ── bezpieczne importy narzędzi ──────────────────────────────────────────────
_TOOL_CLASSES: Dict[str, Type[ToolBase]] = {}
_ERRORS: Dict[str, str] = {}


def _try_register(key: str, modpath: str, clsname: str) -> None:
    """Importuje klasę narzędzia i rejestruje ją pod kluczem, albo zapamiętuje błąd."""
    global _TOOL_CLASSES, _ERRORS
    try:
        mod = __import__(f"{__name__}.{modpath}", fromlist=[clsname])
        cls = getattr(mod, clsname)
        if not isinstance(cls, type):
            raise TypeError(f"{modpath}.{clsname} is not a class")
        _TOOL_CLASSES[key] = cls  # type: ignore[assignment]
    except Exception as ex:  # pragma: no cover
        _ERRORS[key] = f"{type(ex).__name__}: {ex}"


# Priorytet: najpierw narzędzia, które mamy w repo (lub wkrótce będą)
_try_register("rect", "tool_rect_select", "RectSelectTool")
_try_register("ellipse", "tool_ellipse_select", "EllipseSelectTool")

# Opcjonalne — mogą nie być jeszcze dostępne w repo
_try_register("view", "tool_view", "ViewTool")
_try_register("brush", "tool_brush", "BrushTool")
_try_register("eraser", "tool_eraser", "EraserTool")
_try_register("pipette", "tool_pipette", "PipetteTool")
_try_register("measure", "tool_measure", "MeasureTool")
_try_register("move", "tool_move_layer", "MoveLayerTool")

# ── fallbacki / domyślne ─────────────────────────────────────────────────────
DEFAULT_TOOL = (
    "rect" if "rect" in _TOOL_CLASSES else
    ("view" if "view" in _TOOL_CLASSES else next(iter(_TOOL_CLASSES), "rect"))
)

# „view” jako alias do DEFAULT_TOOL (jeśli viewer i tak obsługuje pan/zoom)
if "view" not in _TOOL_CLASSES:
    class _ViewNoop(ToolBase):  # type: ignore
        name = "view"
        def on_activate(self, opts=None): ...
        def draw_overlay(self, canvas): ...
    _TOOL_CLASSES["view"] = _ViewNoop  # type: ignore[assignment]


# ── API publiczne ────────────────────────────────────────────────────────────
def available() -> list[str]:
    """Lista dostępnych (poprawnie załadowanych) narzędzi."""
    return sorted(_TOOL_CLASSES.keys())


def has(name: str) -> bool:
    """Czy narzędzie o podanej nazwie jest dostępne?"""
    return str(name) in _TOOL_CLASSES


def get_tool(name: str) -> Type[ToolBase]:
    """
    Pobierz klasę narzędzia po nazwie; gdy brak — zwróć narzędzie domyślne.
    """
    key = str(name)
    return _TOOL_CLASSES.get(key, _TOOL_CLASSES[DEFAULT_TOOL])


def debug_errors() -> dict[str, str]:
    """Zwrotka błędów importu dla narzędzi opcjonalnych (do diagnostyki)."""
    return dict(_ERRORS)


__all__ = [
    "ToolBase", "ToolEventContext",
    "available", "has", "get_tool", "DEFAULT_TOOL", "debug_errors",
]
