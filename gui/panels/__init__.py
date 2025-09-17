# glitchlab/gui/panels/__init__.py
# -*- coding: utf-8 -*-
"""
Pakiet paneli GUI.

Funkcjonalność:
  • Auto-import modułów paneli: `panel_<name>.py` i `<name>_panel.py`
  • Rejestr paneli (PANEL_REGISTRY) tworzony dynamicznie na podstawie atrybutu `Panel` w module
  • Przyjazne API: get_panel(name) / list_panels()

Konwencja modułu panela:
  - Moduł powinien eksportować symbol `Panel` (klasa/wywoływalny widget).
  - Opcjonalnie może eksportować `PANEL_NAME` lub `NAME` (string) jako kanoniczną nazwę.

Przykład:
  from glitchlab.gui.panels import get_panel
  SpectralPanel = get_panel("spectral_shaper")
  if SpectralPanel:
      widget = SpectralPanel(parent, ctx=...)
"""

from __future__ import annotations

import importlib
import pkgutil
import sys
from types import ModuleType
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Type, Union

# ──────────────────────────────────────────────────────────────────────────────
# Baza (opcjonalna) – łagodny fallback
# ──────────────────────────────────────────────────────────────────────────────
try:
    from .base import PanelBase, PanelContext  # type: ignore
except Exception:
    PanelBase = object  # type: ignore

    class PanelContext:  # type: ignore
        def __init__(self, **kw): self.__dict__.update(kw)

__all__ = [
    "PanelBase",
    "PanelContext",
    "PANEL_REGISTRY",
    "get_panel",
    "list_panels",
]

# ──────────────────────────────────────────────────────────────────────────────
# Wykrywanie i import modułów paneli
# ──────────────────────────────────────────────────────────────────────────────
def _iter_candidate_modules() -> Iterable[str]:
    """
    Zwraca nazwy podmodułów w pakiecie, które wyglądają jak panele.
    Kryteria:
      - nazwa zaczyna się od 'panel_' LUB kończy się na '_panel'
    """
    pkg = importlib.import_module(__name__)
    pkg_path = getattr(pkg, "__path__", None)
    if not pkg_path:
        return []
    for _, mod_name, is_pkg in pkgutil.iter_modules(pkg_path):
        if is_pkg:
            continue
        if mod_name.startswith("panel_") or mod_name.endswith("_panel"):
            yield mod_name


def _safe_import(mod_name: str) -> Optional[ModuleType]:
    """
    Importuje `glitchlab.gui.panels.<mod_name>` z miękką obsługą błędów.
    """
    fq = f"{__name__}.{mod_name}"
    try:
        return importlib.import_module(fq)
    except Exception as e:
        print(f"[panels] skip {mod_name}: {e}", file=sys.stderr)
        return None


def _derive_names_from_module(mod_name: str) -> List[str]:
    """
    Z nazwy modułu tworzy zestaw nazw-aliastów panelu, np.:
      'panel_spectral_shaper' → ['spectral_shaper', 'panel_spectral_shaper']
      'spectral_shaper_panel' → ['spectral_shaper', 'spectral_shaper_panel']
    """
    names = {mod_name}
    if mod_name.startswith("panel_"):
        names.add(mod_name[len("panel_"):])
    if mod_name.endswith("_panel"):
        names.add(mod_name[:-len("_panel")])
    return sorted(names)


def _canonicalize(name: str) -> str:
    """Prosta normalizacja nazwy do klucza rejestru."""
    return (name or "").strip().lower()


# ──────────────────────────────────────────────────────────────────────────────
# Rejestr paneli
# ──────────────────────────────────────────────────────────────────────────────
PANEL_REGISTRY: Dict[str, Union[type, Callable]] = {}


def _register_from_module(mod: ModuleType, mod_name: str) -> None:
    """
    Jeśli moduł eksportuje `Panel`, rejestrujemy go pod kilkoma aliasami.
    Preferowana nazwa:
      - PANEL_NAME albo NAME (jeżeli obecne, string)
    Dodatkowo aliasy wynikające z nazwy modułu.
    """
    try:
        panel_obj = getattr(mod, "Panel", None)
        if panel_obj is None:
            return

        # kanoniczna nazwa z modułu, jeśli dostępna
        preferred = None
        for attr in ("PANEL_NAME", "NAME"):
            val = getattr(mod, attr, None)
            if isinstance(val, str) and val.strip():
                preferred = val.strip()
                break

        # aliasy z nazwy modułu
        aliases = _derive_names_from_module(mod_name)

        keys: List[str] = []
        if preferred:
            keys.append(preferred)
        keys.extend(aliases)

        for k in keys:
            key = _canonicalize(k)
            if not key:
                continue
            # nie nadpisujemy istniejącego wpisu lepszym gorszym aliasem
            if key not in PANEL_REGISTRY:
                PANEL_REGISTRY[key] = panel_obj
    except Exception as e:
        print(f"[panels] register {mod_name}: {e}", file=sys.stderr)


def _autoboot() -> None:
    for mod_name in _iter_candidate_modules():
        mod = _safe_import(mod_name)
        if mod is not None:
            _register_from_module(mod, mod_name)


_autoboot()

# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────
def get_panel(name: str) -> Optional[Union[type, Callable]]:
    """
    Zwraca klasę/wywoływalny `Panel` dla podanej nazwy (aliasy wspierane),
    np.: 'spectral_shaper', 'panel_spectral_shaper', 'spectral_shaper_panel'.
    """
    return PANEL_REGISTRY.get(_canonicalize(name))


def list_panels() -> List[str]:
    """Lista zarejestrowanych nazw paneli (posortowana)."""
    return sorted(PANEL_REGISTRY.keys())
