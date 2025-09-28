# glitchlab/filters/__init__.py
# -*- coding: utf-8 -*-
"""
Jawne ładowanie modułów filtrów (+ aliasy).
Dekorator @register i funkcje get/alias pochodzą z core.registry.
"""

from __future__ import annotations

import importlib
import sys
from typing import Iterable, List, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Registry API – best-effort (różne wersje rdzenia)
# ──────────────────────────────────────────────────────────────────────────────
try:
    from glitchlab.core.registry import get as _get  # type: ignore
except Exception as e:  # pragma: no cover
    _get = None  # type: ignore
    print(f"[filters] WARN: registry.get unavailable: {e}", file=sys.stderr)

# preferuj register_alias; użyj alias jako fallback
_alias_fn = None
try:
    from glitchlab.core.registry import register_alias as _alias_fn  # type: ignore
except Exception:
    try:
        from glitchlab.core.registry import alias as _alias_fn  # type: ignore
    except Exception as e:  # pragma: no cover
        _alias_fn = None  # type: ignore
        print(f"[filters] WARN: registry.alias unavailable: {e}", file=sys.stderr)

# ──────────────────────────────────────────────────────────────────────────────
# Lista modułów do jawnego importu
# ──────────────────────────────────────────────────────────────────────────────
_MODULES: Tuple[str, ...] = (
    "anisotropic_contour_warp",
    "block_mosh",
    "block_mosh_grid",
    "pixel_sort_adaptive",
    "spectral_shaper",  # ⬅ ważne: nowy filtr
    "phase_glitch",
    "rgb_offset",
    "depth_displace",
    "depth_parallax",
    "default_identity",
    "gamma_gain",
    "rgb_glow",
    "tile_tess_probe",
)


# ──────────────────────────────────────────────────────────────────────────────
# Import helpers
# ──────────────────────────────────────────────────────────────────────────────
def _safe_import(modname: str) -> bool:
    """
    Importuje podmoduł `glitchlab.filters.<modname>`.
    Zwraca True jeśli się powiodło; w przeciwnym razie wypisuje ostrzeżenie i zwraca False.
    """
    fq = f"{__name__}.{modname}"
    try:
        importlib.import_module(fq)
        return True
    except Exception as e:
        print(f"[filters] skip {modname}: {e}", file=sys.stderr)
        return False


def load_all(modnames: Iterable[str] = _MODULES) -> List[str]:
    """
    Ładuje wszystkie moduły filtrów. Zwraca listę skutecznie załadowanych nazw.
    Możesz tego użyć w testach: `assert 'spectral_shaper' in load_all()`.
    """
    loaded: List[str] = []
    for m in modnames:
        if _safe_import(m):
            loaded.append(m)
    return loaded


# ──────────────────────────────────────────────────────────────────────────────
# Aliasowanie nazw – zgodność presetów z faktycznymi nazwami filtrów
# ──────────────────────────────────────────────────────────────────────────────
_ALIASES = {
    # stare/przyjazne nazwy → rzeczywisty filtr
    "conture_flow": "anisotropic_contour_warp",
    "anisotropic_contour_flow": "anisotropic_contour_warp",
    "spectral_shaper_lab": "spectral_shaper",
}


def add_aliases(mapping: dict = _ALIASES) -> None:
    """
    Dodaje aliasy z `mapping` do rejestru, o ile API jest dostępne i cel istnieje.
    """
    if _alias_fn is None or _get is None:
        return
    for alias_name, target_name in mapping.items():
        try:
            # upewnij się, że cel istnieje (zarejestrowany przez @register w module filtra)
            _get(target_name)  # może rzucić wyjątkiem jeśli brak
            _alias_fn(alias_name, target_name)  # alias(alias_name, target_name)
        except Exception:
            # ciche pominięcie – np. jeśli filtr nie został zarejestrowany
            pass


# ──────────────────────────────────────────────────────────────────────────────
# Auto-bootstrapping przy imporcie pakietu
# ──────────────────────────────────────────────────────────────────────────────
_loaded = load_all()
add_aliases()

# optional: eksport narzędzi
__all__ = [
    "load_all",
    "add_aliases",
]
