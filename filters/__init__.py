# glitchlab/filters/__init__.py
# -*- coding: utf-8 -*-
"""
Jawne ładowanie modułów filtrów (+ aliasy).
Dekorator @register i funkcje get/alias pochodzą z core.registry.
"""

from __future__ import annotations
import importlib, sys

try:
    from glitchlab.core.registry import get as _get, register_alias
except Exception as e:
    _get = None  # type: ignore
    register_alias = None  # type: ignore
    print(f"[filters] WARN: registry API unavailable: {e}", file=sys.stderr)

_MODULES = (
    "anisotropic_contour_warp",
    "block_mosh_grid",
    "pixel_sort_adaptive",
    "spectral_shaper",
    "phase_glitch",
    "rgb_offset",
    "depth_displace",
    "depth_parallax",
)

def _safe_import(modname: str) -> None:
    try:
        importlib.import_module(f"{__name__}.{modname}")
    except Exception as e:
        print(f"[filters] skip {modname}: {e}", file=sys.stderr)

for _m in _MODULES:
    _safe_import(_m)
del _m

# aliasy (GUI/preset → realny filtr)
_ALIASES = {
    "conture_flow": "anisotropic_contour_warp",
    "anisotropic_contour_flow": "anisotropic_contour_warp",
    "block_mosh": "block_mosh_grid",
    "spectral_shaper_lab": "spectral_shaper",
}

if register_alias is not None and _get is not None:
    for _dst, _src in _ALIASES.items():
        try:
            _get(_src)  # upewnij się, że baza istnieje
            register_alias(_dst, _src)
        except Exception:
            pass

del importlib, sys, _safe_import, _ALIASES, _MODULES, _get, register_alias
