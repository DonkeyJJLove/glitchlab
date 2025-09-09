# glitchlab/filters/__init__.py
# -*- coding: utf-8 -*-
"""
Jawne ładowanie filtrów (bez autoloadera) + aliasy nazw.
Dodawaj tu kolejne importy w miarę dokładania plików filtrów do pakietu.
"""

from __future__ import annotations

# --- 1) Importy modułów filtrów (jawne; każdy w try/except) ---

# Te dwa na pewno masz:
try:
    from . import anisotropic_contour_warp  # rejestruje @register("anisotropic_contour_warp")
except Exception:
    pass

try:
    from . import block_mosh_grid           # rejestruje @register("block_mosh_grid")
except Exception:
    pass

# Jeśli dodasz kolejne pliki, dopisz je tutaj (nie szkodzi, jeśli jeszcze ich nie ma):
for _modname in (
    "pixel_sort_adaptive",
    "spectral_shaper",
    "phase_glitch",
    "noise_perlin_grid",
    "depth_displace",
    "spectral_ring_lab",
):
    try:
        __import__(f"{__name__}.{_modname}", fromlist=["*"])
    except Exception:
        pass


# --- 2) Aliasowanie nazw (literówki / stare nazwy presetów) ---

# Rejestr może udostępniać różne API; obsłużmy oba warianty.
try:
    from glitchlab.core.registry import register, get as _reg_get  # preferowane API
except Exception:  # fallback: tylko 'register' lub bez 'get'
    _reg_get = None
    try:
        from glitchlab.core.registry import register  # type: ignore
    except Exception:
        register = None  # type: ignore

# Spróbuj też bezpośrednio dostać się do słownika rejestru (gdy brak 'get').
_REG_DICT = None
try:
    from glitchlab.core import registry as _reg
    _REG_DICT = getattr(_reg, "_REGISTRY", None)
except Exception:
    pass


def _install_alias(dst: str, src: str) -> None:
    """
    Zarejestruj alias 'dst' wskazujący na filtr 'src'.
    Preferuje registry.get + @register; w razie czego korzysta z _REGISTRY.
    """
    # Wariant z registry.get (najczyściej)
    if register is not None and _reg_get is not None:
        try:
            target = _reg_get(src)
        except Exception:
            return  # brak filtra bazowego
        try:
            @register(dst)  # type: ignore[misc]
            def _alias(*args, **kwargs):
                return target(*args, **kwargs)
            return
        except Exception:
            pass

    # Fallback: manipulacja słownikiem rejestru (jeśli dostępny)
    if _REG_DICT is not None:
        try:
            if src in _REG_DICT and dst not in _REG_DICT:
                _REG_DICT[dst] = _REG_DICT[src]
        except Exception:
            pass


# Zestaw aliasów (po lewej: nazwa używana w presetach/GUI; po prawej: realny filtr)
_alias_map = {
    # literówki / historyczne nazwy:
    "conture_flow": "anisotropic_contour_warp",
    "anisotropic_contour_flow": "anisotropic_contour_warp",

    # popularna literówka "nosh" -> "noise"
    "nosh_perlin_grid": "noise_perlin_grid",
    "perlin_grid": "noise_perlin_grid",

    # skróty / uproszczenia
    "block_mosh": "block_mosh_grid",

    # warianty laboratoryjne → podstawowy
    "spectral_shaper_lab": "spectral_shaper",
    "spectral_ring": "spectral_ring_lab",
}

for _dst, _src in _alias_map.items():
    _install_alias(_dst, _src)

# (opcjonalnie) posprzątaj namespace modułu
del _dst, _src, _alias_map, _install_alias, _REG_DICT
try:
    del _reg, _reg_get
except Exception:
    pass
