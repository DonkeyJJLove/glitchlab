# glitchlab/app/preset_dir_fallback.py
# -*- coding: utf-8 -*-
"""
preset_dir_fallback — zapewnia globalne funkcje _get_preset_dir/_set_preset_dir,
jeśli nie są zdefiniowane w module app.py. Wstrzykuje je do builtins,
aby wywołanie w app.py działało bez zmian.
"""
from __future__ import annotations
import builtins
from pathlib import Path
import os


def _compute_default_dir() -> str:
    # 1) <pakiet>/presets
    try:
        here = Path(__file__).resolve()
        pkg_root = here.parents[1]  # glitchlab/app -> glitchlab
        cand = pkg_root / "presets"
        if cand.exists():
            return str(cand)
    except Exception:
        pass
    # 2) CWD/presets
    cand = Path.cwd() / "presets"
    if cand.exists():
        return str(cand)
    # 3) ~/Documents/glitchlab/presets
    cand = Path.home() / "Documents" / "glitchlab" / "presets"
    os.makedirs(cand, exist_ok=True)
    return str(cand)


# Stan zapamiętany lokalnie w module (prosty cache).
_PRESET_DIR = None


def _get_preset_dir() -> str:
    global _PRESET_DIR
    if not _PRESET_DIR:
        _PRESET_DIR = _compute_default_dir()
    return _PRESET_DIR


def _set_preset_dir(path: str) -> None:
    global _PRESET_DIR
    if not path:
        return
    try:
        p = Path(path)
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)
        _PRESET_DIR = str(p)
    except Exception:
        # zostaw poprzednią wartość
        pass


# Wstrzyknięcie do builtins (tylko jeśli brak).
if not hasattr(builtins, "_get_preset_dir"):
    builtins._get_preset_dir = _get_preset_dir
