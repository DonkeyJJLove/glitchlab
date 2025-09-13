# glitchlab/gui/preset_paths.py
# -*- coding: utf-8 -*-
"""
Robust preset directory discovery & persistence for GlitchLab GUI.

Usage (in app.py):
    from .preset_paths import discover_preset_dirs, get_last_preset_dir, set_last_preset_dir, list_preset_files

Then:
    dirs = discover_preset_dirs()
    current = get_last_preset_dir(dirs)
    # bind to combobox StringVar
    self.var_preset_dir.set(current)
    # when user changes folder:
    set_last_preset_dir(self.var_preset_dir.get())

Functions are Windows/Unix safe and handle env overrides.
"""
from __future__ import annotations
from pathlib import Path
import os, json
from typing import List

# Location for small GUI settings
_SETTINGS = Path.home() / ".glitchlab_gui.json"


def _load_settings() -> dict:
    try:
        if _SETTINGS.exists():
            return json.loads(_SETTINGS.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_settings(cfg: dict) -> None:
    try:
        _SETTINGS.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def discover_preset_dirs() -> List[str]:
    """Return a list of candidate preset directories in priority order."""
    here = Path(__file__).resolve()
    candidates = []
    # 1) env override (can be multiple, separated by os.pathsep)
    env = os.environ.get("GLITCHLAB_PRESETS", "")
    for token in (env.split(os.pathsep) if env else []):
        p = Path(token).expanduser().resolve()
        candidates.append(p)
    # 2) packaged 'presets' next to glitchlab/ (.. / glitches / presets)
    try:
        pkg_root = here.parents[1]  # glitchlab/
        candidates.append((pkg_root / "presets").resolve())
    except Exception:
        pass
    # 3) cwd/presets
    candidates.append((Path.cwd() / "presets").resolve())
    # 4) user home
    candidates.append((Path.home() / "glitchlab" / "presets").resolve())
    # uniquify and exist filter
    seen = set()
    out: List[str] = []
    for p in candidates:
        ps = str(p)
        if ps in seen:
            continue
        seen.add(ps)
        if p.exists() and p.is_dir():
            out.append(ps)
    return out


def list_preset_files(dirpath: str) -> List[str]:
    """Return sorted list of preset files (.yml/.yaml/.json) in dirpath."""
    p = Path(dirpath).expanduser()
    if not p.exists():
        return []
    files = [str(q) for q in p.iterdir() if q.is_file() and q.suffix.lower() in (".yml", ".yaml", ".json")]
    files.sort(key=lambda s: s.lower())
    return files


def get_last_preset_dir(fallback_dirs: List[str]) -> str:
    """Load last chosen preset dir from settings; fall back to first valid."""
    cfg = _load_settings()
    s = cfg.get("preset_dir", "")
    if s:
        ps = Path(s).expanduser()
        if ps.exists() and ps.is_dir():
            return str(ps)
    return fallback_dirs[0] if fallback_dirs else str((Path.cwd() / "presets").resolve())


def set_last_preset_dir(path: str) -> None:
    cfg = _load_settings()
    cfg["preset_dir"] = str(Path(path).expanduser().resolve())
    _save_settings(cfg)
