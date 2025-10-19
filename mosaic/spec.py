# glitchlab/mosaic/spec.py
"""
---
version: 2
kind: module
id: "mosaic-spec"
created_at: "2025-09-13"
name: "glitchlab.mosaic.spec"
author: "GlitchLab v2"
role: "HUD & Mosaic Routing Spec"
description: >
  Specyfikacja routingu HUD/mozaiki. Definiuje zbiory wzorców (wildcards) do
  wyboru kluczy z ctx.cache dla slotów HUD (slot1..slot3) oraz nakładek
  viewportu. Zapewnia normalizację, walidację, głębokie scalanie (merge),
  dopasowanie wzorców do dostępnych kluczy oraz I/O (JSON/YAML).

inputs:
  spec:
    type: "Mapping[str, Any] | None"
    desc: "Cząstkowa specyfikacja użytkownika do zmergowania z domyślną."
  cache_like:
    type: "Mapping[str, Any]"
    desc: "Źródło dostępnych kluczy (np. ctx.cache) do rozstrzygania wzorców."
  path:
    type: "str|Path"
    desc: "Ścieżka do zapisu/odczytu spec (JSON/YAML)."

outputs:
  normalized_spec:
    type: "Dict[str, Any]"
    desc: "Spec w formie kanonicznej (sloty i overlay jako listy stringów)."
  hud_selection:
    type: "Dict[str, Optional[str]]"
    desc: "{slot1, slot2, slot3} → wybrany klucz lub None."
  overlay_key:
    type: "Optional[str]"
    desc: "Wybrany klucz nakładki viewportu lub None."

interfaces:
  exports:
    - "get_default_spec"
    - "normalize_spec"
    - "validate_spec"
    - "merge_spec"
    - "pick_first_available"
    - "resolve_hud_slots"
    - "pick_overlay"
    - "load_spec"
    - "save_spec"
  depends_on: ["json","re","pathlib","yaml?","copy","typing"]
  used_by:
    - "glitchlab.app.widgets.hud"
    - "glitchlab.app.widgets.image_canvas"
    - "glitchlab.app.app"
    - "glitchlab.analysis.exporters"

policy:
  deterministic: true
  side_effects: "tylko operacje plikowe przy load/save"
constraints:
  - "pure Python; brak SciPy/OpenCV"
  - "YAML opcjonalne (PyYAML)"
  - "wzorce: '*' → .*, dopasowanie pełne (anchored)"

hud:
  influence:
    - "Określa priorytety źródeł dla slotów HUD"
    - "Wybiera domyślną nakładkę mozaiki/diag na viewer"
license: "Proprietary"
---
"""
# mosaic/spec.py
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple
from pathlib import Path
import json
import re
import copy

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


# ---------------------------
# Default specification
# ---------------------------

_DEFAULT_SPEC: Dict[str, Any] = {
    "hud": {
        "slot1": ["stage/0/in", "stage/0/metrics_in", "format/jpg_grid"],
        "slot2": ["stage/0/out", "stage/0/metrics_out", "stage/0/fft_mag"],
        "slot3": ["stage/0/diff", "stage/0/diff_stats", "ast/json"],
    },
    "overlays": {
        "viewport": ["stage/*/mosaic", "diag/*/*", "format/jpg_grid"],
    },
}


def get_default_spec() -> Dict[str, Any]:
    """Return a deep copy of the built-in default HUD/mosaic spec."""
    return copy.deepcopy(_DEFAULT_SPEC)


# ---------------------------
# Validation / Normalization
# ---------------------------

def _ensure_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [str(v) for v in x]
    return [str(x)]


def normalize_spec(spec: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """
    Ensure required keys exist and values have canonical shapes.
    - hud.slot{1,2,3}: List[str]
    - overlays.viewport: List[str]
    """
    base = get_default_spec()
    if spec:
        base = merge_spec(base, spec)

    hud = base.get("hud", {})
    base["hud"] = {
        "slot1": _ensure_list(hud.get("slot1")),
        "slot2": _ensure_list(hud.get("slot2")),
        "slot3": _ensure_list(hud.get("slot3")),
    }

    ov = base.get("overlays", {})
    base["overlays"] = {
        "viewport": _ensure_list(ov.get("viewport")),
    }
    return base


def validate_spec(spec: Mapping[str, Any]) -> List[str]:
    """Return a list of validation error strings (empty -> valid)."""
    errors: List[str] = []
    if "hud" not in spec or not isinstance(spec["hud"], Mapping):
        errors.append("missing 'hud' mapping")
    else:
        for slot in ("slot1", "slot2", "slot3"):
            v = spec["hud"].get(slot)
            if not isinstance(v, (list, tuple)):
                errors.append(f"'hud.{slot}' must be a list of patterns")
            else:
                if not all(isinstance(x, (str, int, float)) for x in v):
                    errors.append(f"'hud.{slot}' contains non-primitive values")

    if "overlays" not in spec or not isinstance(spec["overlays"], Mapping):
        errors.append("missing 'overlays' mapping")
    else:
        v = spec["overlays"].get("viewport")
        if not isinstance(v, (list, tuple)):
            errors.append("'overlays.viewport' must be a list of patterns")
    return errors


# ---------------------------
# Merging
# ---------------------------

def merge_spec(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Deep-merge override into base; lists are replaced (not concatenated)."""
    def _merge(a: Any, b: Any) -> Any:
        if isinstance(a, Mapping) and isinstance(b, Mapping):
            out: Dict[str, Any] = {}
            keys = set(a.keys()) | set(b.keys())
            for k in keys:
                if k in a and k in b:
                    out[k] = _merge(a[k], b[k])
                elif k in a:
                    out[k] = copy.deepcopy(a[k])
                else:
                    out[k] = copy.deepcopy(b[k])
            return out
        # lists are replaced by override
        return copy.deepcopy(b)
    return _merge(base, override)


# ---------------------------
# Matching utilities
# ---------------------------

def _pattern_to_regex(pat: str) -> re.Pattern:
    """
    Convert a simple wildcard pattern to regex:
    - '*' matches any run of non-empty characters (greedy), including slashes.
    - patterns are anchored (^...$)
    """
    # Escape regex, then restore '*' semantics.
    parts: List[str] = []
    for ch in pat:
        if ch == "*":
            parts.append(".*")
        else:
            parts.append(re.escape(ch))
    rx = "^" + "".join(parts) + "$"
    return re.compile(rx)


def pick_first_available(patterns: Iterable[str], cache_like: Mapping[str, Any]) -> Optional[str]:
    """
    Return the first key from cache_like matching the ordered pattern list.
    If a pattern is an exact key present in cache, return it immediately.
    Otherwise use wildcard matching.
    """
    cache_keys = list(cache_like.keys())
    # fast path for exact
    for pat in patterns:
        if pat in cache_like:
            return pat
        # wildcard
        if "*" in pat:
            rx = _pattern_to_regex(pat)
            for k in cache_keys:
                if rx.match(k):
                    return k
    return None


def resolve_hud_slots(spec: Mapping[str, Any], cache_like: Mapping[str, Any]) -> Dict[str, Optional[str]]:
    """Pick keys for slot1..slot3 according to the spec and available cache keys."""
    n = normalize_spec(spec)
    hud = n["hud"]
    return {
        "slot1": pick_first_available(hud.get("slot1", ()), cache_like),
        "slot2": pick_first_available(hud.get("slot2", ()), cache_like),
        "slot3": pick_first_available(hud.get("slot3", ()), cache_like),
    }


def pick_overlay(spec: Mapping[str, Any], cache_like: Mapping[str, Any]) -> Optional[str]:
    """Pick an overlay key for the viewer."""
    n = normalize_spec(spec)
    pats = n["overlays"].get("viewport", [])
    return pick_first_available(pats, cache_like)


# ---------------------------
# I/O
# ---------------------------

def load_spec(path: str | Path) -> Dict[str, Any]:
    """
    Load spec from .json or .yaml/.yml (if PyYAML is present).
    Returns a normalized dict; raises on unsupported extensions.
    """
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".json":
        data = json.loads(Path(p).read_text(encoding="utf-8"))
    elif ext in (".yaml", ".yml"):
        if yaml is None:
            raise RuntimeError("PyYAML not available to load YAML spec")
        data = yaml.safe_load(Path(p).read_text(encoding="utf-8"))  # type: ignore
    else:
        raise ValueError(f"unsupported spec format: {ext}")
    return normalize_spec(data or {})


def save_spec(path: str | Path, spec: Mapping[str, Any]) -> None:
    """
    Save spec to .json or .yaml/.yml (if PyYAML is present).
    Writes a normalized version to ensure shape consistency.
    """
    p = Path(path)
    ext = p.suffix.lower()
    data = normalize_spec(spec)
    if ext == ".json":
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    elif ext in (".yaml", ".yml"):
        if yaml is None:
            raise RuntimeError("PyYAML not available to save YAML spec")
        p.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")  # type: ignore
    else:
        raise ValueError(f"unsupported spec format: {ext}")
