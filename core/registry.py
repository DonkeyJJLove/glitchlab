# glitchlab/core/registry.py
from __future__ import annotations

from threading import RLock
from typing import Any, Callable, Dict, List, Optional

__all__ = [
    "register",
    "get",
    "available",
    "canonical",
    "alias",
    "meta",
    "register_alias",
]

# Internal state (lowercased keys)
_LOCK = RLock()
_FUNCS: Dict[str, Callable[..., Any]] = {}
_DEFAULTS: Dict[str, Dict[str, Any]] = {}
_DOCS: Dict[str, str] = {}
_ALIASES: Dict[str, str] = {}  # alias_name -> target_canonical (both lowercased)


def _lc(name: str) -> str:
    if not isinstance(name, str):
        raise TypeError("name must be str")
    return name.strip().lower()


def canonical(name: str) -> str:
    """Resolve alias chain to a canonical registered name. Raises KeyError if not found."""
    key = _lc(name)
    seen = set()
    with _LOCK:
        while key in _ALIASES:
            if key in seen:
                # Break alias loops defensively
                break
            seen.add(key)
            key = _ALIASES[key]
        if key not in _FUNCS:
            raise KeyError(f"Unknown filter: {name!r}")
        return key


def available() -> List[str]:
    """List canonical registered filter names (sorted)."""
    with _LOCK:
        return sorted(_FUNCS.keys())


def get(name: str) -> Callable[..., Any]:
    """Return the callable for a (possibly aliased) filter name."""
    key = canonical(name)
    with _LOCK:
        return _FUNCS[key]


def meta(name: str) -> Dict[str, Any]:
    """Return metadata: {'name','defaults','doc','aliases'} for the (possibly aliased) name."""
    key = canonical(name)
    with _LOCK:
        # aliases that resolve to this canonical
        aliases = [a for a, tgt in _ALIASES.items() if tgt == key]
        return {
            "name": key,
            "defaults": dict(_DEFAULTS.get(key, {})),
            "doc": _DOCS.get(key, ""),
            "aliases": sorted(aliases),
        }


def alias(alias_name: str, target_name: str) -> bool:
    """
    Create/overwrite an alias. Returns True on success, False if target doesn't exist
    or alias would shadow another canonical function (different from target).
    """
    a = _lc(alias_name)
    t = _lc(target_name)
    with _LOCK:
        if t not in _FUNCS:
            return False
        # Do not allow creating alias that collides with an existing canonical function
        if a in _FUNCS and a != t:
            return False
        # Resolve final target to avoid chains where possible
        try:
            t_final = canonical(t)
        except KeyError:
            return False
        # Prevent trivial self-alias loops
        if a == t_final:
            # No-op alias (alias of itself) is acceptable; ensure it's removed if present
            _ALIASES.pop(a, None)
            return True
        _ALIASES[a] = t_final
        return True


def register(name: str, defaults: Optional[Dict[str, Any]] = None, doc: Optional[str] = None):
    """
    Decorator for registering a filter function with optional defaults/doc.
    Usage:
        @register("anisotropic_contour_warp", defaults={"strength":1.0}, doc="Warp along contours")
        def anisotropic_contour_warp(img, ctx, **params): ...
    """
    key = _lc(name)

    def _decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        if not callable(fn):
            raise TypeError("register: fn must be callable")
        with _LOCK:
            _FUNCS[key] = fn
            if defaults is not None:
                if not isinstance(defaults, dict):
                    raise TypeError("register: defaults must be a dict or None")
                _DEFAULTS[key] = dict(defaults)
            else:
                _DEFAULTS.setdefault(key, {})
            if doc is not None:
                if not isinstance(doc, str):
                    raise TypeError("register: doc must be a str or None")
                _DOCS[key] = doc
            else:
                _DOCS.setdefault(key, "")
            # Any alias pointing to this key remains valid; nothing else to do
        return fn

    return _decorator


# Back-compat helper name (no-op wrapper)
def register_alias(dst: str, src: str) -> bool:
    return alias(dst, src)
