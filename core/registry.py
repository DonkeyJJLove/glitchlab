from typing import Callable, Dict

_FILTERS: Dict[str, Callable] = {}


def register(name: str):
    def deco(fn: Callable):
        if name in _FILTERS:
            raise ValueError(f"Filter '{name}' already registered")
        _FILTERS[name] = fn
        return fn

    return deco


def get(name: str) -> Callable:
    if name not in _FILTERS:
        raise KeyError(f"Unknown filter '{name}'")
    return _FILTERS[name]


def available():
    return sorted(_FILTERS.keys())
