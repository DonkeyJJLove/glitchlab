# glitchlab/core/registry.py

registry = {}


def register(name):
    """
    Dekorator do rejestrowania filtrów w globalnym registry.
    """
    def decorator(fn):
        if name in registry:
            raise KeyError(f"Filter '{name}' already registered")
        registry[name] = fn
        return fn
    return decorator


def get(name):
    """
    Pobierz funkcję filtra po nazwie.
    """
    if name not in registry:
        raise KeyError(f"Unknown filter '{name}'")
    return registry[name]


def available():
    """
    Lista dostępnych filtrów (nazwy).
    """
    return list(registry.keys())


# --- Import filtrów, żeby się zarejestrowały ---
from glitchlab.filters import (
    rgb_offset,
    wave_distort,
    pixel_sort,
    block_mosh,
    channel_shuffle,
    color_invert_masked,
    posterize,
    noise,
    protect_edges,
    depth_displace,   # <-- dodajemy nasz nowy filtr
)
