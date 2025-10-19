# glx/__init__.py — optional shim to prefer installed glitchlab.glx
"""
Compatibility shim: tests import `glx.*`. If a real package `glitchlab.glx`
exists (e.g. installed in editable mode), expose its contents; otherwise act as
a local package (so local files glx/tools/*.py will be importable).
"""
try:
    # prefer installed package if available
    from glitchlab import glx as _real  # type: ignore
    # re-export non-private symbols from installed package
    for _n in dir(_real):
        if not _n.startswith("_"):
            globals()[_n] = getattr(_real, _n)
except Exception:
    # fallback: leave as a normal package that will load local modules
    pass
