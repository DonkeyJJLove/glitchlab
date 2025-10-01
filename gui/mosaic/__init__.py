
# glitchlab/gui/mosaic/__init__.py
# Python 3.9 — lazy re-exports to avoid heavy import side effects.

from importlib import import_module
from typing import List

__all__: List[str] = [
    # budowa mozaiki
    "build_mosaic", "build_mosaic_grid", "build_mosaic_hex",
    # AST i kompresja
    "ast_deltas", "compress_ast",
    # Φ/Ψ i metryki
    "phi_region_for", "phi_region_for_balanced", "phi_region_for_entropy",
    "phi_cost", "psi_feedback",
    "mosaic_profile", "distance_ast_mosaic", "invariants_check",
    # narzędzia / run
    "run_once", "sweep", "sign_test_phi2_better",
    # typy danych
    "AstNode", "AstSummary", "Mosaic",
    # stałe
    "EDGE_THR_DEFAULT", "W_DEFAULT",
]


def __getattr__(name: str):
    if name in __all__:
        mod = import_module(".hybrid_ast_mosaic", __name__)
        return getattr(mod, name)
    raise AttributeError(name)


def __dir__():
    return sorted(list(globals().keys()) + __all__)
