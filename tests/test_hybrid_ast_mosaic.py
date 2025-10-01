# tests/test_hybrid_ast_mosaic.py

import json
import pytest

import glitchlab.gui.mosaic.hybrid_ast_mosaic as ham
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

def test_run_once_default():
    """Uruchamia run_once na EXAMPLE_SRC i sprawdza poprawność metryk Φ/Ψ."""
    res = ham.run_once(
        lam=0.5,
        delta=0.25,
        rows=4,
        cols=4,
        thr=ham.EDGE_THR_DEFAULT
    )
    # wynik musi zawierać kluczowe metryki
    for key in ["J_phi1", "J_phi2", "J_phi3", "Align", "S", "H", "Z", "alpha", "beta"]:
        assert key in res, f"Brak klucza {key} w wynikach run_once"

    # sanity checks (wszystkie wartości są skończone i >=0)
    for k, v in res.items():
        if isinstance(v, (int, float)):
            assert v == v and v >= 0, f"{k} ma niepoprawną wartość {v}"

    # align musi być w [0,1]
    assert 0.0 <= res["Align"] <= 1.0


def test_invariants_hold():
    """Sprawdza inwarianty I1–I3 na przykładzie EXAMPLE_SRC."""
    astA = ham.ast_deltas(ham.EXAMPLE_SRC)
    astB = ham.compress_ast(astA, lam=0.5)
    M = ham.build_mosaic(4, 4, seed=7, kind="grid", edge_thr=ham.EDGE_THR_DEFAULT)
    inv = ham.invariants_check(astA, astB, M, ham.EDGE_THR_DEFAULT)

    # wszystkie inwarianty muszą przejść
    for name, ok in inv.items():
        assert ok, f"Inwariant {name} nie został spełniony"


def test_phi_cost_and_psi_feedback():
    """Test projekcji Φ i podnoszenia Ψ."""
    ast_summary = ham.ast_deltas(ham.EXAMPLE_SRC)
    M = ham.build_mosaic(4, 4, seed=1, kind="grid", edge_thr=ham.EDGE_THR_DEFAULT)

    # koszt Φ
    J, details = ham.phi_cost(ast_summary, M, thr=ham.EDGE_THR_DEFAULT, selector=ham.phi_region_for_balanced)
    assert J >= 0
    assert isinstance(details, dict)
    assert all("cost" in d for d in details.values())

    # Ψ-feedback aktualizuje meta-wektory
    ast_after = ham.psi_feedback(ast_summary, M, delta=0.5, thr=ham.EDGE_THR_DEFAULT)
    for nid, node in ast_after.nodes.items():
        assert node.meta.shape == (6,)


if __name__ == "__main__":
    pytest.main([__file__])
