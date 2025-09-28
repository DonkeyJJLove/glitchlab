# -*- coding: utf-8 -*-
"""
tests/test_hybrid_mosaic_algo.py
Zestaw testów własnościowych i regresyjnych dla hybrydowego algorytmu AST⇄Mozaika.
Uruchom:
  pytest -q
"""

from __future__ import annotations
import math
from typing import List

import numpy as np
import pytest

import glitchlab.gui.mosaic.hybrid_ast_mosaic as hma

EDGE_THR = hma.EDGE_THR_DEFAULT  # spójny próg z modułem


# ──────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def ast_raw():
    return hma.ast_deltas(hma.EXAMPLE_SRC)

@pytest.fixture(scope="module", params=[0.0, 0.25, 0.5, 0.75])
def lam(request):
    return request.param

@pytest.fixture(scope="module")
def ast_l(ast_raw, lam):
    return hma.compress_ast(ast_raw, lam)

@pytest.fixture(scope="module", params=[("grid", 12, 12), ("hex", 12, 12)])
def mosaic(request):
    kind, R, C = request.param
    return hma.build_mosaic(rows=R, cols=C, seed=7, kind=kind, edge_thr=EDGE_THR)


# ──────────────────────────────────────────────────────────────────────────────
# INWARIANTY I WŁAŚCIWOŚCI METRYK
# ──────────────────────────────────────────────────────────────────────────────

def test_I1_I3_via_invariants_check(ast_raw, lam, mosaic):
    aL = hma.compress_ast(ast_raw, lam)
    inv = hma.invariants_check(ast_raw, aL, mosaic, EDGE_THR)
    assert inv["I1_alpha_plus_beta_eq_1"]
    assert inv["I3_compression_monotone"]

def test_I2_metric_D_M_basic(mosaic):
    roi = hma.region_ids(mosaic, "roi", EDGE_THR)
    top = hma.region_ids(mosaic, "all", EDGE_THR)[:len(roi)] if len(roi) else []
    # nieujemność + tożsamość
    assert hma.D_M(roi, roi, mosaic, EDGE_THR) == 0.0
    if roi and top:
        d1 = hma.D_M(roi, top, mosaic, EDGE_THR)
        d2 = hma.D_M(top, roi, mosaic, EDGE_THR)
        assert d1 >= 0.0 and d2 >= 0.0
        assert abs(d1 - d2) < 1e-9

def test_CR_AST_monotone_in_lambda(ast_raw):
    # CR_AST(λ) powinno być niemalejące (większa kompresja przy większym λ)
    lams = [0.0, 0.25, 0.5, 0.75]
    vals: List[float] = []
    for lam in lams:
        L = hma.compress_ast(ast_raw, lam)
        cr = (ast_raw.S + ast_raw.H + max(1, ast_raw.Z)) / max(1, L.S + L.H + max(1, L.Z))
        vals.append(cr)
    assert all(vals[i] <= vals[i+1] + 1e-9 for i in range(len(vals)-1))


# ──────────────────────────────────────────────────────────────────────────────
# ZACHOWANIE Φ / Ψ
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("kind", ["heur", "balanced", "entropy"])
def test_phi_cost_defined(ast_l, mosaic, kind):
    sel = dict(
        heur=hma.phi_region_for,
        balanced=hma.phi_region_for_balanced,
        entropy=hma.phi_region_for_entropy,
    )[kind]
    J, det = hma.phi_cost(ast_l, mosaic, EDGE_THR, selector=sel)
    assert isinstance(J, float) and J >= 0.0
    assert isinstance(det, dict) and len(det) > 0

def test_phi2_not_worse_than_phi1_on_average(ast_raw, mosaic):
    lam = 0.60
    aL = hma.compress_ast(ast_raw, lam)
    J1, _ = hma.phi_cost(aL, mosaic, EDGE_THR, selector=hma.phi_region_for)
    J2, _ = hma.phi_cost(aL, mosaic, EDGE_THR, selector=hma.phi_region_for_balanced)
    # nie gwarantujemy przewagi w danym M, ale zwykle J2 ≤ J1
    assert J2 <= J1 or math.isclose(J1, J2, rel_tol=1e-3, abs_tol=1e-6)

def test_psi_feedback_changes_meta(ast_l, mosaic):
    before = {i: n.meta.copy() for i, n in ast_l.nodes.items()}
    after = hma.psi_feedback(ast_l, mosaic, delta=0.25, thr=EDGE_THR)
    changed = sum(int(np.linalg.norm(after.nodes[i].meta - before[i]) > 1e-9) for i in before.keys())
    assert changed > 0


# ──────────────────────────────────────────────────────────────────────────────
# ALIGN / DISTANCE
# ──────────────────────────────────────────────────────────────────────────────

def test_align_in_0_1(ast_l, mosaic):
    ast_after = hma.psi_feedback(ast_l, mosaic, delta=0.25, thr=EDGE_THR)
    align = 1.0 - min(1.0, hma.distance_ast_mosaic(ast_after, mosaic, EDGE_THR))
    assert 0.0 <= align <= 1.0


# ──────────────────────────────────────────────────────────────────────────────
# HEX PACKING — sanity (jeśli dostępny)
# ──────────────────────────────────────────────────────────────────────────────

def test_hex_packing_two_neighbor_scales():
    M = hma.build_mosaic(rows=8, cols=8, seed=3, kind="hex", edge_thr=EDGE_THR)
    centers = getattr(M, "hex_centers", None)
    if centers is None:
        pytest.skip("Brak metadanych hex_centers — test pominięty.")
    # wybierz środek siatki (indeks ≈ wiersz 3, kol 3)
    mid_idx = 3 * M.cols + 3
    x0, y0 = centers[mid_idx]
    # policz odległości do najbliższych sąsiadów (pierwsze ~8-10)
    dists = []
    for i, (x, y) in enumerate(centers):
        if i == mid_idx:
            continue
        d = math.hypot(x - x0, y - y0)
        if d > 0:
            dists.append(d)
    dists = sorted(dists)[:12]
    # zgrupuj odległości z tolerancją — spodziewamy się ~2 klastrów (kolumnowy i wierszowy kierunek)
    def bucketize(vals, tol=1e-6):
        groups = []
        for v in vals:
            placed = False
            for g in groups:
                if abs(g[0] - v) < 1e-6:
                    g[1].append(v); placed = True; break
            if not placed:
                groups.append([v, [v]])
        return [ (g[0], len(g[1])) for g in groups ]
    groups = bucketize(dists)
    # powinny być co najmniej 2 wyraźne grupy
    assert len(groups) >= 2


# ──────────────────────────────────────────────────────────────────────────────
# OPERACYJNY TEST ZNAKU (zredukowany)
# ──────────────────────────────────────────────────────────────────────────────

def test_sign_test_small_sample(ast_raw):
    lam = 0.60
    wins = losses = 0
    for seed in range(20):
        M = hma.build_mosaic(rows=12, cols=12, seed=seed, kind="grid", edge_thr=EDGE_THR)
        aL = hma.compress_ast(ast_raw, lam)
        j1, _ = hma.phi_cost(aL, M, EDGE_THR, selector=hma.phi_region_for)
        j2, _ = hma.phi_cost(aL, M, EDGE_THR, selector=hma.phi_region_for_balanced)
        if j1 > j2:
            wins += 1
        elif j1 < j2:
            losses += 1
    # nie dowód statystyczny, ale w małej próbce spodziewamy się wins >= losses
    assert wins >= losses
