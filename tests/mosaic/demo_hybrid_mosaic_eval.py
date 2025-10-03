# -*- coding: utf-8 -*-
"""
tests/demo_hybrid_mosaic_eval.py

CLI demo/benchmark dla algorytmu hybrydowego AST ⇄ Mozaika (Φ/Ψ).
Zgodny z API: glitchlab.gui.mosaic.hybrid_ast_mosaic

Uruchomienia (przykłady):
  python tests/demo_hybrid_mosaic_eval.py --lam 0.60 --delta 0.25 --seeds 100 --rows 12 --cols 12 --kind grid
  python tests/demo_hybrid_mosaic_eval.py --lam 0.60 --delta 0.25 --seeds 100 --rows 12 --cols 12 --kind hex
"""

from __future__ import annotations
import argparse
import json
from typing import Dict, List

import numpy as np

# Import modułu algorytmu – utrzymujemy krótkie aliasy.
import glitchlab.gui.mosaic.hybrid_ast_mosaic as hma


# ──────────────────────────────────────────────────────────────────────────────
# POMOCNICZE
# ──────────────────────────────────────────────────────────────────────────────

def _sign_test_p(wins: int, losses: int) -> float:
    """Dwustronny test znaku (binomial, p=0.5) na rozstrzygnięciach (bez remisów)."""
    n = wins + losses
    if n == 0:
        return 1.0
    from math import comb
    k = max(wins, losses)  # co najmniej k zwycięstw po stronie dominującej
    return sum(comb(n, t) for t in range(k, n + 1)) / (2 ** n)


def demo_run_once(lam: float, delta: float, rows: int, cols: int,
                  kind: str, edge_thr: float) -> Dict[str, float]:
    """
    Pojedynczy przebieg: liczy J_phi (Φ1/Φ2/Φ3), Align, CR_* i profil (α,β,S,H,Z).
    Używa aktualnego API hma.* (wszystkie funkcje Φ/Ψ/Align biorą próg edge_thr).
    """
    # AST
    ast_raw = hma.ast_deltas(hma.EXAMPLE_SRC)
    ast_l   = hma.compress_ast(ast_raw, lam)

    # Mozaika
    M = hma.build_mosaic(rows=rows, cols=cols, seed=7, kind=kind, edge_thr=edge_thr)

    # Φ – trzy selektory (UWAGA: przekazujemy edge_thr)
    J1, _ = hma.phi_cost(ast_l, M, edge_thr, selector=hma.phi_region_for)
    J2, _ = hma.phi_cost(ast_l, M, edge_thr, selector=hma.phi_region_for_balanced)
    J3, _ = hma.phi_cost(ast_l, M, edge_thr, selector=hma.phi_region_for_entropy)

    # Ψ – feedback z progiem
    ast_after = hma.psi_feedback(ast_l, M, delta, edge_thr)

    # Align – z progiem
    Align = 1.0 - min(1.0, hma.distance_ast_mosaic(ast_after, M, edge_thr))

    # Kompresja AST (jak wcześniej)
    CR_AST = (ast_raw.S + ast_raw.H + max(1, ast_raw.Z)) / max(1, ast_l.S + ast_l.H + max(1, ast_l.Z))

    # Topologia mozaiki (jak wcześniej)
    p_edge = float(np.mean(M.edge > edge_thr))
    CR_TO  = (1.0 / max(1e-6, min(p_edge, 1 - p_edge))) - 1.0

    return dict(J_phi1=J1, J_phi2=J2, J_phi3=J3,
                Align=Align, CR_AST=CR_AST, CR_TO=CR_TO,
                S=ast_l.S, H=ast_l.H, Z=ast_l.Z,
                alpha=ast_l.alpha, beta=ast_l.beta)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(prog="demo_hybrid_mosaic_eval",
                                 description="Demo/benchmark dla hybrydy AST⇄Mozaika (Φ/Ψ)")
    ap.add_argument("--lam", type=float, default=0.60, help="λ – poziom kompresji AST")
    ap.add_argument("--delta", type=float, default=0.25, help="Δ – siła Ψ feedback")
    ap.add_argument("--seeds", type=int, default=100, help="liczba seedów do testu operacyjnego")
    ap.add_argument("--rows", type=int, default=12, help="wiersze mozaiki")
    ap.add_argument("--cols", type=int, default=12, help="kolumny mozaiki")
    ap.add_argument("--kind", type=str, default="grid", choices=["grid", "hex"], help="rodzaj mozaiki")
    ap.add_argument("--edge-thr", type=float, default=hma.EDGE_THR_DEFAULT, help="próg edge dla regionów Φ")
    args = ap.parse_args()

    print("\n=== PROTOKÓŁ DOWODOWY: AST ⇄ Mozaika (Φ/Ψ, ΔS/ΔH/ΔZ, λ/Δ-sweep) ===\n")

    # baseline
    base = demo_run_once(args.lam, args.delta, args.rows, args.cols, args.kind, args.edge_thr)
    print(f"[BASELINE] λ={args.lam:.2f}, Δ={args.delta:.2f}")
    print(json.dumps(base, indent=2))

    # inwarianty
    astA = hma.ast_deltas(hma.EXAMPLE_SRC)
    astB = hma.compress_ast(astA, args.lam)
    M = hma.build_mosaic(args.rows, args.cols, seed=7, kind=args.kind, edge_thr=args.edge_thr)
    inv = hma.invariants_check(astA, astB, M, args.edge_thr)
    print("\n[TESTY INWARIANTÓW / METRYK]")
    for k, v in inv.items():
        print(f"  - {k}: {'PASS' if v else 'FAIL'}")

    # porównanie metod Φ
    print("\n[PORÓWNANIE METOD Φ] (Φ1=heur, Φ2=balanced, Φ3=entropy-fuzzy)")
    aL = hma.compress_ast(astA, args.lam)
    J1, _ = hma.phi_cost(aL, M, args.edge_thr, selector=hma.phi_region_for)
    J2, _ = hma.phi_cost(aL, M, args.edge_thr, selector=hma.phi_region_for_balanced)
    J3, _ = hma.phi_cost(aL, M, args.edge_thr, selector=hma.phi_region_for_entropy)
    imp = (J1 - J2) / max(1e-9, J1) * 100.0
    print(f"  Φ1 (heur):   J_phi = {J1:.6f}")
    print(f"  Φ2 (bal):    J_phi = {J2:.6f}  (improvement vs Φ1: {imp:.2f}%)")
    print(f"  Φ3 (fuzzy):  J_phi = {J3:.6f}  (Δ vs Φ1: {((J3 - J1) / max(1e-9, J1)) * 100:.2f}%)")

    # sweep λ×Δ
    lams: List[float] = [0.0, 0.25, 0.5, 0.75]
    dels: List[float] = [0.0, 0.25, 0.5]
    print("\n[SWEEP λ × Δ]  (Align↑ lepiej, J_phi↓ lepiej, CR_AST↑ = większa kompresja)")
    header = ["λ", "Δ", "Align", "J_phi2", "CR_AST", "CR_TO", "α", "β", "S", "H", "Z"]
    widths = [4, 4, 7, 8, 7, 7, 5, 5, 4, 4, 3]

    def _row(cols): return " | ".join(str(c).ljust(w) for c, w in zip(cols, widths))

    print(_row(header))
    print("-" * (sum(widths) + len(widths) - 1))
    for lam in lams:
        for de in dels:
            r = demo_run_once(lam, de, args.rows, args.cols, args.kind, args.edge_thr)
            print(_row([
                f"{lam:.2f}", f"{de:.2f}", f"{r['Align']:.3f}",
                f"{r['J_phi2']:.4f}", f"{r['CR_AST']:.3f}", f"{r['CR_TO']:.3f}",
                f"{r['alpha']:.2f}", f"{r['beta']:.2f}",
                int(r['S']), int(r['H']), int(r['Z']),
            ]))

    # test operacyjny: Φ2 vs Φ1 na N seedach
    print(f"\n[TEST OPERACYJNY] {args.seeds} seedów — czy Φ2 (balanced) poprawia J_phi vs Φ1?")
    wins = losses = ties = 0
    diffs: List[float] = []
    for seed in range(args.seeds):
        Ms = hma.build_mosaic(args.rows, args.cols, seed=seed, kind=args.kind, edge_thr=args.edge_thr)
        aL = hma.compress_ast(astA, args.lam)
        j1, _ = hma.phi_cost(aL, Ms, args.edge_thr, selector=hma.phi_region_for)
        j2, _ = hma.phi_cost(aL, Ms, args.edge_thr, selector=hma.phi_region_for_balanced)
        d = j1 - j2
        diffs.append(d)
        if d > 0: wins += 1
        elif d < 0: losses += 1
        else: ties += 1
        if (seed + 1) % max(1, args.seeds // 10) == 0:
            print(f"  progress: {seed + 1}/{args.seeds}  | running wins={wins}, losses={losses}, ties={ties}",
                  flush=True)

    mean_diff = float(np.mean(diffs))
    med_diff  = float(np.median(diffs))
    p_sign    = _sign_test_p(wins, losses)

    print(f"  mean(J1-J2) = {mean_diff:.6f}  | median = {med_diff:.6f}")
    print(f"  wins Φ2: {wins}/{wins + losses} (ties={ties}) | sign-test p≈{p_sign:.3g}")

    # JSON summary
    out = dict(
        baseline=dict(lambda_=args.lam, delta=args.delta, **base),
        invariants=inv,
        sweep=dict(lams=lams, deltas=dels),
        op_test=dict(seeds=args.seeds, wins=wins, losses=losses, ties=ties,
                     p_sign=p_sign, mean_improvement=mean_diff, median_improvement=med_diff),
        setup=dict(rows=args.rows, cols=args.cols, kind=args.kind, edge_thr=args.edge_thr),
    )
    print("\n[SUMMARY JSON]")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
