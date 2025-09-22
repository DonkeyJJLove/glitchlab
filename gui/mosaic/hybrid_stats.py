# -*- coding: utf-8 -*-
"""
glitchlab/mosaic/hybrid_stats.py
Statystyczna ocena hybrydy AST⇄Mozaika:
- porównanie selektorów Φ (Φ1 vs Φ2) na wielu seedach,
- bootstrap CI dla średniej i mediany różnic,
- efekt Cliffa (|δ| interpretacja siły),
- sweep Pareto (Align vs J_phi2) z kontrolą CR_TO,
- JSON-raport + CLI.

Wymaga: glitchlab.mosaic.hybrid_ast_mosaic jako hma (Twoja ostatnia wersja).
"""

from __future__ import annotations
import json, argparse, math
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

import hybrid_ast_mosaic as hma


# ─────────────────────────────────────────────────────────────────────────────
# Pomocnicze statystyki
# ─────────────────────────────────────────────────────────────────────────────

def sign_test_p(wins: int, losses: int) -> float:
    """Dwustronny test znaku (dokładny binomial, p=0.5)."""
    n = wins + losses
    if n == 0:
        return 1.0
    from math import comb
    k = max(wins, losses)
    return sum(comb(n, t) for t in range(k, n + 1)) / (2 ** n)


def cliffs_delta(xs: List[float]) -> float:
    """
    Efekt Cliffa δ dla różnic D = J1-J2.
    δ = ( #par > 0  -  #par < 0 ) / (n*n)  ~ w [−1,1]
    Interpretacja (Romano & Wolf):
      |δ| < 0.147: negligible, <0.33: small, <0.474: medium, else: large
    """
    x = np.asarray(xs, float)
    n = len(x)
    if n == 0:
        return 0.0
    # policz porównania do zera
    gt = np.count_nonzero(x > 0)
    lt = np.count_nonzero(x < 0)
    return float((gt - lt) / max(1, n))


def bootstrap_ci(data: List[float], iters: int = 2000, q: Tuple[float, float] = (0.025, 0.975),
                 stat="mean", seed: int = 17) -> Tuple[float, float]:
    """Nieparametryczny bootstrap CI dla średniej lub mediany."""
    rng = np.random.default_rng(seed)
    x = np.asarray(data, float)
    if x.size == 0:
        return (0.0, 0.0)
    stats = []
    for _ in range(iters):
        samp = rng.choice(x, size=x.size, replace=True)
        if stat == "mean":
            s = float(np.mean(samp))
        else:
            s = float(np.median(samp))
        stats.append(s)
    a, b = np.quantile(stats, q)
    return float(a), float(b)


def summarize_diff(diffs: List[float]) -> Dict[str, float]:
    """Pakiet metryk na listę różnic (J1-J2)."""
    wins = int(np.sum(np.asarray(diffs) > 0))
    losses = int(np.sum(np.asarray(diffs) < 0))
    ties = int(np.sum(np.asarray(diffs) == 0))
    p = sign_test_p(wins, losses)
    mean_, med_ = float(np.mean(diffs)) if diffs else 0.0, float(np.median(diffs)) if diffs else 0.0
    ci_mean = bootstrap_ci(diffs, stat="mean")
    ci_med = bootstrap_ci(diffs, stat="median")
    delta = cliffs_delta(diffs)
    return dict(
        wins=wins, losses=losses, ties=ties, p_sign=p,
        mean_diff=mean_, mean_ci_low=ci_mean[0], mean_ci_high=ci_mean[1],
        median_diff=med_, median_ci_low=ci_med[0], median_ci_high=ci_med[1],
        cliffs_delta=delta
    )


# ─────────────────────────────────────────────────────────────────────────────
# Porównanie Φ1 vs Φ2 na seedach
# ─────────────────────────────────────────────────────────────────────────────

def compare_phi_on_seeds(rows: int, cols: int, kind: str, thr: float, lam: float, seeds: int) -> Dict:
    astA = hma.ast_deltas(hma.EXAMPLE_SRC)
    aL = hma.compress_ast(astA, lam)
    diffs = []
    per_seed = []
    for sd in range(seeds):
        M = hma.build_mosaic(rows, cols, seed=sd, kind=kind, edge_thr=thr)
        j1, _ = hma.phi_cost(aL, M, thr, selector=hma.phi_region_for)
        j2, _ = hma.phi_cost(aL, M, thr, selector=hma.phi_region_for_balanced)
        d = j1 - j2
        diffs.append(d)
        per_seed.append(dict(seed=sd, J_phi1=j1, J_phi2=j2, diff=d))
    summary = summarize_diff(diffs)
    return dict(summary=summary, by_seed=per_seed)


# ─────────────────────────────────────────────────────────────────────────────
# Pareto sweep (Align vs J_phi2) z kontrolą CR_TO
# ─────────────────────────────────────────────────────────────────────────────

def pareto_front(points: List[Dict], xkey: str, ykey: str) -> List[Dict]:
    """
    Zwraca punkty niezdominowane (minimalizacja ykey, maksymalizacja xkey).
    """
    pts = sorted(points, key=lambda r: (r[xkey], -r[ykey]), reverse=True)  # sort Align desc, J asc
    front = []
    best_y = float("inf")
    for r in pts:
        y = r[ykey]
        if y < best_y:
            front.append(r)
            best_y = y
    # rosnąco po J
    return sorted(front, key=lambda r: r[ykey])


def sweep_pareto(rows: int, cols: int, kind: str, thr: float,
                 lams: List[float], deltas: List[float], kappa_ab: float,
                 crto_max: float = 20.0) -> Dict:
    """
    Generuje siatkę λ×Δ i wybiera Pareto-front (Align↑, J_phi2↓),
    odrzucając warianty z CR_TO > crto_max (higiena progu).
    """
    pts = []
    for lam in lams:
        for de in deltas:
            r = hma.run_once(lam, de, rows, cols, thr, mosaic_kind=kind, kappa_ab=kappa_ab)
            if r["CR_TO"] <= crto_max:
                pts.append(dict(lambda_=lam, delta_=de, Align=r["Align"], J_phi2=r["J_phi2"],
                                CR_TO=r["CR_TO"], CR_AST=r["CR_AST"], alpha=r["alpha"], beta=r["beta"]))
    front = pareto_front(pts, xkey="Align", ykey="J_phi2")
    return dict(points=pts, pareto=front)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_cli():
    p = argparse.ArgumentParser(prog="hybrid_stats",
                                description="Statystyczny benchmark hybrydy AST⇄Mozaika")
    p.add_argument("--rows", type=int, default=12)
    p.add_argument("--cols", type=int, default=12)
    p.add_argument("--kind", choices=["grid", "hex"], default="grid")
    p.add_argument("--edge-thr", type=float, default=hma.EDGE_THR_DEFAULT)
    p.add_argument("--lam", type=float, default=0.60)
    p.add_argument("--deltas", type=float, nargs="+", default=[0.0, 0.25, 0.5])
    p.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75])
    p.add_argument("--kappa-ab", type=float, default=hma.KAPPA_AB_DEFAULT)
    p.add_argument("--seeds", type=int, default=100)
    p.add_argument("--crto-max", type=float, default=20.0, help="odrzuć punkty z CR_TO > this")
    p.add_argument("--json", action="store_true", help="wypisz JSON")
    return p


def main():
    args = build_cli().parse_args()

    cmp_res = compare_phi_on_seeds(
        rows=args.rows, cols=args.cols, kind=args.kind, thr=args.edge_thr,
        lam=args.lam, seeds=args.seeds
    )

    pareto = sweep_pareto(
        rows=args.rows, cols=args.cols, kind=args.kind, thr=args.edge_thr,
        lams=args.lambdas, deltas=args.deltas, kappa_ab=args.kappa_ab,
        crto_max=args.crto_max
    )

    out = dict(
        setup=dict(rows=args.rows, cols=args.cols, kind=args.kind, edge_thr=args.edge_thr,
                   lam=args.lam, deltas=args.deltas, lambdas=args.lambdas,
                   kappa_ab=args.kappa_ab, seeds=args.seeds, crto_max=args.crto_max),
        phi_compare=cmp_res,
        pareto=pareto
    )

    print("\n=== Hybrid AST⇄Mosaic — Stats ===\n")
    s = cmp_res["summary"]
    print(f"[Φ2 vs Φ1] wins={s['wins']}, losses={s['losses']}, ties={s['ties']}, "
          f"p≈{s['p_sign']:.3g}")
    print(f"  mean ΔJ={s['mean_diff']:.4f}  95% CI [{s['mean_ci_low']:.4f}, {s['mean_ci_high']:.4f}]")
    print(f"  median ΔJ={s['median_diff']:.4f}  95% CI [{s['median_ci_low']:.4f}, {s['median_ci_high']:.4f}]")
    print(f"  Cliff's δ={s['cliffs_delta']:.3f}  "
          f"(~{'negligible' if abs(s['cliffs_delta']) < 0.147 else 'small' if abs(s['cliffs_delta']) < 0.33 else 'medium' if abs(s['cliffs_delta']) < 0.474 else 'large'})")

    print("\n[Pareto: Align↑, J_phi2↓ | ograniczenie CR_TO ≤ %.1f]" % args.crto_max)
    for r in pareto["pareto"]:
        print(f"  λ={r['lambda_']:.2f} Δ={r['delta_']:.2f} | Align={r['Align']:.3f} | "
              f"J_phi2={r['J_phi2']:.4f} | CR_TO={r['CR_TO']:.2f} | α={r['alpha']:.2f} β={r['beta']:.2f}")

    if args.json:
        print("\n[JSON]")
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
