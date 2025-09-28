# scratch3.py — DOWODOWY PROTOTYP: AST ⇄ Mozaika z ΔS/ΔH/ΔZ, λ/Δ-sweep,
#                wieloma wariantami Φ i testami własności metrycznych/inwariantów
# Python 3.9+, deps: numpy

from __future__ import annotations
import ast, math, json, itertools, statistics as stats
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np

# ───────────────────────────────────────────────────────────────────────────────
# 0) PARAMS
# ───────────────────────────────────────────────────────────────────────────────
R, C = 6, 6
EDGE_THR = 0.55
W = dict(wS=1.0, wH=1.0, wZ=0.4)  # wagi dystansu dla Align
RNG = np.random.default_rng(0)

EXAMPLE_SRC = r"""
import os, sys
from pathlib import Path

def hello(msg: str) -> str:
    x = msg.upper()
    print(x)
    return x

def main():
    p = Path('.')
    s = hello("Hi")
    if p.exists():
        print(s)
    return 0

if __name__ == "__main__":
    main()
"""


# ───────────────────────────────────────────────────────────────────────────────
# 1) AST → ΔS/ΔH/ΔZ
# ───────────────────────────────────────────────────────────────────────────────
@dataclass
class AstNode:
    id: int
    label: str
    depth: int
    parent: Optional[int]
    children: List[int] = field(default_factory=list)
    meta: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=float))  # L,S,Sel,Stab,Cau,H


@dataclass
class AstSummary:
    S: int;
    H: int;
    Z: int;
    maxZ: int
    alpha: float;
    beta: float
    nodes: Dict[int, AstNode]
    labels: List[str]


def _meta_for(label: str, depth: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed & 0xFFFFFFFF)
    L, S, Sel, Stab, Cau, H = rng.uniform(0.35, 0.85, size=6)
    if label in ("FunctionDef", "ClassDef"): Stab = max(Stab, 0.8); Sel = max(Sel, 0.6)
    if label in ("If", "While", "For"):       Sel = max(Sel, 0.75); Cau = max(Cau, 0.7)
    if label in ("Call", "Expr"):            L = max(L, 0.6)
    if label in ("Assign",):                Stab = max(Stab, 0.7)
    return np.array([L, S, Sel, Stab, Cau, H], dtype=float)


def ast_deltas(src: str) -> AstSummary:
    tree = ast.parse(src)
    nodes: Dict[int, AstNode] = {}
    S = H = Z = 0;
    maxZ = 0;
    nid = 0

    def add(a: ast.AST, depth: int, parent: Optional[int]) -> int:
        nonlocal nid, S, H, Z, maxZ
        i = nid;
        nid += 1
        lab = a.__class__.__name__
        n = AstNode(i, lab, depth, parent)
        n.meta = _meta_for(lab, depth, seed=hash((lab, depth)))
        nodes[i] = n
        if parent is not None: nodes[parent].children.append(i)

        # Δ-reguły (skrócone)
        if isinstance(a, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            S += 1; H += 1; Z += 1; maxZ = max(maxZ, depth)
        elif isinstance(a, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
            S += 1; Z += 1;   maxZ = max(maxZ, depth)
        elif isinstance(a, ast.Assign):
            S += 1; H += 1
        elif isinstance(a, ast.Call):
            S += 1; H += 2
        elif isinstance(a, (ast.Import, ast.ImportFrom)):
            S += 1; H += len(a.names)
        elif isinstance(a, ast.Name):
            H += 1

        for ch in ast.iter_child_nodes(a): add(ch, depth + 1, i)
        if isinstance(a, (
        ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.If, ast.For, ast.While, ast.With, ast.Try)):
            Z -= 1
        return i

    add(tree, 0, None)
    S = int(S);
    H = int(H);
    Z = int(max(Z, 0))
    tot = max(1, S + H)
    alpha = S / tot;
    beta = H / tot
    return AstSummary(S, H, Z, maxZ, alpha, beta, nodes, [n.label for n in nodes.values()])


def compress_ast(summary: AstSummary, lam: float) -> AstSummary:
    labels = summary.labels
    leaf_ratio = (labels.count("Name") + labels.count("Constant")) / max(1, len(labels))
    S_leaf = int(summary.S * 0.35 * leaf_ratio)
    H_leaf = int(summary.H * 0.35 * leaf_ratio)
    S2 = max(0, summary.S - int(round(lam * S_leaf)))
    H2 = max(0, summary.H - int(round(lam * H_leaf)))
    Z2 = int(round((1 - lam) * summary.Z + lam * math.ceil(max(1, summary.maxZ) / 2)))
    tot = max(1, S2 + H2)
    return AstSummary(S2, H2, Z2, summary.maxZ, S2 / tot, H2 / tot, summary.nodes, summary.labels)


# ───────────────────────────────────────────────────────────────────────────────
# 2) Mozaika i metryka D_M
# ───────────────────────────────────────────────────────────────────────────────
@dataclass
class Mosaic:
    rows: int;
    cols: int
    edge: np.ndarray
    ssim: np.ndarray
    roi: np.ndarray


def build_mosaic(rows=R, cols=C, seed=7) -> Mosaic:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:rows, 0:cols]
    diag = 1.0 - np.abs(xx - yy) / max(rows, cols)
    edge = np.clip(0.45 + 0.5 * diag + 0.06 * rng.standard_normal((rows, cols)), 0, 1).reshape(-1)
    ssim = np.ones(rows * cols)
    roi = np.zeros(rows * cols)
    r0, r1 = int(0.30 * rows), int(0.62 * rows);
    c0, c1 = int(0.30 * cols), int(0.62 * cols)
    for r in range(r0, r1):
        for c in range(c0, c1):
            roi[r * cols + c] = 1.0
    return Mosaic(rows, cols, edge, ssim, roi)


def region_ids(M: Mosaic, kind: str, thr=EDGE_THR) -> List[int]:
    if kind == "edges":  return [i for i, v in enumerate(M.edge) if v > thr]
    if kind == "~edges": return [i for i, v in enumerate(M.edge) if v <= thr]
    if kind == "roi":    return [i for i, v in enumerate(M.roi) if v > 0.5]
    if kind == "all":    return list(range(M.rows * M.cols))
    return []


def tile_dist(i: int, j: int, M: Mosaic, alpha=1.0, beta=0.7, gamma=0.5) -> float:
    x1, y1 = i % M.cols, i // M.cols;
    x2, y2 = j % M.cols, j // M.cols
    geo = math.hypot(x1 - x2, y1 - y2)
    feat = abs(float(M.edge[i]) - float(M.edge[j]))
    label_pen = 1.0 if ((M.edge[i] > EDGE_THR) != (M.edge[j] > EDGE_THR)) else 0.0
    return alpha * geo + beta * feat + gamma * label_pen


def D_M(S: List[int], T: List[int], M: Mosaic, max_match: int = 8) -> float:
    if not S and not T: return 0.0
    if not S or not T:  return float('inf')  # brak porównania
    k = min(len(S), len(T), max_match)  # bezpiecznik na permutacje
    S2, T2 = S[:k], T[:k]
    best = float('inf')
    for perm in itertools.permutations(T2):
        cost = 0.0
        for i, j in zip(S2, perm): cost += tile_dist(i, j, M)
        if cost < best: best = cost
        if best == 0.0: break
    return best


# ───────────────────────────────────────────────────────────────────────────────
# 3) Φ/Ψ — TRZY METODY Φ do porównania + Ψ feedback
# ───────────────────────────────────────────────────────────────────────────────
def phi_region_for(label: str) -> str:
    if label in ("Call", "Expr"):            return "edges"
    if label in ("Assign",):                return "~edges"
    if label in ("If", "For", "While", "With", "Return"): return "all"
    if label in ("FunctionDef", "ClassDef"): return "roi"
    return "~edges"


def phi_region_for_balanced(label: str, M: Mosaic) -> str:
    # wariant Φ2: balansuje klasy wg proporcji globalnych
    p_edge = float(np.mean(M.edge > EDGE_THR))
    if label in ("Call", "Expr"):            return "edges" if p_edge <= 0.65 else "~edges"
    if label in ("Assign",):                return "~edges" if p_edge >= 0.35 else "edges"
    if label in ("FunctionDef", "ClassDef"): return "roi"
    return "all"


def phi_region_for_entropy(label: str, M: Mosaic) -> str:
    # wariant Φ3: gdy granica nieostra (blisko progu), daj "all"
    def near_thr(x): return abs(x - EDGE_THR) <= 0.05

    fuzzy = float(np.mean([near_thr(v) for v in M.edge])) > 0.25
    if fuzzy: return "all"
    return phi_region_for(label)


def centroid(ids: List[int], M: Mosaic) -> Tuple[float, float]:
    if not ids: return (M.cols * 0.5, M.rows * 0.5)
    xs = np.array([i % M.cols for i in ids], float);
    ys = np.array([i // M.cols for i in ids], float)
    return float(xs.mean()), float(ys.mean())


def phi_cost(ast: AstSummary, M: Mosaic, selector) -> Tuple[float, Dict[int, Dict]]:
    details = {};
    total = 0.0
    for n in ast.nodes.values():
        kind = selector(n.label, M) if selector.__code__.co_argcount == 2 else selector(n.label)
        ids = region_ids(M, kind)
        if not ids:
            details[n.id] = dict(kind=kind, cost=0.0, centroid=(None, None));
            continue
        alt = region_ids(M, "edges" if kind == "~edges" else "~edges")
        if not alt: alt = ids[:]  # unik
        alt = alt[:len(ids)] if len(alt) >= len(ids) else (alt + ids[:len(ids) - len(alt)])
        cost = max(0.0, D_M(ids, alt, M))
        details[n.id] = dict(kind=kind, cost=cost, centroid=centroid(ids, M))
        total += cost
    N = max(1, len(ast.nodes))
    return total / float(N), details


def psi_feedback(ast: AstSummary, M: Mosaic, delta: float) -> AstSummary:
    if delta <= 1e-9: return ast
    nodes = ast.nodes
    for n in nodes.values():
        kind = phi_region_for(n.label)
        ids = region_ids(M, kind)
        if not ids: continue
        ed = np.array([M.edge[i] for i in ids], float)
        psi = np.array([
            float(1.0 - ed.mean()),
            float(0.5 + 0.5 * ed.std()),
            float(min(1.0, 0.5 + ed.mean())),
            float(1.0 - ed.std()),
            float(min(1.0, 0.3 + 0.7 * ed.mean())),
            float(0.4 + 0.5 * ed.std())
        ])
        n.meta = (1.0 - delta) * n.meta + delta * psi
    S, H, Z = ast.S, ast.H, ast.Z
    tot = max(1, S + H)
    return AstSummary(S, H, Z, ast.maxZ, S / tot, H / tot, nodes, ast.labels)


# ───────────────────────────────────────────────────────────────────────────────
# 4) Align/Costs + testy „dowodowe”
# ───────────────────────────────────────────────────────────────────────────────
def mosaic_profile(M: Mosaic) -> Tuple[int, int, int, float, float]:
    S = R + C
    H = int(np.sum(M.edge > EDGE_THR))
    Z = 1
    tot = max(1, S + H)
    return S, H, Z, S / tot, H / tot


def distance_ast_mosaic(ast: AstSummary, M: Mosaic, w=W) -> float:
    _, _, _, aM, bM = mosaic_profile(M)
    return (w['wS'] * abs(ast.alpha - aM) +
            w['wH'] * abs(ast.beta - bM) +
            w['wZ'] * abs(ast.Z / max(1, ast.maxZ) - 0.0))


def invariants_check(astA: AstSummary, astB: AstSummary, M: Mosaic) -> Dict[str, bool]:
    # I1: α+β≈1 dla obu stanów
    I1A = abs(astA.alpha + astA.beta - 1.0) < 1e-9
    I1B = abs(astB.alpha + astB.beta - 1.0) < 1e-9
    # I2: metryka D_M: nieujemna, symetryczna, identyczność rozróżnialnych
    roi = region_ids(M, "roi");
    top = region_ids(M, "all")
    I2a = D_M(roi, roi, M) == 0.0
    I2b = D_M(roi, top[:len(roi)], M) >= 0.0
    I2c = abs(D_M(roi, top[:len(roi)], M) - D_M(top[:len(roi)], roi, M)) < 1e-9
    # I3: kompresja nie zwiększa sumy (S+H+Z)
    sumA = astA.S + astA.H + max(1, astA.Z)
    sumB = astB.S + astB.H + max(1, astB.Z)
    I3 = sumB <= sumA
    return {
        "I1_alpha_plus_beta_eq_1": (I1A and I1B),
        "I2_metric_nonneg_sym_identity": (I2a and I2b and I2c),
        "I3_compression_monotone": I3
    }


def nice_row(cols, widths):
    return " | ".join(str(c).ljust(w) for c, w in zip(cols, widths))


# ───────────────────────────────────────────────────────────────────────────────
# 5) MAIN: porównania metod, sweep, testy
# ───────────────────────────────────────────────────────────────────────────────
def run_once(lam: float, delta: float) -> Dict[str, float]:
    ast_raw = ast_deltas(EXAMPLE_SRC)
    ast_l = compress_ast(ast_raw, lam)
    M = build_mosaic(R, C, seed=7)
    # trzy warianty Φ
    J1, _ = phi_cost(ast_l, M, selector=phi_region_for)
    J2, _ = phi_cost(ast_l, M, selector=phi_region_for_balanced)
    J3, _ = phi_cost(ast_l, M, selector=phi_region_for_entropy)
    ast_after = psi_feedback(ast_l, M, delta)
    Align = 1.0 - min(1.0, distance_ast_mosaic(ast_after, M))
    p_edge = float(np.mean(M.edge > EDGE_THR))
    CR_AST = (ast_raw.S + ast_raw.H + max(1, ast_raw.Z)) / max(1, ast_l.S + ast_l.H + max(1, ast_l.Z))
    CR_TO = (1.0 / max(1e-6, min(p_edge, 1 - p_edge))) - 1.0
    return dict(J_phi1=J1, J_phi2=J2, J_phi3=J3, Align=Align, CR_AST=CR_AST, CR_TO=CR_TO,
                S=ast_l.S, H=ast_l.H, Z=ast_l.Z, alpha=ast_l.alpha, beta=ast_l.beta)


def main():
    print("\n=== PROTOKÓŁ DOWODOWY: AST ⇄ Mozaika (Φ/Ψ, ΔS/ΔH/ΔZ, λ/Δ-sweep) ===\n")

    # 1) Jedno przebiegnięcie referencyjne
    lam0, del0 = 0.60, 0.25
    res0 = run_once(lam0, del0)
    print("[BASELINE] λ=%.2f, Δ=%.2f" % (lam0, del0))
    print(json.dumps(res0, indent=2))

    # 2) Testy metryczne/inwarianty
    astA = ast_deltas(EXAMPLE_SRC)
    astB = compress_ast(astA, lam0)
    M = build_mosaic(R, C, seed=7)
    inv = invariants_check(astA, astB, M)
    print("\n[TESTY INWARIANTÓW / METRYK]")
    for k, v in inv.items():
        print(f"  - {k}: {'PASS' if v else 'FAIL'}")

    # 3) Porównanie metod Φ (mniejsze J_phi lepsze)
    print("\n[PORÓWNANIE METOD Φ] (Φ1=heur, Φ2=balanced, Φ3=entropy-fuzzy)")
    J1, _ = phi_cost(compress_ast(astA, lam0), M, selector=phi_region_for)
    J2, _ = phi_cost(compress_ast(astA, lam0), M, selector=phi_region_for_balanced)
    J3, _ = phi_cost(compress_ast(astA, lam0), M, selector=phi_region_for_entropy)
    print(f"  Φ1 (heur):   J_phi = {J1:.6f}")
    print(f"  Φ2 (bal):    J_phi = {J2:.6f}  (improvement vs Φ1: {((J1 - J2) / max(1e-9, J1)) * 100:.2f}%)")
    print(f"  Φ3 (fuzzy):  J_phi = {J3:.6f}  (degradation vs Φ1: {((J3 - J1) / max(1e-9, J1)) * 100:.2f}%)")

    # 4) Sweep po λ i Δ
    lams = [0.0, 0.25, 0.5, 0.75]
    dels = [0.0, 0.25, 0.5]
    print("\n[SWEEP λ × Δ]  (Align↑ lepiej, J_phi↓ lepiej, CR_AST↑ = większa kompresja)")
    header = ["λ", "Δ", "Align", "J_phi1", "J_phi2", "J_phi3", "CR_AST", "CR_TO", "α", "β", "S", "H", "Z"]
    widths = [4, 4, 7, 8, 8, 8, 8, 8, 5, 5, 4, 4, 3]
    print(nice_row(header, widths))
    print("-" * sum(widths) + "-" * (len(widths) - 1))
    for lam in lams:
        for de in dels:
            r = run_once(lam, de)
            row = [f"{lam:.2f}", f"{de:.2f}", f"{r['Align']:.3f}",
                   f"{r['J_phi1']:.4f}", f"{r['J_phi2']:.4f}", f"{r['J_phi3']:.4f}",
                   f"{r['CR_AST']:.3f}", f"{r['CR_TO']:.3f}",
                   f"{r['alpha']:.2f}", f"{r['beta']:.2f}",
                   r['S'], r['H'], r['Z']]
            print(nice_row(row, widths))

    # 5) Test operacyjny (100 seedów): czy Φ2 poprawia J_phi vs Φ1?
    print("\n[TEST OPERACYJNY] 100 losowych seedów mozaiki — czy Φ2 (balanced) poprawia J_phi vs Φ1?")
    diffs = []
    wins = 0
    losses = 0
    zeros = 0
    for seed in range(100):
        Mx = build_mosaic(R, C, seed=seed)
        aL = compress_ast(astA, lam0)
        J1, _ = phi_cost(aL, Mx, selector=phi_region_for)
        J2, _ = phi_cost(aL, Mx, selector=phi_region_for_balanced)
        d = J1 - J2
        diffs.append(d)
        if d > 0:
            wins += 1
        elif d < 0:
            losses += 1
        else:
            zeros += 1
        if (seed + 1) % 10 == 0:
            print(f"  progress: {seed + 1}/100  | running wins={wins}, losses={losses}, ties={zeros}", flush=True)

    mean_diff = float(np.mean(diffs))
    med_diff = float(np.median(diffs))

    # prosta p-wartość testu znaku (binomial, p=0.5)
    from math import comb
    # używamy tylko rozstrzygnięć (bez remisów) – klasyczny test znaku
    n_eff = wins + losses
    if n_eff == 0:
        p_sign = 1.0
    else:
        k = max(wins, losses)  # "co najmniej tyle zwycięstw" dla strony dominującej
        p_sign = sum(comb(n_eff, t) for t in range(k, n_eff + 1)) / (2 ** n_eff)

    print(f"  mean(J1-J2) = {mean_diff:.6f}  | median = {med_diff:.6f}")
    print(f"  wins Φ2: {wins}/{n_eff} (ties={zeros}) | sign-test p≈{p_sign:.3g}")
    # prosta p-wartość znaku (binomial, p=0.5)
    from math import comb
    p = sum(comb(100, k) for k in range(pos, 101)) / (2 ** 100)
    print(f"  sign-test p≈{p:.3g}  (mniejsze → istotne na korzyść Φ2)")

    # 6) Podsumowanie (JSON pod CI)
    print("\n[SUMMARY JSON]")
    print(json.dumps(dict(
        baseline=dict(lam=lam0, delta=del0, **res0),
        invariants=inv,
        sweep=dict(lams=lams, deltas=dels),
        hypothesis=dict(mean_improvement=mean_diff, median_improvement=med_diff,
                        wins=pos, losses=neg, sign_test_p=p)
    ), indent=2))


if __name__ == "__main__":
    main()
