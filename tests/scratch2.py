# scratch2.py — PROTOTYP: AST⇄Mozaika z ΔS/ΔH/ΔZ, λ-kompresją i kosztem J
# Python 3.9+, dependencies: only numpy (standard lib: ast, math, dataclasses, itertools, json)

from __future__ import annotations
import ast, math, json, itertools
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
import numpy as np

# ───────────────────────────────────────────────────────────────────────────────
# 0) USTAWIENIA (możesz zmieniać)
# ───────────────────────────────────────────────────────────────────────────────
R, C = 6, 6  # rozmiar mozaiki (grid demo; hexy w kolejnej iteracji GUI)
EDGE_THR = 0.55  # próg krawędzi: edges vs ~edges
LAMBDA = 0.6  # λ: poziom kompresji AST (0=pełny detal, 1=grupy/supergraf)
DELTA = 0.25  # Δ: siła feedbacku Ψ (Mozaika→AST; modyfikuje meta-wektory)
W = dict(wS=1.0, wH=1.0, wZ=0.4)  # wagi dystansu α/β/Z

RNG = np.random.default_rng(0)

# Przykładowy kod (możesz podmienić)
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
# 1) AST → ΔS/ΔH/ΔZ i meta-węzły
# ───────────────────────────────────────────────────────────────────────────────

@dataclass
class AstNode:
    id: int
    label: str
    depth: int
    parent: Optional[int]
    children: List[int] = field(default_factory=list)
    meta: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=float))  # (L,S,Sel,Stab,Cau,H)


@dataclass
class AstSummary:
    S: int;
    H: int;
    Z: int;
    maxZ: int
    alpha: float;
    beta: float
    nodes: Dict[int, AstNode];
    labels: List[str]


def _meta_for(label: str, depth: int, seed: int) -> np.ndarray:
    # Heurystyki meta (stabilne deterministycznie)
    rng = np.random.default_rng(seed & 0xFFFFFFFF)
    L, S, Sel, Stab, Cau, H = rng.uniform(0.35, 0.85, size=6)
    if label in ("FunctionDef", "ClassDef"): Stab = max(Stab, 0.8); Sel = max(Sel, 0.6)
    if label in ("If", "While", "For"):      Sel = max(Sel, 0.75); Cau = max(Cau, 0.7)
    if label in ("Call", "Expr"):            L = max(L, 0.6)
    if label in ("Assign",):                 Stab = max(Stab, 0.7)
    return np.array([L, S, Sel, Stab, Cau, H], dtype=float)


def ast_deltas(src: str) -> AstSummary:
    tree = ast.parse(src)
    nodes: Dict[int, AstNode] = {}
    nid = 0
    S = H = Z = 0
    maxZ = 0

    def add(a: ast.AST, depth: int, parent: Optional[int]) -> int:
        nonlocal nid, S, H, Z, maxZ
        i = nid;
        nid += 1
        lab = a.__class__.__name__
        node = AstNode(i, lab, depth, parent)
        node.meta = _meta_for(lab, depth, seed=hash((lab, depth)))
        nodes[i] = node
        if parent is not None:
            nodes[parent].children.append(i)

        # Δ-reguły (szkic z Twojej tabeli)
        if isinstance(a, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            S += 1;
            H += 1;
            Z += 1;
            maxZ = max(maxZ, depth)
        elif isinstance(a, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
            S += 1;
            Z += 1;
            maxZ = max(maxZ, depth)
        elif isinstance(a, ast.Assign):
            S += 1;
            H += 1
        elif isinstance(a, ast.Call):
            S += 1;
            H += 2  # ref do f + uses argów (przybliżenie)
        elif isinstance(a, (ast.Import, ast.ImportFrom)):
            S += 1;
            H += len(a.names)
        elif isinstance(a, ast.Name):
            H += 1
        else:
            # drobnica… minimalny wkład semantyczny
            H += 0

        for ch in ast.iter_child_nodes(a):
            add(ch, depth + 1, i)
        if isinstance(a, (
        ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.If, ast.For, ast.While, ast.With, ast.Try)):
            Z -= 1  # exit scope
        return i

    add(tree, 0, None)
    maxZ = max(maxZ, 0)

    S = int(S);
    H = int(H);
    Z = int(max(Z, 0))
    tot = max(1, S + H)
    alpha = S / tot;
    beta = H / tot
    labels = [n.label for n in nodes.values()]
    return AstSummary(S, H, Z, maxZ, alpha, beta, nodes, labels)


# λ-kompresja (uogólnienie): redukujemy „detal” w S/H zgodnie z λ
def compress_ast(summary: AstSummary, lam: float) -> AstSummary:
    # model: S' = S - lam * S_leaf_est, H' = H - lam * H_leaf_est, Z' = round((1-lam)*Z + lam*ceil(Z/2))
    # estymacja „liści”: przybliżenie proporcją Name/Constant/Load/Store
    labels = summary.labels
    leaf_ratio = max(0.0, min(1.0, labels.count("Name") / max(1, len(labels)) + labels.count("Constant") / max(1,
                                                                                                               len(labels))))
    S_leaf = int(summary.S * 0.35 * leaf_ratio)
    H_leaf = int(summary.H * 0.35 * leaf_ratio)
    S2 = max(0, summary.S - int(round(lam * S_leaf)))
    H2 = max(0, summary.H - int(round(lam * H_leaf)))
    Z2 = int(round((1 - lam) * summary.Z + lam * math.ceil(max(1, summary.maxZ) / 2)))
    tot = max(1, S2 + H2)
    return AstSummary(S2, H2, Z2, summary.maxZ, S2 / tot, H2 / tot, summary.nodes, summary.labels)


# ───────────────────────────────────────────────────────────────────────────────
# 2) Mozaika (grid demo) + regiony + D_M
# ───────────────────────────────────────────────────────────────────────────────

@dataclass
class Mosaic:
    rows: int;
    cols: int
    edge: np.ndarray  # (N,) in [0,1]
    ssim: np.ndarray  # (N,)
    roi: np.ndarray  # (N,)


def build_mosaic(rows=R, cols=C, seed=7) -> Mosaic:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:rows, 0:cols]
    diag = 1.0 - np.abs(xx - yy) / max(rows, cols)
    edge = np.clip(0.45 + 0.5 * diag + 0.06 * rng.standard_normal((rows, cols)), 0, 1).reshape(-1)
    ssim = np.ones(rows * cols)
    roi = np.zeros(rows * cols)
    r0, r1 = int(0.30 * rows), int(0.62 * rows)
    c0, c1 = int(0.30 * cols), int(0.62 * cols)
    for r in range(r0, r1):
        for c in range(c0, c1):
            roi[r * cols + c] = 1.0
    return Mosaic(rows, cols, edge, ssim, roi)


def region_ids(M: Mosaic, kind: str, thr=EDGE_THR) -> List[int]:
    if kind == "edges":   return [i for i, v in enumerate(M.edge) if v > thr]
    if kind == "~edges":  return [i for i, v in enumerate(M.edge) if v <= thr]
    if kind == "roi":     return [i for i, v in enumerate(M.roi) if v > 0.5]
    if kind == "all":     return list(range(M.rows * M.cols))
    return []


def tile_dist(i: int, j: int, M: Mosaic, alpha=1.0, beta=0.7, gamma=0.5) -> float:
    x1, y1 = i % M.cols, i // M.cols
    x2, y2 = j % M.cols, j // M.cols
    geo = math.hypot(x1 - x2, y1 - y2)
    feat = abs(float(M.edge[i]) - float(M.edge[j]))
    label_pen = 1.0 if ((M.edge[i] > EDGE_THR) != (M.edge[j] > EDGE_THR)) else 0.0
    return alpha * geo + beta * feat + gamma * label_pen


def D_M(S: List[int], T: List[int], M: Mosaic) -> float:
    if not S or not T: return 0.0
    if len(S) != len(T):
        # dopasuj rozmiary przez próbkowanie z powtórzeniami najbliższych
        k = min(len(S), len(T))
        S = S[:k];
        T = T[:k]
    best = float("inf")
    for perm in itertools.permutations(T):
        cost = 0.0
        for i, j in zip(S, perm):
            cost += tile_dist(i, j, M)
        best = min(best, cost)
        if best == 0.0: break
    return best


# ───────────────────────────────────────────────────────────────────────────────
# 3) Sprzężenia Φ/Ψ
# ───────────────────────────────────────────────────────────────────────────────

# Reguły Φ: przypisanie typów węzłów do regionów (heurystyka)
def phi_region_for(label: str) -> str:
    if label in ("Call", "Expr"):            return "edges"
    if label in ("Assign",):                 return "~edges"
    if label in ("If", "For", "While", "With", "Return"): return "all"
    if label in ("FunctionDef", "ClassDef"):  return "roi"
    return "~edges"


def centroid(ids: List[int], M: Mosaic) -> Tuple[float, float]:
    if not ids:
        return (M.cols * 0.5, M.rows * 0.5)
    xs = np.array([i % M.cols for i in ids], dtype=float)
    ys = np.array([i // M.cols for i in ids], dtype=float)
    return float(xs.mean()), float(ys.mean())


def phi_cost(ast: AstSummary, M: Mosaic) -> Tuple[float, Dict[int, Dict]]:
    # dla każdego węzła oblicz punkt 0 i koszt dopasowania do „własnego” regionu
    details: Dict[int, Dict] = {}
    total = 0.0
    for n in ast.nodes.values():
        kind = phi_region_for(n.label)
        ids = region_ids(M, kind)
        if not ids:
            details[n.id] = dict(kind=kind, cost=0.0, centroid=(None, None));
            continue
        # dopasowanie: porównaj centroid bieżącego regionu z centroidem „sąsiedniego” (kontrola separacji)
        cx, cy = centroid(ids, M)
        # budujemy „kontr-region” jako komplement o tej samej wielkości (używamy prostego doboru)
        alt = region_ids(M, "edges" if kind == "~edges" else "~edges")
        alt = alt[:len(ids)] if len(alt) >= len(ids) else alt + ids[:len(ids) - len(alt)]
        # koszt Φ jako earth-mover-like między regionem a alternatywą (im dalej, tym lepiej rozdzielone klasy)
        cost = max(0.0, D_M(ids, alt, M))
        details[n.id] = dict(kind=kind, cost=cost, centroid=(cx, cy))
        total += cost
    # normalizacja przez liczbę węzłów
    N = max(1, len(ast.nodes))
    return total / float(N), details


# Ψ: feedback – podbij meta zależnie od regionu (np. gdy region „edges”, rośnie selektywność i kauzalność)
def psi_feedback(ast: AstSummary, M: Mosaic, delta: float) -> AstSummary:
    if delta <= 1e-9: return ast
    nodes = ast.nodes
    for n in nodes.values():
        kind = phi_region_for(n.label)
        ids = region_ids(M, kind)
        if not ids: continue
        ed = np.array([M.edge[i] for i in ids], dtype=float)
        # prosty ψ(region): [L,S,Sel,Stab,Cau,H]
        psi = np.array([
            float(1.0 - ed.mean()),  # L: mniej krawędzi → większa lokalność
            float(0.5 + 0.5 * ed.std()),  # S: wariancja → „skala”/złożoność
            float(min(1.0, 0.5 + ed.mean())),  # Sel: więcej edge → większa selektywność
            float(1.0 - ed.std()),  # Stab: niski rozrzut → większa stabilność
            float(min(1.0, 0.3 + 0.7 * ed.mean())),  # Cau: krawędzie dają efekt
            float(0.4 + 0.5 * ed.std())  # H: entropia ~ rozrzut
        ], dtype=float)
        n.meta = (1.0 - delta) * n.meta + delta * psi
    # ΔS/ΔH/ΔZ nie zmieniamy tu (to miękkie meta), ale możemy przeliczyć α/β dla wizualnych decyzji
    S, H, Z = ast.S, ast.H, ast.Z
    tot = max(1, S + H)
    return AstSummary(S, H, Z, ast.maxZ, S / tot, H / tot, nodes, ast.labels)


# ───────────────────────────────────────────────────────────────────────────────
# 4) Dystanse i koszty zbieżności
# ───────────────────────────────────────────────────────────────────────────────

def mosaic_profile(M: Mosaic) -> Tuple[int, int, int, float, float]:
    # syntetyczny profil mozaiki w (S,H,Z): S~liczba spójnych bloków (tu: rząd/kolumna), H~udział edges, Z~1
    # to tylko „profil analityczny” na potrzeby wspólnej metryki α/β/Z
    S = R + C
    H = int(np.sum(M.edge > EDGE_THR))
    Z = 1
    tot = max(1, S + H)
    return S, H, Z, S / tot, H / tot


def distance_ast_mosaic(ast: AstSummary, M: Mosaic, w=W) -> float:
    _, _, _, alphaM, betaM = mosaic_profile(M)
    return (w['wS'] * abs(ast.alpha - alphaM) +
            w['wH'] * abs(ast.beta - betaM) +
            w['wZ'] * abs(ast.Z / max(1, ast.maxZ) - 0.0))  # Z: docelowo różne bucket’y; tu prosty target=0


# ───────────────────────────────────────────────────────────────────────────────
# 5) GŁÓWNY PRZEBIEG
# ───────────────────────────────────────────────────────────────────────────────

def main():
    # AST → Δ
    ast_raw = ast_deltas(EXAMPLE_SRC)
    ast_l = compress_ast(ast_raw, LAMBDA)

    # Mozaika
    M = build_mosaic(R, C, seed=7)

    # Φ: koszt i szczegóły
    J_phi, phi_details = phi_cost(ast_l, M)

    # Ψ: feedback (miękki update meta)
    ast_after = psi_feedback(ast_l, M, DELTA)

    # Align / Całkowity koszt
    Align3D = 1.0 - min(1.0, distance_ast_mosaic(ast_after, M))  # im większy, tym lepiej
    J_total = 0.5 * J_phi + 0.5 * (1.0 - Align3D)

    # „kompresja” (wskaźnik informacyjny)
    CR_AST = (ast_raw.S + ast_raw.H + max(1, ast_raw.Z)) / max(1, ast_l.S + ast_l.H + max(1, ast_l.Z))
    # topologia mozaiki (im większa dominanta klasy edges vs ~edges, tym większy CR_TO)
    p_edge = float(np.mean(M.edge > EDGE_THR))
    CR_TO = (1.0 / max(1e-6, min(p_edge, 1 - p_edge))) - 1.0  # rośnie gdy klasa dominuje

    out = {
        "AST_raw": dict(S=ast_raw.S, H=ast_raw.H, Z=ast_raw.Z, alpha=ast_raw.alpha, beta=ast_raw.beta),
        "AST_lambda": dict(S=ast_l.S, H=ast_l.H, Z=ast_l.Z, alpha=ast_l.alpha, beta=ast_l.beta),
        "CR_AST": CR_AST,
        "CR_TO": CR_TO,
        "Align3D": Align3D,
        "J_phi": J_phi,
        "J_total": J_total,
        "sample_phi": {k: {"kind": v["kind"], "cost": round(v["cost"], 3),
                           "centroid": tuple(round(c, 2) if c is not None else None for c in v["centroid"])}
                       for k, v in list(phi_details.items())[:8]},
    }

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
