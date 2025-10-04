#!/usr/bin/env python
# glitchlab/analysis/ast_mosaic_analyzer.py
# -*- coding: utf-8 -*-
# Python 3.9+
"""
Lekki analizator AST⇄Mozaika na staged pliki .py:
- AST delty (S/H/Z + alpha/beta)
- prosta "Align" (projekcja na mozaikę siatki 6x6; heurystyka bez NumPy)
- metawarstwa (# komentarze, adnotacje @)
- wynik: słownik per plik + agregaty
"""
from __future__ import annotations
import ast, math, hashlib, json, re
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# ── Heurystyki AST (S/H/Z + α/β) ──────────────────────────────────────────────
def ast_summary(src: str) -> Tuple[int, int, int, float, float, int]:
    """
    Zwraca: S,H,Z,alpha,beta,maxZ
    S/H liczone prostymi regułami jak w poprzednich wariantach; Z = głębokość struktur.
    """
    try:
        tree = ast.parse(src)
    except Exception:
        return 0, 0, 0, 0.5, 0.5, 0

    S = H = Z = 0;
    maxZ = 0

    def walk(a: ast.AST, d: int):
        nonlocal S, H, Z, maxZ
        lab = a.__class__.__name__
        # "Δ-reguły"
        if isinstance(a, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            S += 1;
            H += 1;
            Z += 1
        elif isinstance(a, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
            S += 1;
            Z += 1
        elif isinstance(a, ast.Assign):
            S += 1;
            H += 1
        elif isinstance(a, ast.Call):
            S += 1;
            H += 2
        elif isinstance(a, (ast.Import, ast.ImportFrom)):
            S += 1;
            H += len(getattr(a, 'names', []))
        elif isinstance(a, ast.Name):
            H += 1
        maxZ = max(maxZ, d)
        for ch in ast.iter_child_nodes(a):
            walk(ch, d + 1)
        if isinstance(a, (
        ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.If, ast.For, ast.While, ast.With, ast.Try)):
            Z -= 1

    walk(tree, 0)
    Z = max(0, Z)
    tot = max(1, S + H)
    alpha = float(S) / tot;
    beta = float(H) / tot
    return int(S), int(H), int(Z), alpha, beta, int(maxZ)


# ── Prosta mozaika (grid 6x6) i Align ────────────────────────────────────────
def edge_profile(src: str) -> float:
    """
    Surogat 'edge' – gęstość znaków "strukturalnych" ((),:;{}[]) względem długości.
    """
    if not src: return 0.0
    tokens = "()[]{}:;,"
    c = sum(src.count(t) for t in tokens)
    return min(1.0, c / max(1, len(src) / 20.0))


def mosaic_align(alpha: float, beta: float, edge_p: float, wS=1.0, wH=1.0) -> float:
    """
    Oddalone od zera = gorzej; Align = 1 - distance w [0..1]
    """
    aM = 0.5 + 0.3 * (1.0 - edge_p)  # im mniej edge'ów, tym "S" rośnie
    bM = 1.0 - aM
    dist = wS * abs(alpha - aM) + wH * abs(beta - bM)
    return max(0.0, 1.0 - min(1.0, dist))


# ── Metawarstwa: komentarze i adnotacje ──────────────────────────────────────
_meta_re = re.compile(r'^\s*(#.*|@[\w.]+)', re.M)


def meta_extract(src: str, max_lines=8) -> List[str]:
    hits = _meta_re.findall(src)
    if not hits: return []
    # skracamy i porządkujemy
    out = []
    for h in hits[:max_lines]:
        out.append(h.strip())
    return out


# ── Analiza jednego pliku ────────────────────────────────────────────────────
def analyze_py(path: Path) -> Dict:
    try:
        src = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return dict(path=str(path), ok=False, reason="read_fail")

    S, H, Z, alpha, beta, maxZ = ast_summary(src)
    edge_p = edge_profile(src)
    align = mosaic_align(alpha, beta, edge_p)
    meta = meta_extract(src)

    return dict(
        path=str(path),
        ok=True,
        ast=dict(S=S, H=H, Z=Z, alpha=alpha, beta=beta, maxZ=maxZ),
        mosaic=dict(edge_p=edge_p, align=align),
        meta=meta
    )


# ── Agregacja ────────────────────────────────────────────────────────────────
def aggregate(results: List[Dict]) -> Dict:
    n = len([r for r in results if r.get("ok")])
    if n == 0:
        return dict(files=0, S=0, H=0, avg_align=0.0)
    S = sum(r["ast"]["S"] for r in results if r.get("ok"))
    H = sum(r["ast"]["H"] for r in results if r.get("ok"))
    avg_align = sum(r["mosaic"]["align"] for r in results if r.get("ok")) / float(n)
    return dict(files=n, S=S, H=H, avg_align=avg_align)
