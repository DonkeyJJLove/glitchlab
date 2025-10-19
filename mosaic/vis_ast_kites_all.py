#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vis_ast_kites_all.py
Produkcja: wizualizacja AST ⇄ Mozaika z "latawcami" (płaszczyzny polityk) dla KAŻDEGO węzła.
- Spójna z algorytmem Φ/Ψ (balanced) z hybrid_ast_mosaic
- Zawiera legendy, opisy osi, podsumowanie profilu mozaiki, zapis do pliku
- CLI do ustawiania parametrów (rows/cols/thr/seed itp.)

Uruchomienie:
  python vis_ast_kites_all.py --rows 6 --cols 6 --edge-thr 0.55 --seed 7 --out out.png
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Import z dwóch możliwych ścieżek (repo może mieć różne drzewo pakietów)
try:
    from glitchlab.app.mosaic.hybrid_ast_mosaic import (
        ast_deltas, build_mosaic_hex, region_ids, centroid,
        phi_region_for_balanced, EXAMPLE_SRC, EDGE_THR_DEFAULT
    )
except Exception:
    from glitchlab.app.mosaic.hybrid_ast_mosaic import (
        ast_deltas, build_mosaic_hex, region_ids, centroid,
        phi_region_for_balanced, EXAMPLE_SRC, EDGE_THR_DEFAULT
    )

# ------------------------- Konfiguracja/Mapowania -----------------------------

# Kolory dla regionów Φ
REGION_COLOR: Dict[str, str] = {
    "edges": "#d33",  # czerwony
    "~edges": "#36c",  # niebieski
    "roi": "#8233cc",  # fiolet
    "all": "#f39c12"  # pomarańcz
}


# 0=L, 1=S, 2=Sel, 3=Stab, 4=Cau, 5=H
def label2coeff(lbl: str) -> int:
    """Dominujący meta-współczynnik dla węzła AST (semantyka zgodna z algorytmem)."""
    if lbl in ("Call", "Expr"):
        return 2  # Sel
    if lbl == "Assign":
        return 3  # Stab
    if lbl in ("FunctionDef", "AsyncFunctionDef", "ClassDef", "Return", "Raise"):
        return 4  # Cau
    if lbl in ("If", "For", "While", "With", "Try"):
        return 5  # H
    return 3  # Fallback: Stab


def label2region(lbl: str, M, thr: float) -> str:
    """Region Φ (balanced) – uwzględnia kwantyle mozaiki."""
    return phi_region_for_balanced(lbl, M, thr)


# ------------------------------ Dane pomocnicze -------------------------------

@dataclass
class Kite:
    """Reprezentacja jednego latawca (płaszczyzny polityki)."""
    node_id: int
    label: str
    coeff_idx: int
    region_kind: str
    verts: List[Tuple[float, float, float]]  # [O, A(node), M(region)]
    color: str


def node_point(n, coeff_idx: int) -> Tuple[float, float, float]:
    """Punkt dla węzła AST w osi dominującego współczynnika."""
    return float(n.depth), float(n.id), float(n.meta[coeff_idx])


def region_point(M, kind: str, thr: float) -> Optional[Tuple[float, float, float]]:
    """Centroid regionu mozaiki + Z jako średnia edge."""
    ids = region_ids(M, kind, thr)
    if not ids:
        return None
    cx, cy = centroid(ids, M)
    cz = float(np.mean([float(M.edge[i]) for i in ids]))
    return float(cx), float(cy), cz


def make_kite(n, M, thr: float) -> Optional[Kite]:
    """Zbuduj latawiec dla węzła AST: O(0,0,0), A(node), M(region)."""
    coeff = label2coeff(n.label)
    kind = label2region(n.label, M, thr)
    rp = region_point(M, kind, thr)
    if rp is None:
        return None
    p0 = (0.0, 0.0, 0.0)
    pa = node_point(n, coeff)
    pm = rp
    verts = [p0, pa, pm]
    color = REGION_COLOR.get(kind, "#777")
    return Kite(node_id=n.id, label=n.label, coeff_idx=coeff, region_kind=kind, verts=verts, color=color)


def mosaic_profile_text(M, thr: float) -> str:
    """Krótki opis profilu mozaiki do legendy."""
    vals = np.asarray(M.edge, dtype=float)
    p_edge = float(np.mean(vals > thr))
    mu = float(np.mean(vals))
    sd = float(np.std(vals))
    return f"edge_thr={thr:.2f} | p(edge)≈{p_edge:.2f} | mean(edge)≈{mu:.2f} ± {sd:.2f}"


# ------------------------------ Rysowanie sceny -------------------------------

def draw_scene(ast, M, thr: float, annotate_n: int = 6, elev: float = 24, azim: float = -55):
    """Rysuje pełną scenę: mozaika, węzły AST, latawce, legendy/opisy."""
    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=elev, azim=azim)
    ax.set_title("AST ⇄ Mozaika — latawce (płaszczyzny polityk) per węzeł • Φ=balanced")

    # 1) Mozaika: półprzezroczysta powierzchnia
    if getattr(M, "hex_centers", None) is not None:
        xs, ys = zip(*M.hex_centers)
    else:
        cols = int(M.cols);
        rows = int(M.rows)
        xs = [i % cols for i in range(rows * cols)]
        ys = [i // cols for i in range(rows * cols)]
    zs = np.asarray(M.edge, dtype=float)
    try:
        ax.plot_trisurf(xs, ys, zs, cmap='viridis', alpha=0.18, linewidth=0.2, antialiased=True)
    except Exception:
        ax.scatter(xs, ys, zs, s=6, alpha=0.3)

    # 2) Latawce per węzeł
    kites: List[Kite] = []
    for n in ast.nodes.values():
        k = make_kite(n, M, thr)
        if k is None:
            continue
        kites.append(k)
        # płaszczyzna
        ax.add_collection3d(
            Poly3DCollection([k.verts], alpha=0.33, facecolor=k.color, edgecolor="#222", linewidths=0.6))
        # punkty A(node) i M(region)
        _, A, Mpt = k.verts
        ax.scatter(*A, c="#2ecc71", s=18, marker="o")  # AST node
        ax.scatter(*Mpt, c="#111111", s=14, marker="^")  # mosaic centroid

    # 3) Adnotacje: N węzłów o najwyższym Z (dominanta)
    if annotate_n > 0 and kites:
        scored = sorted(kites, key=lambda k: k.verts[1][2], reverse=True)[:annotate_n]
        for k in scored:
            _, A, _ = k.verts
            ax.text(A[0], A[1], A[2],
                    f"{k.label}#{k.node_id}\ncoeff={k.coeff_idx}",
                    fontsize=8, color="#1b1b1b", ha="left", va="bottom")

    # 4) Osi i podpisy
    ax.set_xlabel("Depth (AST)")
    ax.set_ylabel("Node ID / Mosaic X")
    ax.set_zlabel("Dominanta meta / Edge")
    ax.grid(False)

    # 5) Panel legendy (po prawej)
    legend_lines = [
        "REGIONY Φ → kolory:",
        f"  edges   → {REGION_COLOR['edges']}",
        f"  ~edges  → {REGION_COLOR['~edges']}",
        f"  roi     → {REGION_COLOR['roi']}",
        f"  all     → {REGION_COLOR['all']}",
        "",
        "META (per węzeł): [L,S,Sel,Stab,Cau,H]",
        "  0=L (liniowość) | 1=S (struktura)",
        "  2=Sel (selektywność) | 3=Stab (stabilność)",
        "  4=Cau (przyczynowość) | 5=H (heterogeniczność)",
        "",
        "Punkt latawca:",
        "  O=(0,0,0), A=(depth,id,meta[coeff]),",
        "  M=(cx,cy,mean(edge_region))",
        "",
        "Profil mozaiki:",
        mosaic_profile_text(M, thr),
    ]
    fig.text(
        0.74, 0.16, "\n".join(legend_lines),
        fontsize=9, family="monospace", va="bottom", ha="left",
        bbox=dict(boxstyle="round", fc="white", ec="#bbb", alpha=0.9)
    )

    # 6) Mini-legenda kolorów + markery
    from matplotlib.lines import Line2D
    handles = []
    for kind, col in REGION_COLOR.items():
        handles.append(Line2D([0], [0], color=col, lw=6, label=kind))
    handles += [
        Line2D([0], [0], marker='o', color='w', label='AST node', markerfacecolor='#2ecc71', markersize=7),
        Line2D([0], [0], marker='^', color='w', label='Mosaic centroid', markerfacecolor='#111111', markersize=6)
    ]
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.02, 0.98))

    plt.tight_layout()
    return fig, ax


# ---------------------------------- CLI --------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Produkcja: AST⇄Mozaika z latawcami (per węzeł) + legendy/opisy")
    ap.add_argument("--rows", type=int, default=6, help="liczba wierszy mozaiki (hex)")
    ap.add_argument("--cols", type=int, default=6, help="liczba kolumn mozaiki (hex)")
    ap.add_argument("--edge-thr", type=float, default=EDGE_THR_DEFAULT, help="próg edge dla regionów Φ")
    ap.add_argument("--seed", type=int, default=7, help="seed dla mozaiki (reprodukowalność)")
    ap.add_argument("--hex-R", type=float, default=1.0, help="promień heksa (skala XY)")
    ap.add_argument("--annotate-n", type=int, default=6, help="ile węzłów AST opisać etykietą")
    ap.add_argument("--view-elev", type=float, default=24.0, help="kąt podniesienia kamery")
    ap.add_argument("--view-azim", type=float, default=-55.0, help="kąt azymutu kamery")
    ap.add_argument("--dpi", type=int, default=160, help="DPI zapisu")
    ap.add_argument("--out", type=str, default="", help="ścieżka wyjściowa (png/svg). Puste = tylko podgląd.")
    args = ap.parse_args()

    # Dane
    ast = ast_deltas(EXAMPLE_SRC)
    M = build_mosaic_hex(args.rows, args.cols, seed=args.seed, R=args.hex_R)

    # Rysuj
    fig, _ = draw_scene(ast, M, thr=args.edge_thr, annotate_n=args.annotate_n,
                        elev=args.view_elev, azim=args.view_azim)

    # Zapis
    if args.out:
        fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
        print(f"[OK] Zapisano wizualizację do: {args.out}")

    # Podgląd (jeśli środowisko pozwala)
    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()
