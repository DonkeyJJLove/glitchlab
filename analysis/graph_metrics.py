#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
glitchlab.analysis.graph_metrics — globalne metryki grafowe
Python 3.9 • stdlib only

Wejście: ProjectGraph (analysis.project_graph)
Wyjście: .glx/graphs/metrics.json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from glitchlab.analysis.project_graph import ProjectGraph, build_project_graph
# artefakty
try:
    from glitchlab.io.artifacts import GlxArtifacts  # type: ignore
except Exception:  # pragma: no cover
    class GlxArtifacts:  # type: ignore
        def __init__(self) -> None:
            self.repo_root = Path(__file__).resolve().parents[3]
            self.base = self.repo_root / "glitchlab" / ".glx"
            self.base.mkdir(parents=True, exist_ok=True)

        def write_json(self, rel_name: str, obj) -> Path:
            p = (self.base / rel_name).resolve()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
            return p

        def read_json(self, rel_name: str) -> dict:
            p = (self.base / rel_name).resolve()
            return json.loads(p.read_text(encoding="utf-8"))

af = GlxArtifacts()

VERSION = "v1"

@dataclass
class GraphMetricsConfig:
    directed: bool = True                 # traktuj graf jako skierowany dla PR/betweenness/closeness
    use_weights: bool = True              # użyj wag krawędzi, jeśli dostępne
    pagerank_alpha: float = 0.85
    pagerank_tol: float = 1e-8
    pagerank_max_iter: int = 200
    betw_samples: int = 0                 # 0 => auto
    close_samples: int = 0                # 0 => auto
    lpa_max_iter: int = 20                # Label Propagation (na grafie zsymetryzowanym)
    symmetrize_for_lpa: bool = True

def _nodes_sorted(g: ProjectGraph) -> List[str]:
    return sorted(g.nodes.keys())

def _adjacency(g: ProjectGraph, *, directed: bool, use_weights: bool) -> Tuple[Dict[str, List[Tuple[str, float]]],
                                                                               Dict[str, List[Tuple[str, float]]]]:
    """Zwraca (adj_out, adj_in). Gdy directed=False, adj_in=adj_out (symetryzacja)."""
    adj_out: Dict[str, List[Tuple[str, float]]] = {nid: [] for nid in g.nodes.keys()}
    adj_in: Dict[str, List[Tuple[str, float]]] = {nid: [] for nid in g.nodes.keys()}
    for e in g.edges:
        w = float(e.weight) if use_weights else 1.0
        if e.src not in adj_out or e.dst not in adj_in:
            # węzeł mógł nie być zarejestrowany? (ostrożność)
            adj_out.setdefault(e.src, [])
            adj_in.setdefault(e.dst, [])
        adj_out[e.src].append((e.dst, w))
        adj_in[e.dst].append((e.src, w))
        if not directed:
            adj_out[e.dst].append((e.src, w))
            adj_in[e.src].append((e.dst, w))
    # deterministycznie
    for d in (adj_out, adj_in):
        for k in d:
            d[k].sort(key=lambda t: t[0])
    return adj_out, adj_in

def _degree_and_strength(g: ProjectGraph, adj_out, adj_in) -> Tuple[Dict[str,int],Dict[str,int],Dict[str,int],Dict[str,float],Dict[str,float]]:
    deg_out = {nid: len(adj_out.get(nid, [])) for nid in g.nodes}
    deg_in  = {nid: len(adj_in.get(nid, []))  for nid in g.nodes}
    deg_all = {nid: deg_out.get(nid,0) + deg_in.get(nid,0) for nid in g.nodes}
    str_out = {nid: sum(w for _, w in adj_out.get(nid, [])) for nid in g.nodes}
    str_in  = {nid: sum(w for _, w in adj_in.get(nid, []))  for nid in g.nodes}
    return deg_in, deg_out, deg_all, str_in, str_out

def _pagerank(g: ProjectGraph, adj_out, alpha: float, tol: float, max_iter: int) -> Dict[str, float]:
    nodes = _nodes_sorted(g)
    N = len(nodes)
    if N == 0:
        return {}
    idx = {nid: i for i, nid in enumerate(nodes)}
    pr = [1.0 / N] * N
    out_w_sum = [sum(w for _, w in adj_out[n]) for n in nodes]
    dangling = [i for i, s in enumerate(out_w_sum) if s == 0.0]
    for _ in range(max_iter):
        prev = pr[:]
        # teleport
        base = (1.0 - alpha) / N
        pr = [base] * N
        # rozchodzenie
        # wkład wiszących
        dangling_mass = alpha * sum(prev[i] for i in dangling) / N if dangling else 0.0
        for i in range(N):
            pr[i] += dangling_mass
        for u in nodes:
            u_i = idx[u]
            s = out_w_sum[u_i]
            if s > 0.0:
                mass = alpha * prev[u_i]
                for v, w in adj_out[u]:
                    pr[idx[v]] += mass * (w / s)
        # norma L1
        diff = sum(abs(pr[i] - prev[i]) for i in range(N))
        if diff < tol:
            break
    return {nodes[i]: float(pr[i]) for i in range(N)}

def _bfs_distances(source: str, adj_out) -> Dict[str, int]:
    """Najkrótsze odległości (bez wag) od 'source' w grafie skierowanym."""
    dist: Dict[str, int] = {source: 0}
    q = [source]
    head = 0
    while head < len(q):
        u = q[head]; head += 1
        du = dist[u]
        for v, _w in adj_out.get(u, []):
            if v not in dist:
                dist[v] = du + 1
                q.append(v)
    return dist

def _select_samples(nodes: List[str], req: int) -> List[str]:
    N = len(nodes)
    if N == 0:
        return []
    k = max(1, min(req, N))
    # równomierny, deterministyczny wybór co step
    step = max(1, N // k)
    return [nodes[i] for i in range(0, N, step)][:k]

def _betweenness_approx(g: ProjectGraph, adj_out, adj_in, samples: int) -> Dict[str, float]:
    """Brandes z próbkowaniem źródeł (skierowany, bez wag)."""
    nodes = _nodes_sorted(g)
    N = len(nodes)
    if N == 0:
        return {}
    S = _select_samples(nodes, samples if samples > 0 else max(1, min(N, 32)))
    idx = {nid: i for i, nid in enumerate(nodes)}
    CB = [0.0] * N

    for s in S:
        # jednoczesny BFS do policzenia liczby najkrótszych ścieżek σ oraz poprzedników
        # implementacja „light” (skierowana)
        Q: List[str] = []
        Sstack: List[str] = []
        P: Dict[str, List[str]] = {v: [] for v in nodes}
        sigma: Dict[str, float] = {v: 0.0 for v in nodes}
        dist: Dict[str, int] = {v: -1 for v in nodes}

        sigma[s] = 1.0
        dist[s] = 0
        Q.append(s)
        while Q:
            v = Q.pop(0)
            Sstack.append(v)
            for w, _ in adj_out.get(v, []):
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    Q.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    P[w].append(v)

        delta: Dict[str, float] = {v: 0.0 for v in nodes}
        while Sstack:
            w = Sstack.pop()
            for v in P[w]:
                if sigma[w] > 0.0:
                    delta_v = (sigma[v] / sigma[w]) * (1.0 + delta[w])
                    delta[v] += delta_v
            if w != s:
                CB[idx[w]] += delta[w]

    # skala ~ normalizacja przez liczbę próbek i maksimum (do [0,1])
    scale = 1.0 / max(1.0, len(S))
    CB = [c * scale for c in CB]
    mx = max(CB) if CB else 1.0
    if mx > 0.0:
        CB = [c / mx for c in CB]
    return {nodes[i]: float(CB[i]) for i in range(N)}

def _closeness_harmonic_in(g: ProjectGraph, adj_out, samples: int) -> Dict[str, float]:
    """
    Harmonic closeness (inbound): H(v) = sum_{u != v} 1/d(u,v), liczone przez BFS od próbek u.
    Zwraca wartości dla wszystkich v (bo sumujemy wkłady z próbek).
    """
    nodes = _nodes_sorted(g)
    N = len(nodes)
    if N == 0:
        return {}
    S = _select_samples(nodes, samples if samples > 0 else max(1, min(N, 32)))
    idx = {nid: i for i, nid in enumerate(nodes)}
    H = [0.0] * N

    for s in S:
        dist = _bfs_distances(s, adj_out)  # od s do innych
        for v, d in dist.items():
            if v == s:
                continue
            if d > 0:
                H[idx[v]] += 1.0 / float(d)

    # normalizacja przez liczbę próbek
    scale = 1.0 / max(1.0, len(S))
    H = [h * scale for h in H]
    # dodatkowa normalizacja przez (N-1) żeby zbliżyć do [0,1]
    denom = max(1.0, (N - 1))
    H = [h / 1.0 for h in H]  # już względne do liczby próbek; zostawmy w tej skali
    return {nodes[i]: float(H[i]) for i in range(N)}

def _label_propagation(g: ProjectGraph, *, symmetrize: bool, max_iter: int) -> Dict[str, int]:
    """Klasyczne LPA na grafie zsymetryzowanym (deterministycznie)."""
    nodes = _nodes_sorted(g)
    if not nodes:
        return {}
    # budujemy sąsiedztwo nieskierowane
    adj_und: Dict[str, List[str]] = {nid: [] for nid in nodes}
    for e in g.edges:
        u, v = e.src, e.dst
        if u not in adj_und: adj_und[u] = []
        if v not in adj_und: adj_und[v] = []
        adj_und[u].append(v)
        adj_und[v].append(u)
    for k in adj_und:
        adj_und[k].sort()

    # inicjalnie etykieta = własne id
    label: Dict[str, str] = {nid: nid for nid in nodes}

    for _ in range(max_iter):
        changed = 0
        # deterministyczna kolejność
        for n in nodes:
            counts: Dict[str, int] = {}
            for m in adj_und.get(n, []):
                lab = label[m]
                counts[lab] = counts.get(lab, 0) + 1
            if not counts:
                continue
            # wybierz etykietę o największej liczności (tie-break: mniejszy string)
            max_cnt = max(counts.values())
            best_labels = sorted([lab for lab, c in counts.items() if c == max_cnt])
            new_lab = best_labels[0]
            if new_lab != label[n]:
                label[n] = new_lab
                changed += 1
        if changed == 0:
            break

    # przemapuj etykiety na spójne numery 0..K-1 w deterministycznej kolejności
    uniq = sorted(set(label.values()))
    idmap = {lab: i for i, lab in enumerate(uniq)}
    return {n: idmap[label[n]] for n in nodes}

def _edge_density_meta(g: ProjectGraph) -> float:
    """Gęstość dla grafu skierowanego: m / (n*(n-1)) bez pętli."""
    n = len(g.nodes)
    if n <= 1:
        return 0.0
    m = sum(1 for e in g.edges if e.src != e.dst)
    return float(m) / float(n * (n - 1))

def compute_graph_metrics(g: ProjectGraph, cfg: Optional[GraphMetricsConfig] = None) -> dict:
    cfg = cfg or GraphMetricsConfig()
    nodes = _nodes_sorted(g)

    adj_out, adj_in = _adjacency(g, directed=cfg.directed, use_weights=cfg.use_weights)
    deg_in, deg_out, deg_all, str_in, str_out = _degree_and_strength(g, adj_out, adj_in)
    pr = _pagerank(g, adj_out, cfg.pagerank_alpha, cfg.pagerank_tol, cfg.pagerank_max_iter)
    betw = _betweenness_approx(g, adj_out, adj_in, cfg.betw_samples)
    clos = _closeness_harmonic_in(g, adj_out, cfg.close_samples)
    lpa = _label_propagation(g, symmetrize=cfg.symmetrize_for_lpa, max_iter=cfg.lpa_max_iter)
    density = _edge_density_meta(g)

    metrics = {
        "degree_in": deg_in,
        "degree_out": deg_out,
        "degree_total": deg_all,
        "strength_in": {k: float(v) for k, v in str_in.items()},
        "strength_out": {k: float(v) for k, v in str_out.items()},
        "pagerank": pr,
        "betweenness_approx": betw,
        "closeness_harmonic_in": clos,
        "community_lpa": {k: int(v) for k, v in lpa.items()},
    }

    payload = {
        "version": VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "basis": {
            "graph_version": g.meta.get("graph_version", "v1"),
            "graph_hash": g.meta.get("graph_hash") or g.graph_hash(),
            "nodes": len(g.nodes),
            "edges": len(g.edges),
            "edge_density": density,
        },
        "config": asdict(cfg),
        "metrics": metrics,
    }
    return payload

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def run_cli(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="graph_metrics", description="Globalne metryki grafowe dla ProjectGraph")
    p.add_argument("--use-existing-graph", help="Ścieżka do .glx/graphs/project_graph.json (opcjonalnie)")
    p.add_argument("--alpha", type=float, default=0.85, help="PageRank α")
    p.add_argument("--pr-maxiter", type=int, default=200)
    p.add_argument("--pr-tol", type=float, default=1e-8)
    p.add_argument("--betw-samples", type=int, default=0, help="0=auto (<=32)")
    p.add_argument("--close-samples", type=int, default=0, help="0=auto (<=32)")
    p.add_argument("--lpa-iters", type=int, default=20)
    args = p.parse_args(argv)

    # załaduj lub zbuduj graf
    if args.use_existing_graph:
        # próbujemy odczytać artefakt
        try:
            data = json.loads(Path(args.use_existing_graph).read_text(encoding="utf-8"))
        except Exception:
            data = af.read_json("graphs/project_graph.json")
        repo_root = Path(data.get("meta", {}).get("repo_root", ".")) if isinstance(data, dict) else Path(".")
        g = ProjectGraph.from_json_obj(repo_root=repo_root, data=data)
    else:
        g = build_project_graph()

    cfg = GraphMetricsConfig(
        pagerank_alpha=args.alpha,
        pagerank_tol=args.pr_tol,
        pagerank_max_iter=args.pr_maxiter,
        betw_samples=args.betw_samples,
        close_samples=args.close_samples,
        lpa_max_iter=args.lpa_iters,
    )
    payload = compute_graph_metrics(g, cfg)
    out = af.write_json("graphs/metrics.json", payload)
    print(str(out))
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(run_cli())
