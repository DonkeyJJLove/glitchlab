# glitchlab/core/graph.py
"""
version: 2
kind: module
id: "core-graph"
created_at: "2025-09-11"
name: "glitchlab.core.graph"
author: "GlitchLab v2"
role: "DAG Builder & Metrics Aggregator"
description: >
  Buduje liniowy DAG (nodes/edges) na podstawie steps oraz metryk zapisanych w ctx.cache
  przez pipeline. Agreguje czasy, metryki in/out, diff_stats i wylicza delty (out−in).
  Eksportuje lekki JSON gotowy dla HUD/GUI pod kluczem 'ast/json'.
inputs:
  steps: "list[Step{name:str, params:dict}]"
  cache: "dict (ctx.cache) z kluczami stage/{i}/metrics_in|metrics_out|diff_stats|t_ms"
outputs:
  graph:
    nodes: "list[Node{id,name,params,t_ms,metrics_in,metrics_out,diff_stats,delta?,status}]"
    edges: "list[Edge{src,dst}]"
    meta:  "{steps_count:int, metrics_keys:list[str], has_missing:bool}"
  cache_key: "ast/json"  # miejsce zapisu w ctx.cache (opcjonalnie)
record_model:
  Node:  ["id","name","params","t_ms","metrics_in","metrics_out","diff_stats","delta?","status"]
  Edge:  ["src","dst"]
  Graph: ["nodes[]","edges[]","meta{steps_count,metrics_keys,has_missing}"]
interfaces:
  exports: ["build_graph_from_cache","export_ast_json","build_and_export_graph"]
  depends_on: ["typing"]  # brak ciężkich zależności
  used_by: ["glitchlab.core.pipeline","glitchlab.app","glitchlab.analysis.exporters"]
policy:
  deterministic: true
  side_effects: "opcjonalny zapis do ctx.cache['ast/json']"
constraints:
  - "no SciPy/OpenCV"
  - "JSON-serializowalny output (bez macierzy)"
telemetry:
  metrics: ["delta per metric","t_ms per stage","diff_stats{mean,p95,max}"]
hud:
  channels:
    ast_json: "ast/json"
license: "Proprietary"
---
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, TypedDict


class Node(TypedDict, total=False):
    id: int
    name: str
    params: Dict[str, Any]
    t_ms: float
    metrics_in: Dict[str, float]
    metrics_out: Dict[str, float]
    diff_stats: Dict[str, float]
    delta: Dict[str, float]        # metrics_out - metrics_in (wspólne klucze)
    status: str                    # "ok" | "missing"

class Edge(TypedDict):
    src: int
    dst: int

class Graph(TypedDict):
    nodes: List[Node]
    edges: List[Edge]
    meta: Dict[str, Any]


def _as_float_dict(d: Mapping[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in (d or {}).items():
        try:
            out[k] = float(v)
        except Exception:
            # pomiń wartości nienumeryczne
            continue
    return out


def _compute_delta(m_in: Mapping[str, Any], m_out: Mapping[str, Any]) -> Dict[str, float]:
    a = _as_float_dict(m_in)
    b = _as_float_dict(m_out)
    keys = a.keys() & b.keys()
    return {k: b[k] - a[k] for k in keys}


def build_graph_from_cache(
    steps: List[Mapping[str, Any]],
    cache: Dict[str, Any],
    *,
    attach_delta: bool = True,
) -> Graph:
    """
    Buduje liniowy DAG (i->i+1) korzystając z metryk zapisanych w ctx.cache przez pipeline.
    Nie odczytuje obrazów; wyłącznie liczby i parametry.
    """
    nodes: List[Node] = []
    edges: List[Edge] = []

    n = len(steps)
    for i in range(n):
        s = steps[i]
        name = str(s.get("name", "")).strip()
        params = dict(s.get("params", {}))

        m_in = _as_float_dict(cache.get(f"stage/{i}/metrics_in", {}))
        m_out = _as_float_dict(cache.get(f"stage/{i}/metrics_out", {}))
        diff_stats = _as_float_dict(cache.get(f"stage/{i}/diff_stats", {}))
        t_ms = float(cache.get(f"stage/{i}/t_ms", 0.0)) if f"stage/{i}/t_ms" in cache else 0.0

        node: Node = {
            "id": i,
            "name": name,
            "params": params,
            "t_ms": t_ms,
            "metrics_in": m_in,
            "metrics_out": m_out,
            "diff_stats": diff_stats,
            "status": "ok" if (m_in or m_out or diff_stats or t_ms > 0.0) else "missing",
        }
        if attach_delta:
            node["delta"] = _compute_delta(m_in, m_out)

        nodes.append(node)

        if i < n - 1:
            edges.append({"src": i, "dst": i + 1})

    meta: Dict[str, Any] = {
        "steps_count": n,
        "metrics_keys": sorted(
            set().union(
                *[nodes[i].get("metrics_out", {}).keys() for i in range(n)]  # type: ignore[arg-type]
            )
        ) if n else [],
        "has_missing": any(nd.get("status") == "missing" for nd in nodes),
    }

    return {"nodes": nodes, "edges": edges, "meta": meta}


def export_ast_json(
    graph: Graph,
    ctx_like: Optional[Mapping[str, Any]] = None,
    *,
    cache_key: str = "ast/json",
) -> Dict[str, Any]:
    """
    Zwraca serializowalny JSON (dict). Jeśli podasz obiekt z 'cache' (np. Ctx),
    zapisze również pod wskazanym kluczem.
    """
    # graf jest już JSON-serializowalny (słowniki/liczby/listy)
    if ctx_like is not None:
        cache = None
        # obsłuż zarówno Ctx z atrybutem 'cache', jak i dict zawierający cache
        if hasattr(ctx_like, "cache"):
            cache = getattr(ctx_like, "cache", None)
        elif isinstance(ctx_like, Mapping):
            cache = ctx_like.get("cache")
        if isinstance(cache, dict):
            cache[cache_key] = graph
    return graph


def build_and_export_graph(
    steps: List[Mapping[str, Any]],
    ctx_like: Any,
    *,
    attach_delta: bool = True,
    cache_key: str = "ast/json",
) -> Dict[str, Any]:
    """
    Wygodne połączenie: buduj graf z ctx.cache i zapisz go jako JSON w cache.
    """
    cache: Dict[str, Any]
    if hasattr(ctx_like, "cache"):
        cache = getattr(ctx_like, "cache")
    elif isinstance(ctx_like, Mapping) and "cache" in ctx_like:
        cache = ctx_like["cache"]  # type: ignore[index]
    else:
        raise TypeError("ctx_like must expose a dict 'cache'")

    graph = build_graph_from_cache(steps, cache, attach_delta=attach_delta)
    export_ast_json(graph, ctx_like, cache_key=cache_key)
    return graph
