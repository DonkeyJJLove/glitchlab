# glitchlab/core/astmap.py
"""
version: 2
kind: module
id: "core-astmap"
created_at: "2025-09-11"
name: "glitchlab.core.astmap"
author: "GlitchLab v2" [LINUX][REPO][REFRESH]
role: "Python AST → Graph & Mosaic Projector"
description: >
  Parsuje źródło Pythona do AST, buduje lekki graf semantyczny (funkcje/klasy/połączenia)
  z metrykami złożoności i opcjonalnie rzutuje je na mozaikę jako overlay RGB.
  Eksportuje JSON grafu pod kluczem 'ast/json' dla HUD/APP(gui).
inputs:
  source: {type: "str", desc: "tekst źródłowy Pythona"}
  mosaic: {type: "Mosaic", optional: true, desc: "mapa z glitchlab.core.mosaic"}
  map_spec:
    R: {metric: "complexity", range: [1, 10]}
    G: {metric: "fan_out",   range: [0, 5]}
    B: {metric: "loc",       range: [0, 200]}
outputs:
  graph:
    nodes: "list[{id,name,qualname,kind,metrics{loc,branches,calls,fan_out,fan_in,complexity}}]"
    edges: "list[{src,dst,type:'call'|'contains'}]"
    meta:  "{functions:int, classes:int, nodes:int, edges:int}"
  overlay_rgb?: {type: "uint8 (H,W,3)", desc: "kolorowa projekcja metryk na mozaikę"}
  cache_key: "ast/json"  # miejsce zapisu grafu w ctx.cache (opcjonalnie)
interfaces:
  exports: ["build_ast","ast_to_graph","project_ast_to_mosaic","export_ast_json"]
  depends_on: ["ast","numpy"]
  used_by: ["glitchlab.app","glitchlab.core.mosaic","glitchlab.analysis.exporters","glitchlab.core.graph"]
policy:
  deterministic: true
  side_effects: "opcjonalny zapis do ctx.cache['ast/json']"
constraints:
  - "no SciPy/OpenCV"
  - "output JSON-serializowalny (bez obiektów AST)"
  - "bez ciężkich analiz CFG/SSA"
telemetry:
  node_metrics: ["loc","branches","calls","fan_out","fan_in","complexity"]
hud:
  channels:
    ast_json: "ast/json"
license: "Proprietary"
---
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw

__all__ = [
    "build_ast",
    "ast_to_graph",
    "project_ast_to_mosaic",
    "export_ast_json",
]


# --------------------------------------------------------------------------------------
# AST build
# --------------------------------------------------------------------------------------

def build_ast(source: str) -> ast.AST:
    """Parsuje kod źródłowy Pythona do drzewa AST (bez exec)."""
    if not isinstance(source, str):
        raise TypeError("build_ast: source must be a string")
    return ast.parse(source)


# --------------------------------------------------------------------------------------
# Graf semantyczny
# --------------------------------------------------------------------------------------

@dataclass
class _DefRef:
    node_id: int
    name: str
    kind: str
    ast_node: ast.AST


def _fqname(stack: List[str], name: str) -> str:
    return ".".join([*stack, name]) if stack else name


def _count_subtree_metrics(n: ast.AST) -> Tuple[int, int]:
    """Zwraca (weight, branching) w poddrzewie: weight = liczba node'ów, branching = liczba {If,For,While,Try}."""
    w = 0
    b = 0
    for x in ast.walk(n):
        w += 1
        if isinstance(x, (ast.If, ast.For, ast.While, ast.Try)):
            b += 1
    return w, b


def ast_to_graph(tree: ast.AST) -> Dict[str, Any]:
    """
    Buduje lekki graf:
      nodes: [{id, name, kind, metrics: {weight, branching, fan_in, fan_out}}]
      edges: [{src, dst, type: "calls"|"contains"}]
      meta:  {node_count, edge_count, kinds}
    Uwzględnia funkcje/klasy i relacje wywołań między zdefiniowanymi funkcjami (po nazwie).
    """
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    defs_by_name: Dict[str, _DefRef] = {}
    scope_stack: List[str] = []

    # 1) Pass: zarejestruj definicje klas i funkcji (wraz z metrykami wag/gałęzi)
    class _DefCollector(ast.NodeVisitor):
        def generic_visit(self, node: ast.AST):
            if isinstance(node, ast.ClassDef):
                fq = _fqname(scope_stack, node.name)
                nid = len(nodes)
                w, br = _count_subtree_metrics(node)
                nodes.append({
                    "id": nid,
                    "name": fq,
                    "kind": "class",
                    "metrics": {"weight": int(w), "branching": int(br), "fan_in": 0, "fan_out": 0},
                })
                defs_by_name[fq] = _DefRef(nid, fq, "class", node)
                # containment: parent -> child
                if scope_stack:
                    parent = defs_by_name.get(".".join(scope_stack))
                    if parent:
                        edges.append({"src": parent.node_id, "dst": nid, "type": "contains"})
                scope_stack.append(node.name)
                super().generic_visit(node)
                scope_stack.pop()
                return

            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fq = _fqname(scope_stack, node.name)
                nid = len(nodes)
                w, br = _count_subtree_metrics(node)
                nodes.append({
                    "id": nid,
                    "name": fq,
                    "kind": "function",
                    "metrics": {"weight": int(w), "branching": int(br), "fan_in": 0, "fan_out": 0},
                })
                defs_by_name[fq] = _DefRef(nid, fq, "function", node)
                if scope_stack:
                    parent = defs_by_name.get(".".join(scope_stack))
                    if parent:
                        edges.append({"src": parent.node_id, "dst": nid, "type": "contains"})
                scope_stack.append(node.name)
                super().generic_visit(node)
                scope_stack.pop()
                return

            super().generic_visit(node)

    _DefCollector().visit(tree)

    # 2) Pass: zbierz krawędzie 'calls' między funkcjami zdefiniowanymi w tym module
    #    (w oparciu o aktualny kontekst funkcji; proste rozpoznawanie ast.Name)
    current_fn_stack: List[str] = []

    class _CallCollector(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef):
            current_fn_stack.append(_fqname(scope_stack, node.name))
            scope_stack.append(node.name)
            self.generic_visit(node)
            scope_stack.pop()
            current_fn_stack.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
            current_fn_stack.append(_fqname(scope_stack, node.name))
            scope_stack.append(node.name)
            self.generic_visit(node)
            scope_stack.pop()
            current_fn_stack.pop()

        def visit_ClassDef(self, node: ast.ClassDef):
            scope_stack.append(node.name)
            self.generic_visit(node)
            scope_stack.pop()

        def visit_Call(self, node: ast.Call):
            if not current_fn_stack:
                # połączenia interesują nas tylko wewnątrz funkcji/metody
                self.generic_visit(node)
                return
            caller = defs_by_name.get(current_fn_stack[-1])
            # Uproszczenie: tylko nazwy bez kwalifikatorów (ast.Name). Atrybuty pomijamy.
            callee_name = None
            if isinstance(node.func, ast.Name):
                callee_name = node.func.id
                # dopasuj preferencyjnie definicję w tym samym zakresie (FQ) albo globalną
                fq_local = _fqname(scope_stack[:-1], callee_name) if scope_stack else callee_name
                ref = defs_by_name.get(fq_local) or defs_by_name.get(callee_name)
            else:
                ref = None
            if caller and ref:
                edges.append({"src": caller.node_id, "dst": ref.node_id, "type": "calls"})
            self.generic_visit(node)

    _CallCollector().visit(tree)

    # 3) fan_in/fan_out
    fan_out = {n["id"]: 0 for n in nodes}
    fan_in = {n["id"]: 0 for n in nodes}
    for e in edges:
        if e.get("type") == "calls":
            fan_out[e["src"]] += 1
            fan_in[e["dst"]] += 1
    for n in nodes:
        n["metrics"]["fan_out"] = int(fan_out[n["id"]])
        n["metrics"]["fan_in"] = int(fan_in[n["id"]])

    kinds = sorted(list({n["kind"] for n in nodes}))
    graph: Dict[str, Any] = {
        "nodes": nodes,
        "edges": edges,
        "meta": {"node_count": len(nodes), "edge_count": len(edges), "kinds": kinds},
    }
    return graph


# --------------------------------------------------------------------------------------
# Projekcja na mozaikę
# --------------------------------------------------------------------------------------

def _mosaic_label_raster(mosaic: Mapping[str, Any]) -> np.ndarray:
    lab = mosaic.get("raster")
    if not isinstance(lab, np.ndarray):
        raise ValueError("project_ast_to_mosaic: invalid mosaic['raster']")
    return lab


def _normalize_values(vals: List[float]) -> Tuple[float, float]:
    if not vals:
        return 0.0, 1.0
    vmin = float(min(vals))
    vmax = float(max(vals))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def _fill_cell(overlay: np.ndarray, lab: np.ndarray, cid: int, color01: Tuple[float, float, float]) -> None:
    mask = (lab == cid)
    if not np.any(mask):
        return
    r = int(np.clip(color01[0] * 255.0 + 0.5, 0, 255))
    g = int(np.clip(color01[1] * 255.0 + 0.5, 0, 255))
    b = int(np.clip(color01[2] * 255.0 + 0.5, 0, 255))
    overlay[mask] = (r, g, b)


def project_ast_to_mosaic(
        graph: Mapping[str, Any],
        mosaic: Mapping[str, Any],
        *,
        map_spec: Optional[Mapping[str, Tuple[str, Tuple[float, float]]]] = None,
) -> np.ndarray:
    """
    Projektuje metryki węzłów AST na komórki mozaiki (1 węzeł -> 1 komórka, w kolejności).
    Domyślna projekcja:
      R <- branching, G <- weight, B <- fan_out  (każdy znormalizowany do [0,1] po grafie)
    """
    H, W = mosaic.get("size", (0, 0))
    if not (isinstance(H, int) and isinstance(W, int) and H > 0 and W > 0):
        raise ValueError("project_ast_to_mosaic: invalid mosaic['size']")
    lab = _mosaic_label_raster(mosaic)
    cells = mosaic.get("cells", [])
    overlay = np.zeros((H, W, 3), dtype=np.uint8)

    nodes = list(graph.get("nodes", []))
    if not nodes or not cells:
        return overlay

    # wyciągnij metryki
    weights = [float(n.get("metrics", {}).get("weight", 0.0)) for n in nodes]
    branchings = [float(n.get("metrics", {}).get("branching", 0.0)) for n in nodes]
    fanouts = [float(n.get("metrics", {}).get("fan_out", 0.0)) for n in nodes]

    w_lo, w_hi = _normalize_values(weights)
    b_lo, b_hi = _normalize_values(branchings)
    f_lo, f_hi = _normalize_values(fanouts)

    if map_spec is None:
        map_spec = {
            "R": ("branching", (b_lo, b_hi)),
            "G": ("weight", (w_lo, w_hi)),
            "B": ("fan_out", (f_lo, f_hi)),
        }

    def norm(val: float, lo: float, hi: float) -> float:
        if hi <= lo:
            return 0.0
        t = (val - lo) / (hi - lo)
        return float(np.clip(t, 0.0, 1.0))

    # przydział 1:1 (z podcięciem), kolejność wg nodes
    count = min(len(nodes), len(cells))
    for i in range(count):
        n = nodes[i]
        m = n.get("metrics", {})
        # wartości
        r_name, (r_lo, r_hi) = map_spec.get("R", ("branching", (b_lo, b_hi)))
        g_name, (g_lo, g_hi) = map_spec.get("G", ("weight", (w_lo, w_hi)))
        b_name, (b1_lo, b1_hi) = map_spec.get("B", ("fan_out", (f_lo, f_hi)))
        r = norm(float(m.get(r_name, 0.0)), r_lo, r_hi)
        g = norm(float(m.get(g_name, 0.0)), g_lo, g_hi)
        b = norm(float(m.get(b_name, 0.0)), b1_lo, b1_hi)
        _fill_cell(overlay, lab, int(cells[i]["id"]), (r, g, b))

    return overlay


# --------------------------------------------------------------------------------------
# Eksport JSON do ctx.cache
# --------------------------------------------------------------------------------------

def export_ast_json(
        graph: Mapping[str, Any],
        ctx_like: Optional[Mapping[str, Any]] = None,
        *,
        cache_key: str = "ast/json",
) -> Dict[str, Any]:
    """
    Zwraca dict gotowy do serializacji JSON. Jeśli ctx_like posiada 'cache' (dict),
    zapisuje kopię pod kluczem cache_key.
    """

    # JSON-friendly kopia (upewnij się, że typy są serializowalne)
    def _jsonify(obj: Any) -> Any:
        if isinstance(obj, (str, int, float, type(None), bool)):
            return obj
        if isinstance(obj, list):
            return [_jsonify(x) for x in obj]
        if isinstance(obj, dict):
            return {str(k): _jsonify(v) for k, v in obj.items()}
        # wszystko inne – zamień na string
        return str(obj)

    data = {
        "nodes": _jsonify(graph.get("nodes", [])),
        "edges": _jsonify(graph.get("edges", [])),
        "meta": _jsonify(graph.get("meta", {})),
    }

    if ctx_like is not None:
        cache = getattr(ctx_like, "cache", None) if hasattr(ctx_like, "cache") else ctx_like.get(
            "cache")  # type: ignore
        if isinstance(cache, dict):
            cache[cache_key] = data
    return data
