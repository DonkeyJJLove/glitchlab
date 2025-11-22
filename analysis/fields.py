#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
glitchlab.analysis.fields — rejestr „pól operacyjnych” dla metasoczewek
Python 3.9 • stdlib only

Poprawki:
- Uniwersalny resolver ścieżek artefaktów (_glx_path) — bez wymagania GlxArtifacts.path().
- Bez zależności od ProjectGraph.from_json_obj (ładujemy JSON → prosty graf).
- resolve_field przyjmuje **_ignored (kompatybilność z wywołaniami z presetów/reguł).
- Pola sim_shz / dist_shz (pseudo-odległość SHZ z wagami i percentylową referencją).
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple, Iterable, Union

# ── Artefakty (.glx/*) z fallbackiem ─────────────────────────────────────────
try:
    from glitchlab.io.artifacts import GlxArtifacts  # type: ignore
except Exception:  # pragma: no cover
    class GlxArtifacts:  # type: ignore
        def __init__(self, repo_root: Optional[Path] = None) -> None:
            base_root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
            if (base_root / "glitchlab").exists() and (base_root / "glitchlab").is_dir():
                base_root = base_root / "glitchlab"
            self.repo_root = base_root.resolve()
            self.base = (self.repo_root / ".glx").resolve()
            self.base.mkdir(parents=True, exist_ok=True)
        def read_text(self, rel_name: str) -> str:
            p = (self.base / rel_name).resolve()
            return p.read_text(encoding="utf-8")
        def read_json(self, rel_name: str) -> dict:
            return json.loads(self.read_text(rel_name))
        def write_text(self, rel_name: str, text: str) -> Path:
            p = (self.base / rel_name).resolve()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")
            return p
        def write_json(self, rel_name: str, obj) -> Path:
            return self.write_text(rel_name, json.dumps(obj, ensure_ascii=False, indent=2))

# ── Importy projektowe z łagodnym fallbackiem ────────────────────────────────
try:
    from glitchlab.analysis.project_graph import build_project_graph  # type: ignore
except Exception:  # pragma: no cover
    try:
        from analysis.project_graph import build_project_graph  # type: ignore
    except Exception:
        build_project_graph = None  # type: ignore

try:
    from glitchlab.analysis.field_cache import FieldCache  # type: ignore
except Exception:  # pragma: no cover
    try:
        from analysis.field_cache import FieldCache  # type: ignore
    except Exception:
        FieldCache = None  # type: ignore

try:
    from glitchlab.analysis.ast_index import ast_summary_of_file  # type: ignore
except Exception:  # pragma: no cover
    try:
        from analysis.ast_index import ast_summary_of_file  # type: ignore
    except Exception:
        ast_summary_of_file = None  # type: ignore

__all__ = ["FieldSpec", "FieldsRegistry", "resolve_field"]

# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FieldSpec:
    name: str
    normalize: bool = True
    aggregate: str = "mean"
    blend: Optional[List[Dict[str, Union[str, float]]]] = None
    note: str = ""
    params: Dict[str, Any] = field(default_factory=dict)  # dla sim_shz/dist_shz itd.

# ── Prosty model grafu (na potrzeby pól) ─────────────────────────────────────
@dataclass(frozen=True)
class _Node:
    id: str
    kind: str
    label: str
    meta: Dict[str, Any]

@dataclass(frozen=True)
class _Edge:
    src: str
    dst: str
    kind: str

class _SimpleGraph:
    def __init__(self, nodes: Dict[str, _Node], edges: List[_Edge], meta: Dict[str, Any]) -> None:
        self.nodes = nodes
        self.edges = edges
        self.meta = meta

# ── Pomocnicze I/O i ścieżki ─────────────────────────────────────────────────
def _safe_read_json(af: GlxArtifacts, rel: str) -> Optional[dict]:
    try:
        return af.read_json(rel)
    except Exception:
        return None

def _glx_dir_from_af(af: GlxArtifacts, repo_root: Path) -> Path:
    """
    Najbezpieczniej wydobyć katalog .glx z obiektu af (różne warianty interfejsu),
    a w ostateczności repo_root/.glx.
    """
    # popularne atrybuty
    for name in ("base", "artifacts_dir", "dir", "root"):
        if hasattr(af, name):
            try:
                p = Path(getattr(af, name))
                if p.exists() or True:
                    return p.resolve()
            except Exception:
                pass
    # popularne metody
    for name in ("get_dir", "ensure_glx_dir", "get_glx_dir", "ensure_artifacts_dir", "artifacts_dir"):
        if hasattr(af, name):
            try:
                fn = getattr(af, name)
                glx = fn() if callable(fn) else fn
                return Path(glx).resolve()
            except Exception:
                pass
    # fallback
    return (repo_root / ".glx").resolve()

def _glx_path(af: GlxArtifacts, rel: str, repo_root: Path) -> Path:
    """
    Zwraca Path do .glx/<rel> niezależnie od implementacji GlxArtifacts.
    """
    # jeżeli obiekt ma .path(rel) – skorzystaj
    if hasattr(af, "path"):
        try:
            return Path(getattr(af, "path")(rel)).resolve()  # type: ignore[misc]
        except Exception:
            pass
    # w przeciwnym razie połącz katalog .glx z rel
    base = _glx_dir_from_af(af, repo_root)
    return (base / rel).resolve()

def _payload_to_graph(payload: Dict[str, Any]) -> _SimpleGraph:
    nmap: Dict[str, _Node] = {}
    nodes = payload.get("nodes") or []
    if isinstance(nodes, dict):
        items = [{"id": k, **(v or {})} for k, v in nodes.items()]
    else:
        items = list(nodes)
    for it in items:
        nid = str(it.get("id", ""))
        if not nid:
            continue
        nmap[nid] = _Node(
            id=nid,
            kind=str(it.get("kind", "default")),
            label=str(it.get("label", nid)),
            meta=dict(it.get("meta") or {}),
        )
    edges_list: List[_Edge] = []
    for e in payload.get("edges") or []:
        try:
            edges_list.append(_Edge(str(e.get("src")), str(e.get("dst")), str(e.get("kind", "link"))))
        except Exception:
            continue
    meta = dict(payload.get("meta") or {})
    return _SimpleGraph(nmap, edges_list, meta)

def _try_build_graph(repo_root: Path) -> Optional[_SimpleGraph]:
    if build_project_graph is None:
        return None
    try:
        g = build_project_graph(repo_root)
        nodes = [
            {"id": n.id, "kind": getattr(n, "kind", "default"), "label": getattr(n, "label", n.id), "meta": getattr(n, "meta", {})}
            for n in getattr(g, "nodes", {}).values()
        ] if isinstance(getattr(g, "nodes", None), dict) else list(getattr(g, "nodes", []))
        edges = [{"src": e.src, "dst": e.dst, "kind": getattr(e, "kind", "link")} for e in getattr(g, "edges", [])]
        payload = {"nodes": nodes, "edges": edges, "meta": getattr(g, "meta", {})}
        return _payload_to_graph(payload)
    except Exception:
        return None

# ── Normalizacja i agregacja ─────────────────────────────────────────────────
def _minmax_norm(d: Dict[str, float]) -> Dict[str, float]:
    if not d:
        return {}
    vals = list(d.values())
    mn, mx = min(vals), max(vals)
    if mx <= mn:
        return {k: 0.0 for k in d.keys()}
    scale = mx - mn
    return {k: (float(v) - mn) / scale for k, v in d.items()}

def _aggregate(values: Iterable[float], mode: str) -> float:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return 0.0
    m = (mode or "mean").strip().lower()
    if m == "sum":
        return float(sum(vals))
    if m == "max":
        return float(max(vals))
    return float(sum(vals) / len(vals))

# ── Indeksy relacji (module↔file, file↔func) ─────────────────────────────────
class _Index:
    def __init__(self, G: _SimpleGraph) -> None:
        self.G = G
        self.files_by_module: Dict[str, List[str]] = {}
        self.funcs_by_file: Dict[str, List[str]] = {}
        self.file_path_by_node: Dict[str, str] = {}
        self._build()
    def _build(self) -> None:
        for nid, n in self.G.nodes.items():
            if n.kind == "file":
                p = str(n.meta.get("path") or "")
                if p:
                    self.file_path_by_node[nid] = p
        for e in self.G.edges:
            s = self.G.nodes.get(e.src)
            d = self.G.nodes.get(e.dst)
            if not s or not d:
                continue
            if e.kind == "define":
                if s.kind == "module" and d.kind == "file":
                    self.files_by_module.setdefault(s.id, []).append(d.id)
                elif s.kind == "file" and d.kind == "func":
                    self.funcs_by_file.setdefault(s.id, []).append(d.id)

# ── Główna klasa rejestru pól ────────────────────────────────────────────────
class FieldsRegistry:
    def __init__(self, repo_root: Optional[Path] = None) -> None:
        self.repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
        self.af = GlxArtifacts(self.repo_root)
        payload = _safe_read_json(self.af, "graphs/project_graph.json")
        if payload:
            self.G = _payload_to_graph(payload)
        else:
            built = _try_build_graph(self.repo_root)
            self.G = built if built else _SimpleGraph({}, [], {})
        self.idx = _Index(self.G)
        self.fc = FieldCache(self.repo_root) if FieldCache else None
        self._coverage = _safe_read_json(self.af, "tests/coverage.json") or _safe_read_json(self.af, "quality/coverage.json")
        self._smells = _safe_read_json(self.af, "quality/code_smells.json")
        self._delta = _safe_read_json(self.af, "delta_report.json")
        self._ast_snapshot = _safe_read_json(self.af, "graphs/ast_index.json")
        self._cache: Dict[str, Dict[str, float]] = {}

    def available(self) -> Dict[str, str]:
        base = {
            "degree": "continuous",
            "in_degree": "continuous",
            "out_degree": "continuous",
            "pagerank": "continuous",
            "betweenness_approx": "continuous",
            "closeness_harm": "continuous",
            "community_label": "categorical",
            "connection_intensity": "continuous",
            "edge_density": "continuous",
            "ast_s": "continuous",
            "ast_h": "continuous",
            "ast_z": "continuous",
            "code_smells": "continuous",
            "test_coverage": "continuous",
            "churn": "continuous",
            "recentness": "continuous",
            "sim_shz": "continuous",
            "dist_shz": "continuous",
            "blend": "continuous",
        }
        if self.fc:
            try:
                base.update(self.fc.available_fields())
            except Exception:
                pass
        return base

    def resolve(self, spec: Union[str, FieldSpec, Dict]) -> Dict[str, float]:
        if isinstance(spec, str):
            fs = FieldSpec(name=spec)
        elif isinstance(spec, dict):
            fs = FieldSpec(**spec)
        else:
            fs = spec
        name = fs.name.strip().lower()

        if name == "blend":
            return self._resolve_blend(fs)
        if name in {"degree", "in_degree", "out_degree", "pagerank",
                    "betweenness_approx", "closeness_harm",
                    "community_label", "connection_intensity"}:
            return self._resolve_graph_field(fs)
        if name in {"edge_density", "local_edge_density"}:
            return self._resolve_local_edge_density(fs)
        if name in {"ast_s", "ast_h", "ast_z"}:
            return self._resolve_ast_SHZ(fs)
        if name in {"code_smells", "test_coverage"}:
            return self._resolve_quality(fs)
        if name in {"churn", "recentness"}:
            return self._resolve_delta(fs)
        if name in {"sim_shz", "dist_shz"}:
            return self._resolve_shz_similarity(fs)
        if self.fc:
            try:
                data = self.fc.get_field(name, normalize=fs.normalize, ensure_fresh=True)
                if data:
                    return data
            except Exception:
                pass
        return {nid: 0.0 for nid in self.G.nodes.keys()}

    # ── Implementacje pól ─────────────────────────────────────────────────────
    def _resolve_graph_field(self, fs: FieldSpec) -> Dict[str, float]:
        if not self.fc:
            return {nid: 0.0 for nid in self.G.nodes.keys()}
        name = fs.name.strip().lower()
        fc_name = {
            "degree": "degree",
            "in_degree": "degree_in",
            "out_degree": "degree_out",
            "pagerank": "pagerank",
            "betweenness_approx": "betweenness_approx",
            "closeness_harm": "closeness_harmonic_in",
            "community_label": "community",
            "connection_intensity": "strength_out",
        }.get(name, name)
        force_norm = fs.normalize and fc_name not in {"community", "community_lpa", "community_label"}
        try:
            data = self.fc.get_field(fc_name, normalize=force_norm, ensure_fresh=True)  # type: ignore
        except Exception:
            data = None
        if not data:
            return {nid: 0.0 for nid in self.G.nodes.keys()}
        if fc_name in {"community", "community_lpa", "community_label"}:
            return {nid: float(v) for nid, v in data.items()}
        return data

    def _resolve_local_edge_density(self, fs: FieldSpec) -> Dict[str, float]:
        if "edge_density" in self._cache:
            d = self._cache["edge_density"]
            return _minmax_norm(d) if fs.normalize else d
        neighbors: Dict[str, set] = {nid: set() for nid in self.G.nodes.keys()}
        undirected = set()
        for e in self.G.edges:
            if e.src == e.dst:
                continue
            neighbors[e.src].add(e.dst)
            neighbors[e.dst].add(e.src)
            a, b = (e.src, e.dst) if e.src < e.dst else (e.dst, e.src)
            undirected.add((a, b))
        density: Dict[str, float] = {}
        for nid, nbrs in neighbors.items():
            k = len(nbrs)
            if k <= 1:
                density[nid] = 0.0
                continue
            m = 0
            lst = list(nbrs)
            for i in range(k):
                ni = lst[i]
                for j in range(i + 1, k):
                    nj = lst[j]
                    a, b = (ni, nj) if ni < nj else (nj, ni)
                    if (a, b) in undirected:
                        m += 1
            max_possible = k * (k - 1) / 2.0
            density[nid] = float(m) / max_possible if max_possible > 0 else 0.0
        self._cache["edge_density"] = density
        return _minmax_norm(density) if fs.normalize else density

    def _resolve_ast_SHZ(self, fs: FieldSpec) -> Dict[str, float]:
        which = fs.name.strip().lower().split("_", 1)[-1]  # 's'|'h'|'z'
        per_path: Dict[str, Tuple[float, float, float]] = {}
        if isinstance(self._ast_snapshot, dict) and "files" in self._ast_snapshot:
            files = self._ast_snapshot["files"]
            if isinstance(files, dict):
                for p, rec in files.items():
                    try:
                        S = float(rec.get("S", 0.0)); H = float(rec.get("H", 0.0)); Z = float(rec.get("Z", 0.0))
                        per_path[str(p)] = (S, H, Z)
                    except Exception:
                        pass
        def _get_SHZ_for_path(path: str) -> Tuple[float, float, float]:
            if path in per_path:
                return per_path[path]
            if ast_summary_of_file is None:
                return (0.0, 0.0, 0.0)
            try:
                s = ast_summary_of_file(Path(path))  # type: ignore
                if not s:
                    return (0.0, 0.0, 0.0)
                return (float(getattr(s, "S", 0.0)), float(getattr(s, "H", 0.0)), float(getattr(s, "Z", 0.0)))
            except Exception:
                return (0.0, 0.0, 0.0)
        per_file_val: Dict[str, float] = {}
        for fid, path in self.idx.file_path_by_node.items():
            S, H, Z = _get_SHZ_for_path(path)
            v = S if which == "s" else (H if which == "h" else Z)
            per_file_val[fid] = float(v)
        out: Dict[str, float] = {}
        out.update(per_file_val)
        for mid, file_ids in self.idx.files_by_module.items():
            vals = [per_file_val.get(fid, 0.0) for fid in file_ids]
            out[mid] = _aggregate(vals, fs.aggregate)
        for fid, fn_ids in self.idx.funcs_by_file.items():
            v = per_file_val.get(fid, 0.0)
            for fn in fn_ids:
                out[fn] = v
        for nid in self.G.nodes.keys():
            if nid not in out:
                out[nid] = 0.0
        return _minmax_norm(out) if fs.normalize else out

    def _resolve_quality(self, fs: FieldSpec) -> Dict[str, float]:
        name = fs.name.strip().lower()
        per_file: Dict[str, float] = {}
        if name == "test_coverage" and isinstance(self._coverage, dict):
            src = self._coverage.get("files") if "files" in self._coverage else self._coverage
            if isinstance(src, dict):
                path2v = {str(k): float(v) for k, v in src.items() if isinstance(v, (int, float))}
                for fid, path in self.idx.file_path_by_node.items():
                    per_file[fid] = path2v.get(path, 0.0)
        elif name == "code_smells" and isinstance(self._smells, dict):
            src = self._smells.get("files") if "files" in self._smells else self._smells
            if isinstance(src, dict):
                path2v = {str(k): float(v) for k, v in src.items() if isinstance(v, (int, float))}
                for fid, path in self.idx.file_path_by_node.items():
                    per_file[fid] = path2v.get(path, 0.0)
        else:
            for fid in self.idx.file_path_by_node.keys():
                per_file[fid] = 0.0
        out: Dict[str, float] = {}
        out.update(per_file)
        for mid, file_ids in self.idx.files_by_module.items():
            vals = [per_file.get(fid, 0.0) for fid in file_ids]
            out[mid] = _aggregate(vals, fs.aggregate)
        for fid, fn_ids in self.idx.funcs_by_file.items():
            v = per_file.get(fid, 0.0)
            for fn in fn_ids:
                out[fn] = v
        for nid in self.G.nodes.keys():
            if nid not in out:
                out[nid] = 0.0
        return _minmax_norm(out) if fs.normalize else out

    def _resolve_delta(self, fs: FieldSpec) -> Dict[str, float]:
        changed: Dict[str, Dict[str, float]] = {}
        if isinstance(self._delta, dict):
            files = self._delta.get("files")
            if isinstance(files, list):
                for it in files:
                    try:
                        p = str(it.get("path", "")).strip()
                        if not p:
                            continue
                        loc = it.get("loc") or {}
                        add = float(loc.get("add", 0.0)); dele = float(loc.get("del", 0.0))
                        changed[p] = {"churn": add + dele, "recent": 1.0}
                    except Exception:
                        continue
        per_file_churn: Dict[str, float] = {}
        per_file_recent: Dict[str, float] = {}
        for fid, path in self.idx.file_path_by_node.items():
            info = changed.get(path)
            if info:
                per_file_churn[fid] = float(info.get("churn", 0.0))
                per_file_recent[fid] = float(info.get("recent", 1.0))
            else:
                per_file_churn[fid] = 0.0
                per_file_recent[fid] = 0.0
        target = per_file_churn if fs.name.strip().lower() == "churn" else per_file_recent
        out: Dict[str, float] = {}
        out.update(target)
        for mid, file_ids in self.idx.files_by_module.items():
            vals = [target.get(fid, 0.0) for fid in file_ids]
            out[mid] = _aggregate(vals, fs.aggregate)
        for fid, fn_ids in self.idx.funcs_by_file.items():
            v = target.get(fid, 0.0)
            for fn in fn_ids:
                out[fn] = v
        for nid in self.G.nodes.keys():
            if nid not in out:
                out[nid] = 0.0
        return _minmax_norm(out) if fs.normalize else out

    def _resolve_shz_similarity(self, fs: FieldSpec) -> Dict[str, float]:
        def _get_SHZ_for_path(path: str) -> Tuple[float, float, float]:
            if isinstance(self._ast_snapshot, dict):
                rec = (self._ast_snapshot.get("files") or {}).get(path) if "files" in self._ast_snapshot else None
                if isinstance(rec, dict):
                    try:
                        return (float(rec.get("S", 0.0)), float(rec.get("H", 0.0)), float(rec.get("Z", 0.0)))
                    except Exception:
                        pass
            if ast_summary_of_file is None:
                return (0.0, 0.0, 0.0)
            try:
                s = ast_summary_of_file(Path(path))  # type: ignore
                if not s:
                    return (0.0, 0.0, 0.0)
                return (float(getattr(s, "S", 0.0)), float(getattr(s, "H", 0.0)), float(getattr(s, "Z", 0.0)))
            except Exception:
                return (0.0, 0.0, 0.0)

        file_SHZ: Dict[str, Tuple[float, float, float]] = {}
        for fid, path in self.idx.file_path_by_node.items():
            file_SHZ[fid] = _get_SHZ_for_path(path)

        perc = float(fs.params.get("target_percentile", 95.0))
        ws, wh, wz = fs.params.get("weights", [1.0, 1.0, 0.25])
        ws, wh, wz = float(ws), float(wh), float(wz)

        def _percentile(vals: List[float], p: float) -> float:
            if not vals:
                return 0.0
            vals = sorted(vals)
            p = max(0.0, min(100.0, p))
            if p == 100.0:
                return float(vals[-1])
            rank = (p / 100.0) * (len(vals) - 1)
            lo = int(math.floor(rank)); hi = int(math.ceil(rank))
            if lo == hi:
                return float(vals[lo])
            frac = rank - lo
            return float(vals[lo] * (1 - frac) + vals[hi] * frac)

        Ss = [S for (S, _, _) in file_SHZ.values()]
        Hs = [H for (_, H, _) in file_SHZ.values()]
        Zs = [Z for (_, _, Z) in file_SHZ.values()]
        R = (_percentile(Ss, perc), _percentile(Hs, perc), _percentile(Zs, perc))

        per_file_dist: Dict[str, float] = {}
        max_d = 1e-9
        for fid, (S, H, Z) in file_SHZ.items():
            d = math.sqrt(ws * (S - R[0]) ** 2 + wh * (H - R[1]) ** 2 + wz * (Z - R[2]) ** 2)
            per_file_dist[fid] = d
            if d > max_d:
                max_d = d
        for fid in per_file_dist.keys():
            per_file_dist[fid] = per_file_dist[fid] / max_d if max_d > 0 else 0.0

        per_module: Dict[str, float] = {}
        for mid, file_ids in self.idx.files_by_module.items():
            vals = [per_file_dist.get(fid, 0.0) for fid in file_ids]
            per_module[mid] = _aggregate(vals, fs.aggregate)
        per_func: Dict[str, float] = {}
        for fid, fns in self.idx.funcs_by_file.items():
            v = per_file_dist.get(fid, 0.0)
            for fn in fns:
                per_func[fn] = v

        want_sim = (fs.name.strip().lower() == "sim_shz")
        out: Dict[str, float] = {}
        for nid in self.G.nodes.keys():
            n = self.G.nodes[nid]
            if n.kind == "file":
                v = per_file_dist.get(nid, 0.0)
            elif n.kind == "module":
                v = per_module.get(nid, 0.0)
            elif n.kind == "func":
                v = per_func.get(nid, 0.0)
            else:
                v = 0.0
            if want_sim:
                v = 1.0 - v
            out[nid] = float(max(0.0, min(1.0, v)))
        return out

    def _resolve_blend(self, fs: FieldSpec) -> Dict[str, float]:
        parts = fs.blend or []
        if not parts:
            return {nid: 0.0 for nid in self.G.nodes.keys()}
        comp: List[Tuple[Dict[str, float], float]] = []
        for item in parts:
            fname = str(item.get("field", "")).strip()
            w = float(item.get("w", 1.0))
            if not fname:
                continue
            sub = self.resolve(FieldSpec(name=fname, normalize=True))
            comp.append((sub, w))
        out: Dict[str, float] = {nid: 0.0 for nid in self.G.nodes.keys()}
        for sub, w in comp:
            for nid, v in sub.items():
                out[nid] = out.get(nid, 0.0) + w * float(v)
        return _minmax_norm(out) if fs.normalize else out

# ── Funkcja modułowa (tolerancja na nieznane kwargs) ─────────────────────────
def resolve_field(spec: Union[str, FieldSpec, Dict], repo_root: Optional[Path] = None, **_ignored) -> Dict[str, float]:
    return FieldsRegistry(repo_root).resolve(spec)

# ── CLI (podgląd/eksport) ────────────────────────────────────────────────────
def run_cli(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="fields", description="Rejestr pól operacyjnych (export JSON)")
    p.add_argument("name", help="Nazwa pola (np. pagerank, ast_s, sim_shz, dist_shz, blend)")
    p.add_argument("--spec", help="Ścieżka do JSON spec (dla parametryzacji, np. sim_shz/dist_shz/blend)")
    p.add_argument("--no-norm", action="store_true", help="Wyłącz normalizację [0,1] (tam gdzie dotyczy)")
    p.add_argument("--aggregate", choices=["mean", "sum", "max"], default="mean")
    p.add_argument("--export", action="store_true", help="Zapisz do .glx/graphs/fields/<name>.json")
    p.add_argument("--repo-root", "--repo", dest="repo_root", default=None, help="Root repo (domyślnie autodetekcja)")
    args = p.parse_args(argv)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else None

    if args.spec:
        spec_obj = json.loads(Path(args.spec).read_text(encoding="utf-8"))
        if "name" not in spec_obj:
            spec_obj["name"] = args.name
        if "normalize" not in spec_obj:
            spec_obj["normalize"] = not args.no_norm
        if "aggregate" not in spec_obj:
            spec_obj["aggregate"] = args.aggregate
        fs = FieldSpec(**spec_obj)
    else:
        fs = FieldSpec(name=args.name, normalize=not args.no_norm, aggregate=args.aggregate)

    reg = FieldsRegistry(repo_root)
    data = reg.resolve(fs)

    af = GlxArtifacts(repo_root or reg.repo_root)

    if args.export:
        out = {
            "field": asdict(fs),
            "meta": {
                "nodes": len(data),
                "sources": {
                    "project_graph": str(_glx_path(af, "graphs/project_graph.json", reg.repo_root)),
                    "metrics": str(_glx_path(af, "graphs/metrics.json", reg.repo_root)),
                    "delta_report": str(_glx_path(af, "delta_report.json", reg.repo_root)),
                    "coverage": str(_glx_path(af, "tests/coverage.json", reg.repo_root)),
                    "code_smells": str(_glx_path(af, "quality/code_smells.json", reg.repo_root)),
                    "ast_index": str(_glx_path(af, "graphs/ast_index.json", reg.repo_root)),
                }
            },
            "values": data,
        }
        path = _glx_path(af, f"graphs/fields/{fs.name}.json", reg.repo_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(str(path))
    else:
        for i, (nid, v) in enumerate(list(data.items())[:20]):
            print(f"{nid}\t{v:.6f}" if isinstance(v, (int, float)) else f"{nid}\t{v}")
        if len(data) > 20:
            print(f"... ({len(data)-20} more)")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(run_cli())
