#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
glitchlab.analysis.field_cache — cache i rewalidacja „fields” dla metasoczewek
Python 3.9 • stdlib only

Cel:
- Dostarczyć warstwę cache/rewalidacji dla pól wykorzystywanych przez metasoczewki.
- Zapewnić spójność z aktualnym grafem projektu oraz (opcjonalnie) z delta_report.
- Umożliwić pobieranie pól per-węzeł z normalizacją do [0,1] (gdy ma sens).

We:
- .glx/graphs/project_graph.json  (kanoniczny graf projektu)
- .glx/graphs/metrics.json        (globalne metryki grafowe)
- .glx/delta_report.json          (opcjonalnie; fingerprint Δ)

Wy:
- API: FieldCache(repo_root).get_field(name, normalize=True) -> Dict[node_id, float|int]
- .glx/graphs/field_cache.json    (podsumowanie stanu cache: wersje, hashe, dostępne pola)

Zależności:
- analysis.project_graph.ProjectGraph
- analysis.graph_metrics.compute_graph_metrics (do re-kalkulacji metryk)
- io.artifacts.GlxArtifacts (z fallbackiem)

Uwagi:
- Jeśli metrics.json nie istnieje lub jest niespójny z project_graph.json (hash),
  poleci automatyczna przebudowa metryk (PageRank, betweenness_approx, closeness_harmonic_in, itp.).
- Pola „aliasy”: degree → degree_total; betweenness → betweenness_approx; closeness → closeness_harmonic_in; community → community_lpa
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

# ── Artefakty (.glx/*) ────────────────────────────────────────────────────────
try:
    from glitchlab.io.artifacts import GlxArtifacts  # type: ignore
except Exception:  # pragma: no cover
    class GlxArtifacts:  # type: ignore
        def __init__(self) -> None:
            self.repo_root = Path(__file__).resolve().parents[3]
            self.base = self.repo_root / "glitchlab" / ".glx"
            self.base.mkdir(parents=True, exist_ok=True)

        def read_json(self, rel_name: str) -> dict:
            p = (self.base / rel_name).resolve()
            return json.loads(p.read_text(encoding="utf-8"))

        def write_json(self, rel_name: str, obj) -> Path:
            p = (self.base / rel_name).resolve()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
            return p

        def path(self, rel_name: str) -> Path:
            return (self.base / rel_name).resolve()

af = GlxArtifacts()

# ── Graf projektu i metryki ───────────────────────────────────────────────────
from glitchlab.analysis.project_graph import ProjectGraph, build_project_graph
try:
    from glitchlab.analysis.graph_metrics import compute_graph_metrics, GraphMetricsConfig  # type: ignore
except Exception:  # pragma: no cover
    compute_graph_metrics = None  # type: ignore
    GraphMetricsConfig = None     # type: ignore

VERSION = "v1"

# Mapowanie aliasów pól na rzeczywiste nazwy z metrics.json
FIELD_ALIASES = {
    "degree": "degree_total",
    "degree_total": "degree_total",
    "degree_in": "degree_in",
    "degree_out": "degree_out",
    "strength": "strength_out",
    "strength_out": "strength_out",
    "strength_in": "strength_in",
    "pagerank": "pagerank",
    "betweenness": "betweenness_approx",
    "betweenness_approx": "betweenness_approx",
    "closeness": "closeness_harmonic_in",
    "closeness_harmonic_in": "closeness_harmonic_in",
    "community": "community_lpa",
    "community_lpa": "community_lpa",
}

# Pola, które traktujemy jako „ciągłe” i normalizujemy do [0,1] z min-max
CONTINUOUS_FIELDS = {
    "degree_in",
    "degree_out",
    "degree_total",
    "strength_in",
    "strength_out",
    "pagerank",
    "betweenness_approx",
    "closeness_harmonic_in",
}

# Pola kategorialne (normalizacja domyślnie OFF; zwracamy id/etykiety)
CATEGORICAL_FIELDS = {
    "community_lpa",
}


def _safe_read_json(rel: str) -> Optional[dict]:
    try:
        return af.read_json(rel)
    except Exception:
        return None


def _extract_delta_hash(delta_report: Optional[dict]) -> Optional[str]:
    if not isinstance(delta_report, dict):
        return None
    # próbujemy kilku znanych miejsc
    for key in ("hash", "fingerprint", "report_hash"):
        v = delta_report.get(key)
        if isinstance(v, str) and v:
            return v
        if isinstance(v, dict):
            hv = v.get("hash")
            if isinstance(hv, str) and hv:
                return hv
    # fallback: zserializuj wybrane sekcje i policz skrót „logiczny” (bez algorytmów hashujących — light)
    try:
        subset = {k: delta_report.get(k) for k in ("base", "head", "stats", "files")}
        payload = json.dumps(subset, sort_keys=True, separators=(",", ":"))
        # prosty skrót: długość + kilka znaków
        return f"len{len(payload)}:{payload[:16]}"
    except Exception:
        return None


def _minmax_norm(d: Dict[str, float]) -> Dict[str, float]:
    if not d:
        return {}
    vals = list(d.values())
    mn, mx = min(vals), max(vals)
    if mx <= mn:
        # stała: zwracamy zera
        return {k: 0.0 for k in d.keys()}
    scale = mx - mn
    return {k: (float(v) - mn) / scale for k, v in d.items()}


@dataclass
class FieldCacheSummary:
    version: str
    generated_utc: str
    basis: dict            # {graph_hash, nodes, edges, metrics_version, delta_hash}
    fields: Dict[str, dict]  # nazwa → {"kind":"continuous|categorical", "normalized":bool}
    sources: dict          # ścieżki artefaktów


class FieldCache:
    """
    Warstwa cache pól meta-soczewki nad grafem projektu.
    Zapewnia spójność z grafem i metrykami; potrafi wymusić rekalkulację metryk.
    """

    def __init__(self, repo_root: Optional[Path] = None) -> None:
        self.repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[3]
        self._graph: Optional[ProjectGraph] = None
        self._metrics_payload: Optional[dict] = None
        self._delta_report: Optional[dict] = None
        self._summary: Optional[FieldCacheSummary] = None

    # ── Ładowanie artefaktów ──────────────────────────────────────────────────

    def _load_graph(self) -> ProjectGraph:
        if self._graph is not None:
            return self._graph
        data = _safe_read_json("graphs/project_graph.json")
        if data:
            self._graph = ProjectGraph.from_json_obj(repo_root=self.repo_root, data=data)
        else:
            # brak artefaktu — budujemy
            self._graph = build_project_graph()
        return self._graph

    def _load_metrics(self) -> Optional[dict]:
        if self._metrics_payload is not None:
            return self._metrics_payload
        self._metrics_payload = _safe_read_json("graphs/metrics.json")
        return self._metrics_payload

    def _load_delta(self) -> Optional[dict]:
        if self._delta_report is not None:
            return self._delta_report
        self._delta_report = _safe_read_json("delta_report.json")
        return self._delta_report

    # ── Spójność / rewalidacja ────────────────────────────────────────────────

    def _ensure_metrics_up_to_date(self, force_recompute: bool = False) -> dict:
        """
        Sprawdza spójność metrics.json z bieżącym grafem (graph_hash).
        W razie potrzeby (lub na żądanie) przelicza metryki i zapisuje artefakt.
        Zwraca aktualny payload metryk.
        """
        g = self._load_graph()
        payload = self._load_metrics()
        g_hash = g.meta.get("graph_hash") or g.graph_hash()

        need_recompute = force_recompute or (payload is None)
        if payload and not force_recompute:
            try:
                basis = payload.get("basis", {})
                m_hash = basis.get("graph_hash")
                if m_hash != g_hash:
                    need_recompute = True
            except Exception:
                need_recompute = True

        if need_recompute:
            if compute_graph_metrics is None or GraphMetricsConfig is None:
                # bez zależności — policzymy minimalny zestaw stopni/siły lokalnie
                payload = self._compute_minimal_metrics_fallback(g)
            else:
                cfg = GraphMetricsConfig()
                payload = compute_graph_metrics(g, cfg)
            self._metrics_payload = payload
            af.write_json("graphs/metrics.json", payload)

        return payload

    @staticmethod
    def _compute_minimal_metrics_fallback(g: ProjectGraph) -> dict:
        # Prostolinijne liczenie degree/strength + meta, gdy nie mamy modułu metryk
        deg_in = {nid: 0 for nid in g.nodes}
        deg_out = {nid: 0 for nid in g.nodes}
        str_in = {nid: 0.0 for nid in g.nodes}
        str_out = {nid: 0.0 for nid in g.nodes}
        for e in g.edges:
            deg_out[e.src] += 1
            deg_in[e.dst] += 1
            w = float(e.weight)
            str_out[e.src] += w
            str_in[e.dst] += w
        deg_all = {nid: deg_in[nid] + deg_out[nid] for nid in g.nodes}
        density = _edge_density_meta(g)
        payload = {
            "version": "v1-min",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "basis": {
                "graph_version": g.meta.get("graph_version", "v1"),
                "graph_hash": g.meta.get("graph_hash") or g.graph_hash(),
                "nodes": len(g.nodes),
                "edges": len(g.edges),
                "edge_density": density,
            },
            "config": {"mode": "minimal"},
            "metrics": {
                "degree_in": deg_in,
                "degree_out": deg_out,
                "degree_total": deg_all,
                "strength_in": str_in,
                "strength_out": str_out,
            },
        }
        return payload

    # ── API: pobieranie pól ───────────────────────────────────────────────────

    def available_fields(self) -> Dict[str, str]:
        """
        Zwraca mapę: nazwa_pola -> 'continuous'|'categorical'
        (wraz z aliasami).
        """
        payload = self._ensure_metrics_up_to_date(force_recompute=False)
        m = payload.get("metrics", {})
        present = set(m.keys())
        out: Dict[str, str] = {}
        for public, actual in FIELD_ALIASES.items():
            if actual in present or public in present:
                kind = "categorical" if actual in CATEGORICAL_FIELDS or public in CATEGORICAL_FIELDS else "continuous"
                out[public] = kind
        # jawnie dodaj „actual” nazwy
        for k in present:
            kind = "categorical" if k in CATEGORICAL_FIELDS else "continuous"
            out[k] = kind
        return out

    def get_field(self, name: str, *, normalize: bool = True, default: float = 0.0,
                  ensure_fresh: bool = True) -> Dict[str, float]:
        """
        Zwraca słownik {node_id: value} dla zadanego pola.
        - name: może być aliasem (np. „degree” → „degree_total”).
        - normalize=True: min-max do [0,1] (tylko dla pól ciągłych).
        - ensure_fresh=True: sprawdza spójność metryk i w razie potrzeby przebudowuje.
        """
        actual = FIELD_ALIASES.get(name, name)
        payload = self._ensure_metrics_up_to_date(force_recompute=False if ensure_fresh else False)
        metrics = payload.get("metrics", {})
        data = metrics.get(actual)
        if data is None and name in metrics:
            data = metrics[name]
        if data is None:
            # spróbuj policzyć lokalnie (degree_total)
            if actual == "degree_total":
                g = self._load_graph()
                deg_in = {nid: 0 for nid in g.nodes}
                deg_out = {nid: 0 for nid in g.nodes}
                for e in g.edges:
                    deg_out[e.src] += 1
                    deg_in[e.dst] += 1
                data = {nid: deg_in[nid] + deg_out[nid] for nid in g.nodes}
            else:
                return {}

        # Kategoryczne pola — nie normalizujemy domyślnie
        if normalize and actual in CONTINUOUS_FIELDS:
            data = _minmax_norm({k: float(v) for k, v in data.items()})
        else:
            # rzutuj do float dla spójności (poza community)
            if actual not in CATEGORICAL_FIELDS:
                data = {k: float(v) for k, v in data.items()}
        return data

    # ── Podsumowanie cache ────────────────────────────────────────────────────

    def write_summary(self) -> Path:
        g = self._load_graph()
        payload = self._ensure_metrics_up_to_date(force_recompute=False)
        delta = self._load_delta()
        delta_hash = _extract_delta_hash(delta)
        basis = {
            "graph_hash": g.meta.get("graph_hash") or g.graph_hash(),
            "nodes": len(g.nodes),
            "edges": len(g.edges),
            "metrics_version": payload.get("version"),
            "edge_density": payload.get("basis", {}).get("edge_density"),
            "delta_hash": delta_hash,
        }
        fields_meta: Dict[str, dict] = {}
        for fname, kind in self.available_fields().items():
            fields_meta[fname] = {"kind": kind, "normalized": (kind == "continuous")}
        summary = FieldCacheSummary(
            version=VERSION,
            generated_utc=datetime.now(timezone.utc).isoformat(),
            basis=basis,
            fields=fields_meta,
            sources={
                "project_graph": str(af.path("graphs/project_graph.json")),
                "metrics": str(af.path("graphs/metrics.json")),
                "delta_report": str(af.path("delta_report.json")),
            },
        )
        self._summary = summary
        return af.write_json("graphs/field_cache.json", asdict(summary))


# ── Pomocnicze: gęstość ───────────────────────────────────────────────────────

def _edge_density_meta(g: ProjectGraph) -> float:
    n = len(g.nodes)
    if n <= 1:
        return 0.0
    m = sum(1 for e in g.edges if e.src != e.dst)
    return float(m) / float(n * (n - 1))


# ── CLI ───────────────────────────────────────────────────────────────────────

def run_cli(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(prog="field_cache", description="Cache/rewalidacja pól (fields) dla metasoczewek")
    p.add_argument("--force", action="store_true", help="Wymuś przeliczenie metryk (ignoruj cache)")
    p.add_argument("--list", action="store_true", help="Wypisz dostępne pola i zakończ")
    args = p.parse_args(argv)

    fc = FieldCache()
    if args.force:
        fc._ensure_metrics_up_to_date(force_recompute=True)
    if args.list:
        fields = fc.available_fields()
        for k in sorted(fields.keys()):
            print(f"{k}: {fields[k]}")
        return 0

    out = fc.write_summary()
    print(str(out))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(run_cli())
