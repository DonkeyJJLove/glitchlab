# glitchlab/analysis/formats.py
# -*- coding: utf-8 -*-
"""
Formalne formaty JSON artefaktów analizy grafu / metasoczewek w GlitchLab.

Obsługiwane artefakty (.glx/graphs):
  - project_graph.json         — globalny graf projektu (węzły/krawędzie + meta).
  - metrics.json               — globalne metryki grafowe (mapa: nazwa_metryki -> {node_id: value}).
  - field_cache.json           — cache wyliczonych pól (hashy, statystyk) dla metasoczewek.
  - meta_*.json                — wynik metasoczewki (podgraf) + _meta.spec/anchors/metrics.
Dodatkowo: pomocnicza walidacja (bez twardej zależności od jsonschema).

Uwaga: staramy się być *kompatybilni wstecznie* — schematy dopuszczają
pewne warianty (np. meta vs _meta; opcjonalne weight na krawędziach).

Brak zależności zewnętrznych — walidator działa w trybie lekko-ścisłym.
Jeśli w środowisku jest zainstalowany `jsonschema`, zostanie użyty.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union
import hashlib
import json
import math

__all__ = [
    # wersje
    "FORMATS_VERSION",
    "PROJECT_GRAPH_VERSION",
    "GRAPH_METRICS_VERSION",
    "FIELD_CACHE_VERSION",
    "META_LENS_VERSION",
    # schematy i API
    "SCHEMAS",
    "schema_for",
    "json_hash",
    "normalize_meta_key",
    "validate_project_graph",
    "validate_graph_metrics",
    "validate_field_cache",
    "validate_meta_lens",
    "validate",  # ogólne
]

# ─────────────────────────────────────────────────────────────────────────────
# Wersjonowanie artefaktów
# ─────────────────────────────────────────────────────────────────────────────

FORMATS_VERSION: str = "2025.10"
PROJECT_GRAPH_VERSION: str = "PG-1"
GRAPH_METRICS_VERSION: str = "GM-1"
FIELD_CACHE_VERSION: str = "FC-1"
META_LENS_VERSION: str = "ML-1"

# ─────────────────────────────────────────────────────────────────────────────
# Pomocnicze
# ─────────────────────────────────────────────────────────────────────────────

def json_hash(obj: Mapping[str, Any]) -> str:
    """
    Stabilny SHA-256 dla obiektu JSON (UTF-8, sort_keys, separators=(',',':')).
    """
    b = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(b).hexdigest()


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def _is_str(x: Any) -> bool:
    return isinstance(x, str)


def _as_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}


def _arr(x: Any) -> List[Any]:
    return list(x) if isinstance(x, (list, tuple)) else []


# ─────────────────────────────────────────────────────────────────────────────
# Specyfikacje (JSON Schema-ish) — luźne, ale jednoznaczne
# ─────────────────────────────────────────────────────────────────────────────

SCHEMAS: Dict[str, Dict[str, Any]] = {
    "project_graph": {
        "version": PROJECT_GRAPH_VERSION,
        "required": ["graph"],
        "properties": {
            "version": {"type": "string"},
            "meta": {"type": "object"},  # np. {"graph_hash": "...", "ts": "...", ...}
            "graph": {
                "type": "object",
                "required": ["nodes", "edges"],
                "properties": {
                    "nodes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["id", "kind", "label"],
                            "properties": {
                                "id": {"type": "string"},
                                "kind": {"type": "string"},
                                "label": {"type": "string"},
                                "meta": {"type": "object"},
                            },
                        },
                    },
                    "edges": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["src", "dst"],
                            "properties": {
                                "src": {"type": "string"},
                                "dst": {"type": "string"},
                                "kind": {"type": "string"},
                                "weight": {"type": "number"},
                            },
                        },
                    },
                },
            },
        },
    },
    "graph_metrics": {
        "version": GRAPH_METRICS_VERSION,
        "required": ["metrics"],
        "properties": {
            "version": {"type": "string"},
            "meta": {"type": "object"},  # {"graph_hash": "...", "ts": "...", ...}
            "metrics": {
                "type": "object",  # nazwa_metryki -> {node_id: value}
                "additionalProperties": {
                    "type": "object",
                    "additionalProperties": {"type": "number"},
                },
            },
        },
    },
    "field_cache": {
        "version": FIELD_CACHE_VERSION,
        "required": ["fields"],
        "properties": {
            "version": {"type": "string"},
            "meta": {"type": "object"},  # {"graph_hash", "metrics_hash", ...}
            "fields": {
                "type": "object",  # field_name -> summary
                "additionalProperties": {
                    "type": "object",
                    "properties": {
                        "hash": {"type": "string"},
                        "ts": {"type": "string"},
                        "n": {"type": "number"},
                        "min": {"type": "number"},
                        "max": {"type": "number"},
                        "mean": {"type": "number"},
                        "p50": {"type": "number"},
                        "p90": {"type": "number"},
                    },
                },
            },
        },
    },
    "meta_lens": {
        "version": META_LENS_VERSION,
        "required": [["meta"], ["_meta"], ["nodes", "edges"]],
        "properties": {
            # dopuszczamy zarówno "meta" jak i "_meta"
            "meta": {"type": "object"},
            "_meta": {"type": "object"},
            "nodes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["id", "kind", "label"],
                    "properties": {
                        "id": {"type": "string"},
                        "kind": {"type": "string"},
                        "label": {"type": "string"},
                        "meta": {"type": "object"},
                    },
                },
            },
            "edges": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["src", "dst"],
                    "properties": {
                        "src": {"type": "string"},
                        "dst": {"type": "string"},
                        "kind": {"type": "string"},
                        "weight": {"type": "number"},
                    },
                },
            },
        },
        # oczekiwane pola w meta/_meta
        "meta_expected": {
            "spec": {"type": "object"},       # m.in. level/center/depth/...
            "anchors": {"type": "array"},     # lista id węzłów
            "metrics": {"type": "object"},    # zagregowane metryki okna
        },
    },
}


def schema_for(kind: str) -> Dict[str, Any]:
    """
    Pobierz schemat dla: 'project_graph' | 'graph_metrics' | 'field_cache' | 'meta_lens'.
    """
    if kind not in SCHEMAS:
        raise KeyError(f"unknown schema kind: {kind}")
    return SCHEMAS[kind]

# ─────────────────────────────────────────────────────────────────────────────
# Normalizacja i walidacja „lekka” (fallback bez jsonschema)
# ─────────────────────────────────────────────────────────────────────────────

def normalize_meta_key(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ujednolica meta/_meta — jeżeli występuje tylko '_meta', dodaje lustrzaną 'meta',
    aby narzędzia mogły korzystać z jednego klucza.
    """
    if "_meta" in payload and "meta" not in payload:
        payload["meta"] = payload["_meta"]
    return payload


def _require_keys(obj: Mapping[str, Any], keys: Iterable[str], errors: List[str], prefix: str = "") -> None:
    for k in keys:
        if k not in obj:
            errors.append(f"{prefix}missing key: {k}")


def _validate_nodes(nodes: Any, errors: List[str], prefix: str = "") -> None:
    if not isinstance(nodes, list):
        errors.append(f"{prefix}nodes must be a list")
        return
    for i, n in enumerate(nodes):
        if not isinstance(n, dict):
            errors.append(f"{prefix}nodes[{i}] must be an object")
            continue
        for k in ("id", "kind", "label"):
            if k not in n or not _is_str(n[k]):
                errors.append(f"{prefix}nodes[{i}].{k} must be a string")
        if "meta" in n and not isinstance(n["meta"], dict):
            errors.append(f"{prefix}nodes[{i}].meta must be an object when present")


def _validate_edges(edges: Any, errors: List[str], prefix: str = "") -> None:
    if not isinstance(edges, list):
        errors.append(f"{prefix}edges must be a list")
        return
    for i, e in enumerate(edges):
        if not isinstance(e, dict):
            errors.append(f"{prefix}edges[{i}] must be an object")
            continue
        for k in ("src", "dst"):
            if k not in e or not _is_str(e[k]):
                errors.append(f"{prefix}edges[{i}].{k} must be a string")
        if "weight" in e and not _is_num(e["weight"]):
            errors.append(f"{prefix}edges[{i}].weight must be a number when present")
        if "kind" in e and not _is_str(e["kind"]):
            errors.append(f"{prefix}edges[{i}].kind must be a string when present")


# —— Walidatory per artefakt ————————————————————————————————

def validate_project_graph(payload: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Minimalna walidacja struktury project_graph.json.
    Zwraca: (ok, lista_błędów)
    """
    errors: List[str] = []
    if not isinstance(payload, dict):
        return False, ["payload must be an object"]
    if "graph" not in payload or not isinstance(payload["graph"], dict):
        errors.append("missing object: graph")
        return False, errors

    graph = payload["graph"]
    _require_keys(graph, ["nodes", "edges"], errors, prefix="graph.")
    _validate_nodes(graph.get("nodes"), errors, prefix="graph.")
    _validate_edges(graph.get("edges"), errors, prefix="graph.")

    # meta jest opcjonalna, ale jeśli występuje i zawiera graph_hash, powinien być stringiem
    meta = _as_dict(payload.get("meta"))
    if "graph_hash" in meta and not _is_str(meta["graph_hash"]):
        errors.append("meta.graph_hash must be a string when present")

    return (len(errors) == 0), errors


def validate_graph_metrics(payload: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Walidacja metrics.json (nazwa_metryki -> {node_id: value})
    """
    errors: List[str] = []
    if not isinstance(payload, dict):
        return False, ["payload must be an object"]

    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        return False, ["missing object: metrics"]

    for mname, mvals in metrics.items():
        if not _is_str(mname):
            errors.append("metric name must be a string")
            continue
        if not isinstance(mvals, dict):
            errors.append(f"metrics.{mname} must be an object (node_id -> number)")
            continue
        for nid, val in mvals.items():
            if not _is_str(nid) or not _is_num(val):
                errors.append(f"metrics.{mname}[{nid!r}] must be number, with string node_id")

    # meta (opcjonalnie) – jeżeli jest graph_hash, sprawdź typ
    meta = _as_dict(payload.get("meta"))
    if "graph_hash" in meta and not _is_str(meta["graph_hash"]):
        errors.append("meta.graph_hash must be a string when present")

    return (len(errors) == 0), errors


def validate_field_cache(payload: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Walidacja field_cache.json. Schemat jest celowo elastyczny:
    - 'fields' (wymagane): dict field_name -> {hash?, ts?, n?, min?, max?, mean?, p50?, p90?}
    - 'meta' (opc.): {graph_hash?, metrics_hash?, ...}
    """
    errors: List[str] = []
    if not isinstance(payload, dict):
        return False, ["payload must be an object"]

    fields = payload.get("fields")
    if not isinstance(fields, dict):
        errors.append("missing object: fields")
        return False, errors

    for fname, summary in fields.items():
        if not _is_str(fname):
            errors.append("field name must be a string")
            continue
        if not isinstance(summary, dict):
            errors.append(f"fields.{fname} must be an object")
            continue
        # Typy popularnych kluczy (opcjonalne)
        if "hash" in summary and not _is_str(summary["hash"]):
            errors.append(f"fields.{fname}.hash must be a string")
        for numk in ("n", "min", "max", "mean", "p50", "p90"):
            if numk in summary and not _is_num(summary[numk]):
                errors.append(f"fields.{fname}.{numk} must be a number")
        if "ts" in summary and not _is_str(summary["ts"]):
            errors.append(f"fields.{fname}.ts must be a string")

    # meta (opcjonalnie)
    meta = _as_dict(payload.get("meta"))
    if "graph_hash" in meta and not _is_str(meta["graph_hash"]):
        errors.append("meta.graph_hash must be a string when present")
    if "metrics_hash" in meta and not _is_str(meta["metrics_hash"]):
        errors.append("meta.metrics_hash must be a string when present")

    return (len(errors) == 0), errors


def validate_meta_lens(payload: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Walidacja meta_*.json (wynik metasoczewki).
    Akceptuje top-level: nodes[], edges[], meta lub _meta (oba dopuszczalne).
    W meta oczekujemy co najmniej: spec (obj), anchors (list), metrics (obj).
    """
    errors: List[str] = []
    if not isinstance(payload, dict):
        return False, ["payload must be an object"]

    # nodes/edges
    _validate_nodes(payload.get("nodes"), errors)
    _validate_edges(payload.get("edges"), errors)

    # meta lub _meta
    meta = payload.get("meta") or payload.get("_meta")
    if not isinstance(meta, dict):
        errors.append("missing object: meta/_meta")
        return False, errors

    # spodziewane pola meta
    if "spec" in meta and not isinstance(meta["spec"], dict):
        errors.append("meta.spec must be an object when present")
    if "anchors" in meta and not isinstance(meta["anchors"], list):
        errors.append("meta.anchors must be an array when present")
    if "metrics" in meta and not isinstance(meta["metrics"], dict):
        errors.append("meta.metrics must be an object when present")

    return (len(errors) == 0), errors


# ─────────────────────────────────────────────────────────────────────────────
# Walidacja ogólna (z opcjonalnym użyciem jsonschema)
# ─────────────────────────────────────────────────────────────────────────────

def _validate_with_jsonschema(kind: str, payload: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Jeżeli w systemie jest zainstalowana biblioteka `jsonschema`, skorzystaj z niej.
    W przeciwnym razie zwróć (False, ["jsonschema not available"]) i caller użyje fallbacku.
    """
    try:
        import jsonschema  # type: ignore
    except Exception:
        return False, ["jsonschema not available"]

    # zbuduj minimalny JSON Schema na podstawie SCHEMAS (przeniesienie pól)
    s = SCHEMAS.get(kind)
    if not s:
        return False, [f"unknown schema kind: {kind}"]

    # ad-hoc — generujemy prosty schema z części 'properties' i required
    js_schema = {"type": "object", "properties": s.get("properties", {}), "additionalProperties": True}
    req = s.get("required")
    if isinstance(req, list):
        # Obsłużamy złożone required: listy list (any-of)
        # jsonschema nie wspiera „dowolnej z list required” bez anyOf – więc budujemy anyOf.
        if req and isinstance(req[0], list):
            js_schema["anyOf"] = [{"required": r} for r in req if isinstance(r, list)]
        else:
            js_schema["required"] = req
    try:
        jsonschema.validate(payload, js_schema)  # type: ignore
        return True, []
    except Exception as e:
        return False, [str(e)]


def validate(kind: str, payload: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Walidacja ogólna po nazwie schematu.
    Najpierw próbujemy jsonschema (jeśli dostępne), potem fallback „lekki”.
    """
    ok, errs = _validate_with_jsonschema(kind, payload)
    if ok:
        return True, []

    # fallback
    if kind == "project_graph":
        return validate_project_graph(payload)
    if kind == "graph_metrics":
        return validate_graph_metrics(payload)
    if kind == "field_cache":
        return validate_field_cache(payload)
    if kind == "meta_lens":
        return validate_meta_lens(payload)
    return False, [f"unknown schema kind: {kind}"]
