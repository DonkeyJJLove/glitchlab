# glitchlab/analysis/grammar/events.py
# -*- coding: utf-8 -*-
"""
Kontrakty zdarzeń (BUS) używanych przez warstwę analityczną GlitchLab.

Definiuje:
  • Stałe topiców (TOPIC_*),
  • Minimalistyczne typy payloadów (TypedDict),
  • Lekki walidator validate_event_payload(topic, payload).

Tematy (topics):
  - analytics.delta.ready
      payload: { "delta_report": {..}, "meta"?: { "ts"?: str, "version"?: str } }

  - analytics.invariants.violation
      payload: { "violations": {..}, "meta"?: { "ts"?: str, "version"?: str } }

  - analytics.scope.metrics.updated
      payload: {
        "kind": "graph_metrics",
        "paths"?: { "metrics"?: str },
        "meta"?: { "graph_hash"?: str, "metrics_hash"?: str, "ts"?: str, "version"?: str },
        "summary"?: {..}   # np. zliczenia/zakresy na potrzeby HUD/telemetrii
      }

  - analytics.scope.meta.ready
      payload: {
        "kind": "meta_lens",
        "level": str,          # "project"|"module"|"file"|"func"|"bus"|...
        "name":  str,          # np. "glitchlab.core" lub "auto"
        "paths": { "json": str, "dot"?: str },
        "anchors_count"?: int,
        "window_metrics"?: {..},   # np. nodes_total, edges_total, by_kind, AST_S/H/Z
        "meta"?: { "ts"?: str, "version"?: str }
      }

Zależności: wyłącznie stdlib (typing).
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, List, Optional, Literal, TypedDict

# Wersja kontraktu (schematów) w tej warstwie
EVENTS_SCHEMA_VERSION: str = "v1"

# ──────────────────────────────────────────────────────────────────────────────
# Topics (stałe)
# ──────────────────────────────────────────────────────────────────────────────

TOPIC_ANALYTICS_DELTA_READY: str = "analytics.delta.ready"
TOPIC_ANALYTICS_INVARIANTS_VIOLATION: str = "analytics.invariants.violation"
TOPIC_SCOPE_METRICS_UPDATED: str = "analytics.scope.metrics.updated"
TOPIC_SCOPE_META_READY: str = "analytics.scope.meta.ready"

__all__ = [
    "EVENTS_SCHEMA_VERSION",
    "TOPIC_ANALYTICS_DELTA_READY",
    "TOPIC_ANALYTICS_INVARIANTS_VIOLATION",
    "TOPIC_SCOPE_METRICS_UPDATED",
    "TOPIC_SCOPE_META_READY",
    # typy payloadów
    "BaseMeta",
    "DeltaReadyPayload",
    "InvariantsViolationPayload",
    "ScopeMetricsUpdatedPaths",
    "ScopeMetricsUpdatedMeta",
    "ScopeMetricsUpdatedPayload",
    "ScopeMetaReadyPaths",
    "ScopeMetaReadyPayload",
    # walidator
    "validate_event_payload",
]

# ──────────────────────────────────────────────────────────────────────────────
# Minimalne typy payloadów (TypedDict)
# ──────────────────────────────────────────────────────────────────────────────

class BaseMeta(TypedDict, total=False):
    ts: str              # ISO8601 UTC
    version: str         # np. EVENTS_SCHEMA_VERSION lub wersja nadawcy

# --- analytics.delta.ready ----------------------------------------------------

class DeltaReadyPayload(TypedDict, total=False):
    delta_report: Dict[str, Any]  # wymagane (walidator dopilnuje)
    meta: BaseMeta

# --- analytics.invariants.violation ------------------------------------------

class InvariantsViolationPayload(TypedDict, total=False):
    violations: Dict[str, Any]    # wymagane
    meta: BaseMeta

# --- analytics.scope.metrics.updated -----------------------------------------

class ScopeMetricsUpdatedPaths(TypedDict, total=False):
    metrics: str   # ścieżka do .glx/graphs/metrics.json

class ScopeMetricsUpdatedMeta(TypedDict, total=False):
    graph_hash: str
    metrics_hash: str
    ts: str
    version: str

class ScopeMetricsUpdatedPayload(TypedDict, total=False):
    kind: Literal["graph_metrics"]      # rekomendowane
    paths: ScopeMetricsUpdatedPaths
    meta: ScopeMetricsUpdatedMeta
    summary: Dict[str, Any]             # np. {"metrics": ["pagerank","betweenness"], "nodes": 1234}

# --- analytics.scope.meta.ready ----------------------------------------------

class ScopeMetaReadyPaths(TypedDict, total=False):
    json: str       # ścieżka do meta_<level>_<name>.json
    dot: str        # opcjonalnie .dot

class ScopeMetaReadyPayload(TypedDict, total=False):
    kind: Literal["meta_lens"]
    level: str
    name: str
    paths: ScopeMetaReadyPaths
    anchors_count: int
    window_metrics: Dict[str, Any]
    meta: BaseMeta

# ──────────────────────────────────────────────────────────────────────────────
# Walidator kształtu payloadu (lekki, bez zewnętrznych zależności)
# ──────────────────────────────────────────────────────────────────────────────

def _is_mapping(x: Any) -> bool:
    return isinstance(x, Mapping)

def _req_keys(payload: Mapping[str, Any], keys: List[str], errs: List[str]) -> None:
    for k in keys:
        if k not in payload:
            errs.append(f"missing key: '{k}'")

def validate_event_payload(topic: str, payload: Any) -> tuple[bool, List[str]]:
    """
    Minimalna walidacja kształtu ładunku dla zadanego topicu.
    Zwraca: (ok, errors[]). Nie rzuca wyjątków — przeznaczona do fast-path w publisherach.

    Uwaga: Walidacja jest celowo „łagodna” – sprawdza tylko kluczowe pola
    i typy bazowe, bez pełnych kontraktów domenowych.
    """
    errors: List[str] = []

    if not _is_mapping(payload):
        return False, ["payload must be a mapping"]

    if topic == TOPIC_ANALYTICS_DELTA_READY:
        _req_keys(payload, ["delta_report"], errors)
        if "delta_report" in payload and not _is_mapping(payload["delta_report"]):
            errors.append("delta_report must be a mapping")

    elif topic == TOPIC_ANALYTICS_INVARIANTS_VIOLATION:
        _req_keys(payload, ["violations"], errors)
        if "violations" in payload and not _is_mapping(payload["violations"]):
            errors.append("violations must be a mapping")

    elif topic == TOPIC_SCOPE_METRICS_UPDATED:
        # wszystkie pola są formalnie opcjonalne, ale rekomendujemy 'kind' i/lub 'paths'
        if "kind" in payload and payload["kind"] not in ("graph_metrics",):
            errors.append("kind must be 'graph_metrics' when present")
        if "paths" in payload and not _is_mapping(payload["paths"]):
            errors.append("paths must be a mapping")
        if "paths" in payload and _is_mapping(payload["paths"]):
            p = payload["paths"]
            if "metrics" in p and not isinstance(p["metrics"], str):
                errors.append("paths.metrics must be a string")
        if "meta" in payload and not _is_mapping(payload["meta"]):
            errors.append("meta must be a mapping")

    elif topic == TOPIC_SCOPE_META_READY:
        _req_keys(payload, ["level", "name", "paths"], errors)
        if "kind" in payload and payload["kind"] not in ("meta_lens",):
            errors.append("kind must be 'meta_lens' when present")
        if "level" in payload and not isinstance(payload["level"], str):
            errors.append("level must be a string")
        if "name" in payload and not isinstance(payload["name"], str):
            errors.append("name must be a string")
        if "paths" in payload and not _is_mapping(payload["paths"]):
            errors.append("paths must be a mapping")
        if "paths" in payload and _is_mapping(payload["paths"]):
            p = payload["paths"]
            if "json" not in p:
                errors.append("paths.json is required")
            elif not isinstance(p["json"], str):
                errors.append("paths.json must be a string")
            if "dot" in p and not isinstance(p["dot"], str):
                errors.append("paths.dot must be a string")
        if "anchors_count" in payload and not isinstance(payload["anchors_count"], int):
            errors.append("anchors_count must be an int")
        if "window_metrics" in payload and not _is_mapping(payload["window_metrics"]):
            errors.append("window_metrics must be a mapping")
        if "meta" in payload and not _is_mapping(payload["meta"]):
            errors.append("meta must be a mapping")

    else:
        errors.append(f"unknown topic: {topic!r}")

    return (len(errors) == 0), errors
