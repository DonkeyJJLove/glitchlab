# glitchlab/analysis/reporting.py
# -*- coding: utf-8 -*-
"""
Reporting helpers:
- Zapis artefaktów do .glx/ (delta_report.json, commit_analysis.json, spec_state.json)
  z wykorzystaniem glitchlab.io.artifacts (jeśli dostępny), z atomowym zapisem.
- Publikacja zdarzeń BUS (łagodna integracja, no-op jeśli BUS niedostępny):
    * "analytics.delta.ready"               payload: delta_report
    * "analytics.invariants.violation"      payload: invariants_result
    * "analytics.scope.metrics.updated"     payload: scope metrics (meta, hash, ścieżka)
    * "analytics.scope.meta.ready"          payload: meta-lens (spec, anchors, metrics, ścieżka)
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union


# ─────────────────────────────────────────────────────────────────────────────
# Artefakty: integracja z glitchlab.io.artifacts (z łagodnym fallbackiem)
# ─────────────────────────────────────────────────────────────────────────────

def _import_artifacts():
    try:
        # preferuj import pakietowy
        import glitchlab.io.artifacts as art  # type: ignore
        return art
    except Exception:
        try:
            # alternatywna ścieżka (gdy uruchamiane poza pakietem)
            import io.artifacts as art  # type: ignore
            return art
        except Exception:
            return None


def _call0_or1(fn, arg):
    """Wywołaj fn() albo fn(arg) — w zależności od sygnatury."""
    try:
        return fn(arg)  # type: ignore[arg-type]
    except TypeError:
        return fn()  # type: ignore[misc]


def _get_glx_dir(repo_root: Optional[Path] = None) -> Path:
    """
    Ustal katalog .glx/:
      - jeśli dostępny glitchlab.io.artifacts → użyj jego resolvera,
      - w przeciwnym razie: <repo_root or CWD>/.glx
    """
    art = _import_artifacts()

    base = Path.cwd() if repo_root is None else Path(repo_root)
    if art:
        # różne warianty API resolverów
        for name in ("ensure_glx_dir", "get_glx_dir", "ensure_artifacts_dir", "artifacts_dir"):
            if hasattr(art, name):
                glx_dir = _call0_or1(getattr(art, name), base)
                p = Path(glx_dir)
                p.mkdir(parents=True, exist_ok=True)
                return p.resolve()

    p = (base / ".glx").resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _get_graphs_dir(repo_root: Optional[Path] = None) -> Path:
    """Zwraca katalog .glx/graphs (tworzy jeśli nie istnieje)."""
    glx = _get_glx_dir(repo_root)
    graphs = glx / "graphs"
    graphs.mkdir(parents=True, exist_ok=True)
    return graphs


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    """
    Atomowy zapis JSON (UTF-8, \n, indent=2) do pliku 'path'.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=str(path.parent)) as tmp:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_sha256(payload: Dict[str, Any]) -> str:
    """Stabilny hash treści JSON (sort_keys, separators bez spacji)."""
    b = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(b).hexdigest()


def _load_json_if_exists(p: Path) -> Dict[str, Any]:
    try:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """Płytko-głębokie scalanie słowników (dict w dict)."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)  # type: ignore[index]
        else:
            dst[k] = v
    return dst


# ─────────────────────────────────────────────────────────────────────────────
# BUS (publish) — lekka integracja z event_bus; fallback na no-op
# ─────────────────────────────────────────────────────────────────────────────

def _get_bus_publisher():
    """
    Zwraca funkcję publish(topic, payload) lub no-op, jeśli BUS niedostępny.
    Obsługiwane warianty:
      - glitchlab.src.app.event_bus.publish(topic, payload)
      - glitchlab.src.app.event_bus.EventBus().publish(topic, payload)
      - fallback: no-op
    """
    # wariant 1: globalna funkcja publish
    try:
        from glitchlab.src.app.event_bus import publish as _publish  # type: ignore
        return lambda topic, payload: _publish(topic, payload)
    except Exception:
        pass

    # wariant 2: instancja EventBus
    try:
        from glitchlab.src.app.event_bus import EventBus  # type: ignore
        try:
            if hasattr(EventBus, "global_bus"):
                bus = EventBus.global_bus()  # type: ignore[attr-defined]
            else:
                bus = EventBus()
            return lambda topic, payload: bus.publish(topic, payload)  # type: ignore[attr-defined]
        except Exception:
            pass
    except Exception:
        pass

    # fallback: no-op
    return lambda topic, payload: None


_PUBLISH = _get_bus_publisher()


# ─────────────────────────────────────────────────────────────────────────────
# API: delta / commit / spec / invariants (jak wcześniej)
# ─────────────────────────────────────────────────────────────────────────────

def save_delta_report(delta_report: Dict[str, Any],
                      repo_root: Optional[Path] = None) -> Path:
    """
    Zapisuje .glx/delta_report.json i publikuje BUS: analytics.delta.ready
    """
    glx_dir = _get_glx_dir(repo_root)
    out_p = glx_dir / "delta_report.json"
    _atomic_write_json(out_p, delta_report)
    # publish
    try:
        _PUBLISH("analytics.delta.ready", dict(delta_report=delta_report, path=str(out_p), ts=_now_iso()))
    except Exception:
        pass
    return out_p


def save_commit_analysis(commit_analysis: Dict[str, Any],
                         repo_root: Optional[Path] = None) -> Path:
    """
    Zapisuje .glx/commit_analysis.json (bez publikacji BUS).
    """
    glx_dir = _get_glx_dir(repo_root)
    out_p = glx_dir / "commit_analysis.json"
    _atomic_write_json(out_p, commit_analysis)
    return out_p


def save_spec_state(spec_state: Dict[str, Any],
                    repo_root: Optional[Path] = None) -> Path:
    """
    Zapisuje .glx/spec_state.json (bez publikacji BUS).
    """
    glx_dir = _get_glx_dir(repo_root)
    out_p = glx_dir / "spec_state.json"
    _atomic_write_json(out_p, spec_state)
    return out_p


def _update_spec_scope(section: str, payload: Dict[str, Any], repo_root: Optional[Path]) -> None:
    """
    Uaktualnij .glx/spec_state.json w sekcji scope.{metrics|meta}.
    """
    glx_dir = _get_glx_dir(repo_root)
    spec_p = glx_dir / "spec_state.json"
    cur = _load_json_if_exists(spec_p)
    envelope = {"scope": {section: payload}}
    merged = _deep_update(cur, envelope)
    _atomic_write_json(spec_p, merged)


def publish_invariants_violation(violations_payload: Dict[str, Any]) -> None:
    """
    Publikuje BUS: analytics.invariants.violation
    """
    try:
        _PUBLISH("analytics.invariants.violation", dict(violations=violations_payload, ts=_now_iso()))
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# NOWE API: metasoczewki (metrics + meta-lens)
# ─────────────────────────────────────────────────────────────────────────────

def save_scope_metrics(metrics_payload: Dict[str, Any],
                       repo_root: Optional[Path] = None) -> Path:
    """
    Zapisz .glx/graphs/metrics.json i opublikuj BUS: analytics.scope.metrics.updated.
    Aktualizuje .glx/spec_state.json: scope.metrics{ ts, hash, meta }.
    """
    graphs = _get_graphs_dir(repo_root)
    out_p = graphs / "metrics.json"
    _atomic_write_json(out_p, metrics_payload)

    # meta + hash (jeśli brak w payload, policz z całego JSON)
    meta = metrics_payload.get("meta") or metrics_payload.get("_meta") or {}
    ghash = meta.get("graph_hash") or _json_sha256(metrics_payload)

    env = dict(
        ts=_now_iso(),
        path=str(out_p),
        meta=meta,
        hash=ghash,
    )

    # publish
    try:
        _PUBLISH("analytics.scope.metrics.updated", env)
    except Exception:
        pass

    # spec_state
    try:
        _update_spec_scope("metrics", env, repo_root)
    except Exception:
        pass

    return out_p


def notify_scope_metrics_from_file(metrics_json_path: Union[str, Path],
                                   repo_root: Optional[Path] = None) -> None:
    """
    Odczytaj istniejący metrics.json i opublikuj BUS + zaktualizuj spec_state.
    Przydatne, jeśli zapisem zarządza inny moduł (analysis/graph_metrics.py).
    """
    p = Path(metrics_json_path)
    payload = _load_json_if_exists(p)
    if not payload:
        return
    meta = payload.get("meta") or payload.get("_meta") or {}
    ghash = meta.get("graph_hash") or _json_sha256(payload)
    env = dict(ts=_now_iso(), path=str(p), meta=meta, hash=ghash)

    try:
        _PUBLISH("analytics.scope.metrics.updated", env)
    except Exception:
        pass

    try:
        _update_spec_scope("metrics", env, repo_root)
    except Exception:
        pass


def save_scope_meta(level: str,
                    name: str,
                    meta_payload: Dict[str, Any],
                    *,
                    dot_str: Optional[str] = None,
                    repo_root: Optional[Path] = None) -> Tuple[Path, Optional[Path]]:
    """
    Zapisz .glx/graphs/meta_<level>_<name>.json (+ opcjonalnie .dot)
    i opublikuj BUS: analytics.scope.meta.ready.
    Aktualizuje .glx/spec_state.json: scope.meta{ ts, level, name, anchors, metrics }.

    Uwaga: jeśli generowanie artefaktów wykonuje analysis/scope_meta.py,
    możesz zamiast tego użyć notify_scope_meta_from_file(path).
    """
    safe_name = str(name).replace("/", "_").replace(":", "_")
    graphs = _get_graphs_dir(repo_root)
    json_p = graphs / f"meta_{level}_{safe_name}.json"
    _atomic_write_json(json_p, meta_payload)

    dot_p: Optional[Path] = None
    if dot_str is not None:
        dot_p = graphs / f"meta_{level}_{safe_name}.dot"
        dot_p.parent.mkdir(parents=True, exist_ok=True)
        dot_p.write_text(dot_str, encoding="utf-8")

    _publish_scope_meta(json_p, meta_payload, level, safe_name, repo_root)
    return json_p, dot_p


def notify_scope_meta_from_file(meta_json_path: Union[str, Path],
                                repo_root: Optional[Path] = None) -> None:
    """
    Odczytaj istniejący meta_*.json i opublikuj BUS + zaktualizuj spec_state.
    """
    p = Path(meta_json_path)
    payload = _load_json_if_exists(p)
    if not payload:
        return
    # heurystyka nazwy/poziomu z nazwy pliku
    fname = p.name
    level = "custom"
    name = "auto"
    if fname.startswith("meta_") and fname.endswith(".json"):
        core = fname[len("meta_"):-len(".json")]
        parts = core.split("_", 1)
        if len(parts) == 2:
            level, name = parts[0], parts[1]
        else:
            level = core

    _publish_scope_meta(p, payload, level, name, repo_root)


def _publish_scope_meta(json_path: Path,
                        payload: Dict[str, Any],
                        level: str,
                        name: str,
                        repo_root: Optional[Path]) -> None:
    meta = payload.get("_meta") or payload.get("meta") or {}

    env = dict(
        ts=_now_iso(),
        level=level,
        name=name,
        path=str(json_path),
        spec=meta.get("spec"),
        anchors=meta.get("anchors"),
        metrics=meta.get("metrics"),
    )

    try:
        _PUBLISH("analytics.scope.meta.ready", env)
    except Exception:
        pass

    try:
        _update_spec_scope("meta", env, repo_root)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Zbiorczy zapis wielu artefaktów (rozszerzony o metasoczewki)
# ─────────────────────────────────────────────────────────────────────────────

def save_all(delta_report: Optional[Dict[str, Any]] = None,
             commit_analysis: Optional[Dict[str, Any]] = None,
             spec_state: Optional[Dict[str, Any]] = None,
             invariants_result: Optional[Dict[str, Any]] = None,
             *,
             scope_metrics: Optional[Dict[str, Any]] = None,
             scope_meta: Optional[Tuple[str, str, Dict[str, Any]]] = None,
             repo_root: Optional[Path] = None) -> Dict[str, Optional[Path]]:
    """
    Wygodny zapis wielu artefaktów + publikacja odpowiednich zdarzeń.
    Dodatkowo:
      - scope_metrics:      payload do zapisania w .glx/graphs/metrics.json + publish
      - scope_meta:         krotka (level, name, payload) do zapisania + publish

    Zwraca mapę ścieżek zapisanych artefaktów (None, gdy nie zapisano).
    """
    paths: Dict[str, Optional[Path]] = {
        "delta_report": None,
        "commit_analysis": None,
        "spec_state": None,
        "scope_metrics": None,
        "scope_meta_json": None,
    }

    if delta_report is not None:
        paths["delta_report"] = save_delta_report(delta_report, repo_root=repo_root)

    if commit_analysis is not None:
        paths["commit_analysis"] = save_commit_analysis(commit_analysis, repo_root=repo_root)

    if spec_state is not None:
        paths["spec_state"] = save_spec_state(spec_state, repo_root=repo_root)

    if invariants_result is not None:
        publish_invariants_violation(invariants_result)

    if scope_metrics is not None:
        paths["scope_metrics"] = save_scope_metrics(scope_metrics, repo_root=repo_root)

    if scope_meta is not None:
        lvl, name, payload = scope_meta
        json_p, _ = save_scope_meta(lvl, name, payload, repo_root=repo_root)
        paths["scope_meta_json"] = json_p

    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Mini-CLI (rozszerzony)
# ─────────────────────────────────────────────────────────────────────────────

def _read_json_file(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def _cli(argv=None) -> int:
    """
    Przykłady:
      python -m glitchlab.analysis.reporting --delta delta.json
      python -m glitchlab.analysis.reporting --delta delta.json --commit commit.json
      python -m glitchlab.analysis.reporting --delta delta.json --spec spec.json --viols viols.json
      # metasoczewki:
      python -m glitchlab.analysis.reporting --scope-metrics .glx/graphs/metrics.json
      python -m glitchlab.analysis.reporting --scope-meta .glx/graphs/meta_module_glitchlab.core.json
    """
    import argparse

    ap = argparse.ArgumentParser(prog="reporting", description="Zapis artefaktów .glx/ + publikacja BUS")
    ap.add_argument("--repo-root", default=None, help="Root repo (domyślnie CWD)")
    ap.add_argument("--delta", default=None, help="Ścieżka do delta_report.json (źródło)")
    ap.add_argument("--commit", default=None, help="Ścieżka do commit_analysis.json (źródło)")
    ap.add_argument("--spec", default=None, help="Ścieżka do spec_state.json (źródło)")
    ap.add_argument("--viols", default=None, help="Ścieżka do invariants_violation.json (źródło)")

    # nowe
    ap.add_argument("--scope-metrics", default=None, help="Ścieżka do graphs/metrics.json (opublikuj i zarejestruj)")
    ap.add_argument("--scope-meta", default=None, help="Ścieżka do graphs/meta_*.json (opublikuj i zarejestruj)")

    args = ap.parse_args(argv)
    root = Path(args.repo_root).resolve() if args.repo_root else None

    # standardowe
    delta = _read_json_file(Path(args.delta)) if args.delta else None
    commit = _read_json_file(Path(args.commit)) if args.commit else None
    spec = _read_json_file(Path(args.spec)) if args.spec else None
    viols = _read_json_file(Path(args.viols)) if args.viols else None

    out = save_all(delta_report=delta, commit_analysis=commit,
                   spec_state=spec, invariants_result=viols, repo_root=root)

    # metasoczewki - tryb "publish istniejących plików"
    if args.scope_metrics:
        notify_scope_metrics_from_file(Path(args.scope_metrics), repo_root=root)
        out["scope_metrics"] = Path(args.scope_metrics)
    if args.scope_meta:
        notify_scope_meta_from_file(Path(args.scope_meta), repo_root=root)
        out["scope_meta_json"] = Path(args.scope_meta)

    # wypisz krótką informację dla CI/logów
    payload = {k: (str(v) if isinstance(v, Path) else (str(v) if v is not None else None)) for k, v in out.items()}
    print(json.dumps({"ok": True, "written": payload}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
