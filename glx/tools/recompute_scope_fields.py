#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
glx.tools.recompute_scope_fields — batch do globalnego przeliczenia metryk i odświeżenia cache pól

Cel:
- Przeliczyć globalne metryki grafowe na CAŁYM grafie projektu.
- Zaktualizować/odświeżyć cache pól (FieldCache), aby metasoczewki korzystały ze świeżych danych.
- (Opcjonalnie) opublikować zdarzenie BUS o aktualizacji metryk.

Zależności:
- glitchlab.analysis.project_graph
- glitchlab.analysis.graph_metrics
- glitchlab.analysis.field_cache
- (opcjonalnie) glitchlab.analysis.grammar.events  + BUS (jeśli dostępny w środowisku)
- stdlib only

Przykłady:
  # podstawowy przebieg: wykryj/utwórz project_graph, przelicz metrics, odśwież cache
  python -m glx.tools.recompute_scope_fields

  # przelicz wymuszając rebuild grafu i publikację na BUS
  python -m glx.tools.recompute_scope_fields --rebuild-graph --publish-bus

  # wskaż niestandardowy root repo
  python -m glx.tools.recompute_scope_fields --repo-root /path/to/repo

  # wykonaj bez publikacji BUS (domyślne) i z wymuszeniem przebudowy cache
  python -m glx.tools.recompute_scope_fields --force-cache
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Pomocnicze: artefakty, zapis atomowy, BUS
# ──────────────────────────────────────────────────────────────────────────────

def _import_artifacts():
    """Łagodny import resolvera katalogu .glx/."""
    try:
        import glitchlab.io.artifacts as art  # type: ignore
        return art
    except Exception:
        try:
            import io.artifacts as art  # type: ignore
            return art
        except Exception:
            return None


def _call0_or1(fn, arg):
    try:
        return fn(arg)  # type: ignore[arg-type]
    except TypeError:
        return fn()  # type: ignore[misc]


def _get_glx_dir(repo_root: Optional[Path] = None) -> Path:
    art = _import_artifacts()
    base = Path.cwd() if repo_root is None else Path(repo_root)
    if art:
        for name in ("ensure_glx_dir", "get_glx_dir", "ensure_artifacts_dir", "artifacts_dir"):
            if hasattr(art, name):
                glx = _call0_or1(getattr(art, name), base)
                p = Path(glx).resolve()
                p.mkdir(parents=True, exist_ok=True)
                return p
    p = (base / ".glx").resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=str(path.parent)) as tmp:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_p = Path(tmp.name)
    tmp_p.replace(path)


def _jsonify(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, (list, tuple)):
        return [ _jsonify(x) for x in obj ]
    if isinstance(obj, dict):
        return { k: _jsonify(v) for k, v in obj.items() }
    return obj


def _get_bus_publisher():
    """
    Zwraca publish(topic, payload) lub no-op. Obsługuje:
      - glitchlab.src.app.event_bus.publish
      - glitchlab.src.app.event_bus.EventBus().publish
    """
    try:
        from glitchlab.src.app.event_bus import publish as _pub  # type: ignore
        return lambda topic, payload: _pub(topic, payload)
    except Exception:
        pass
    try:
        from glitchlab.src.app.event_bus import EventBus  # type: ignore
        try:
            bus = EventBus.global_bus() if hasattr(EventBus, "global_bus") else EventBus()  # type: ignore[attr-defined]
            return lambda topic, payload: bus.publish(topic, payload)  # type: ignore[attr-defined]
        except Exception:
            pass
    except Exception:
        pass
    return lambda topic, payload: None


# ──────────────────────────────────────────────────────────────────────────────
# Importy modułów analizy (łagodne, z czytelnymi błędami)
# ──────────────────────────────────────────────────────────────────────────────

def _import_pg_mod():
    try:
        import glitchlab.analysis.project_graph as pg  # type: ignore
        return pg
    except Exception as e:
        print(f"[recompute][ERROR] Nie można zaimportować analysis.project_graph: {e}", file=sys.stderr)
        raise

def _import_metrics_mod():
    try:
        import glitchlab.analysis.graph_metrics as gm  # type: ignore
        return gm
    except Exception as e:
        print(f"[recompute][ERROR] Nie można zaimportować analysis.graph_metrics: {e}", file=sys.stderr)
        raise

def _import_field_cache():
    try:
        from glitchlab.analysis.field_cache import FieldCache  # type: ignore
        return FieldCache
    except Exception as e:
        print(f"[recompute][WARN] FieldCache niedostępny: {e}", file=sys.stderr)
        return None

def _import_events_topics():
    try:
        from glitchlab.analysis.grammar.events import TOPIC_SCOPE_METRICS_UPDATED  # type: ignore
        return {"metrics_updated": TOPIC_SCOPE_METRICS_UPDATED}
    except Exception:
        return {"metrics_updated": "analytics.scope.metrics.updated"}


# ──────────────────────────────────────────────────────────────────────────────
# Główna logika: przelicz metryki, odśwież cache
# ──────────────────────────────────────────────────────────────────────────────

def _load_json(p: Path) -> Optional[Dict[str, Any]]:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _ensure_project_graph(repo_root: Path, glx_dir: Path, rebuild: bool) -> Tuple[Dict[str, Any], Path]:
    pg_mod = _import_pg_mod()
    graphs_dir = glx_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    pg_path = graphs_dir / "project_graph.json"

    if rebuild or not pg_path.exists():
        # zbuduj od nowa
        try:
            # preferuj API modułu
            if hasattr(pg_mod, "build_project_graph"):
                pg = pg_mod.build_project_graph(repo_root)  # type: ignore[attr-defined]
                # zapisz modułowym writerem jeśli jest
                if hasattr(pg_mod, "save_project_graph"):
                    pg_mod.save_project_graph(pg, repo_root=repo_root)  # type: ignore[attr-defined]
                else:
                    payload = _jsonify(pg)
                    _atomic_write_json(pg_path, payload if isinstance(payload, dict) else {"graph": payload})
            else:
                # brak API — spróbuj legacy CLI
                raise RuntimeError("build_project_graph() not found")
        except Exception as e:
            print(f"[recompute][ERROR] Budowa project_graph nie powiodła się: {e}", file=sys.stderr)
            raise
    # wczytaj JSON do payloadu
    payload = _load_json(pg_path)
    if payload is None:
        # fallback: jeśli mamy w pamięci obiekt pg z earlier branch — serializuj
        raise RuntimeError(f"Brak/Nieczytelny {pg_path}")
    return payload, pg_path


def _compute_and_write_metrics(repo_root: Path, glx_dir: Path, project_graph_payload: Dict[str, Any]) -> Tuple[Dict[str, Any], Path]:
    gm = _import_metrics_mod()
    graphs_dir = glx_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    out_path = graphs_dir / "metrics.json"

    # preferuj compute_* API modułu
    result: Optional[Dict[str, Any]] = None
    try:
        if hasattr(gm, "compute_graph_metrics"):
            # compute_graph_metrics akceptuje ProjectGraph lub payload — przekaż payload
            result = gm.compute_graph_metrics(project_graph_payload)  # type: ignore[arg-type]
        elif hasattr(gm, "compute_metrics"):
            result = gm.compute_metrics(project_graph_payload)  # type: ignore[misc]
    except Exception as e:
        print(f"[recompute][ERROR] compute_graph_metrics() nie powiodło się: {e}", file=sys.stderr)
        raise

    if result is None:
        raise RuntimeError("Nie udało się uzyskać wyników metryk (None).")

    # zapis modułowym writerem jeśli jest
    try:
        if hasattr(gm, "save_graph_metrics"):
            p = gm.save_graph_metrics(result, repo_root=repo_root)  # type: ignore[attr-defined]
            # upewnij się, że mamy payload z dysku (meta może zostać ubogacona)
            reloaded = _load_json(Path(p)) if isinstance(p, (str, Path)) else None
            if reloaded:
                result = reloaded
                out_path = Path(p)
        else:
            _atomic_write_json(out_path, result)
    except Exception as e:
        print(f"[recompute][WARN] save_graph_metrics() nie powiodło się — zapis bezpośredni: {e}", file=sys.stderr)
        _atomic_write_json(out_path, result)

    return result, out_path


def _delta_fingerprint(glx_dir: Path) -> Optional[str]:
    # opcjonalnie pobierz odcisk z delta_report.json jeżeli obecny
    dr = glx_dir / "delta_report.json"
    if not dr.exists():
        return None
    try:
        obj = json.loads(dr.read_text(encoding="utf-8"))
    except Exception:
        return None
    # preferuj pola 'fingerprint'/'hash', w przeciwnym razie zbuduj deterministyczny hash lokalny
    for k in ("fingerprint", "hash", "delta_hash", "report_hash"):
        v = obj.get(k)
        if isinstance(v, str) and v:
            return v
    try:
        # stabilna serializacja
        raw = json.dumps(obj, separators=(",", ":"), sort_keys=True)
        # prosty skrót (bez zależności zewn.)
        import hashlib
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    except Exception:
        return None


def _refresh_field_cache(repo_root: Path, glx_dir: Path,
                         project_graph_payload: Dict[str, Any],
                         metrics_payload: Dict[str, Any],
                         force_cache: bool) -> Tuple[Optional[Path], Dict[str, Any]]:
    """
    Próbuje użyć FieldCache, a gdy niedostępny — tworzy minimalne podsumowanie cache.
    Zwraca (ścieżka field_cache.json lub None, summary dict).
    """
    fc_cls = _import_field_cache()
    graphs_dir = glx_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    out_path = graphs_dir / "field_cache.json"

    summary: Dict[str, Any] = {}
    delta_fp = _delta_fingerprint(glx_dir)

    if fc_cls is None:
        # Minimalne podsumowanie (fallback)
        summary = {
            "version": "v1",
            "sources": {
                "project_graph_hash": metrics_payload.get("_meta", {}).get("graph_hash") or project_graph_payload.get("_meta", {}).get("hash"),
                "metrics_hash": metrics_payload.get("_meta", {}).get("metrics_hash"),
                "delta_fingerprint": delta_fp,
            },
            "fields": {},  # właściwe wektory są rozwiązywane przez analysis/fields.py w czasie użycia
        }
        _atomic_write_json(out_path, summary)
        return out_path, summary

    # Spróbuj użyć publicznego API FieldCache w różnych wariantach (zachowawczo)
    try:
        # konstruktory: FieldCache(repo_root=...), FieldCache(glx_dir=...) lub bezarg.
        try:
            fc = fc_cls(repo_root=repo_root)  # type: ignore[call-arg]
        except Exception:
            try:
                fc = fc_cls(glx_dir=glx_dir)  # type: ignore[call-arg]
            except Exception:
                fc = fc_cls()  # type: ignore[call-arg]
        # przekazanie źródeł
        # preferowane metody:
        if hasattr(fc, "bind_sources"):
            fc.bind_sources(project_graph_payload, metrics_payload, delta_fp=delta_fp)  # type: ignore[attr-defined]
        elif hasattr(fc, "update_sources"):
            fc.update_sources(project_graph_payload, metrics_payload, delta_fingerprint=delta_fp)  # type: ignore[attr-defined]
        elif hasattr(fc, "set_sources"):
            fc.set_sources(project_graph_payload, metrics_payload, delta_fingerprint=delta_fp)  # type: ignore[attr-defined]

        # przebudowa/rewalidacja
        if force_cache and hasattr(fc, "rebuild_all"):
            fc.rebuild_all()  # type: ignore[attr-defined]
        elif hasattr(fc, "revalidate_or_rebuild"):
            fc.revalidate_or_rebuild(force=force_cache)  # type: ignore[attr-defined]
        elif hasattr(fc, "rebuild"):
            fc.rebuild(force=force_cache)  # type: ignore[attr-defined]

        # zapis
        if hasattr(fc, "save"):
            p = fc.save(glx_dir=glx_dir)  # type: ignore[attr-defined]
            out_p = Path(p) if isinstance(p, (str, Path)) else out_path
            out_path = out_p
        elif hasattr(fc, "write_json"):
            p = fc.write_json(out_path)  # type: ignore[attr-defined]
            out_p = Path(p) if isinstance(p, (str, Path)) else out_path
            out_path = out_p
        else:
            # spróbuj uzyskać summary i zapisać samodzielnie
            if hasattr(fc, "summary"):
                summary = getattr(fc, "summary")  # type: ignore[attr-defined]
                if not isinstance(summary, dict):
                    summary = _jsonify(summary)
            elif hasattr(fc, "to_json"):
                summary = fc.to_json()  # type: ignore[attr-defined]
            else:
                summary = {
                    "version": "v1",
                    "sources": {
                        "project_graph_hash": metrics_payload.get("_meta", {}).get("graph_hash"),
                        "metrics_hash": metrics_payload.get("_meta", {}).get("metrics_hash"),
                        "delta_fingerprint": delta_fp,
                    },
                }
            _atomic_write_json(out_path, _jsonify(summary))
        # jeżeli nie zapisaliśmy summary wyżej — spróbuj ponownie odczytać z dysku, aby zwrócić spójny obraz
        if not summary:
            j = _load_json(out_path)
            summary = j if isinstance(j, dict) else {"path": str(out_path)}
        return out_path, _jsonify(summary)
    except Exception as e:
        print(f"[recompute][WARN] FieldCache: fallback do minimalnego zapisu: {e}", file=sys.stderr)
        summary = {
            "version": "v1",
            "sources": {
                "project_graph_hash": metrics_payload.get("_meta", {}).get("graph_hash") or project_graph_payload.get("_meta", {}).get("hash"),
                "metrics_hash": metrics_payload.get("_meta", {}).get("metrics_hash"),
                "delta_fingerprint": delta_fp,
            },
            "fields": {},
        }
        _atomic_write_json(out_path, summary)
        return out_path, summary


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _cli(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="glx-recompute-scope-fields",
        description="Przelicz globalne metryki grafu projektu i odśwież cache pól metasoczewek."
    )
    ap.add_argument("--repo-root", default=None, help="Root repo (domyślnie CWD)")
    ap.add_argument("--rebuild-graph", action="store_true", help="Wymuś przebudowę project_graph.json")
    ap.add_argument("--force-cache", action="store_true", help="Wymuś pełną przebudowę FieldCache")
    ap.add_argument("--publish-bus", action="store_true", help="Opublikuj zdarzenie BUS po aktualizacji metryk")
    ap.add_argument("--quiet", action="store_true", help="Tylko minimalny JSON na stdout")
    args = ap.parse_args(argv)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path.cwd().resolve()
    glx_dir = _get_glx_dir(repo_root)

    # 1) Upewnij się, że mamy project_graph
    pg_payload, pg_path = _ensure_project_graph(repo_root, glx_dir, rebuild=args.rebuild_graph)

    # 2) Przelicz i zapisz metrics.json
    metrics_payload, metrics_path = _compute_and_write_metrics(repo_root, glx_dir, pg_payload)

    # 3) Odśwież FieldCache
    fc_path, fc_summary = _refresh_field_cache(repo_root, glx_dir, pg_payload, metrics_payload, force_cache=args.force_cache)

    # 4) (Opcjonalnie) publikacja BUS
    bus_info = None
    if args.publish_bus:
        topics = _import_events_topics()
        publish = _get_bus_publisher()
        payload = {
            "graph_meta": metrics_payload.get("_meta", {}),
            "paths": {
                "project_graph": str(pg_path),
                "metrics": str(metrics_path),
                "field_cache": str(fc_path) if fc_path else None,
            }
        }
        try:
            publish(topics["metrics_updated"], payload)
            bus_info = {"topic": topics["metrics_updated"], "ok": True}
        except Exception as e:
            bus_info = {"topic": topics["metrics_updated"], "ok": False, "error": str(e)}

    # 5) Wyjście
    out = {
        "ok": True,
        "repo_root": str(repo_root),
        "written": {
            "project_graph": str(pg_path),
            "metrics": str(metrics_path),
            "field_cache": str(fc_path) if fc_path else None,
        },
        "field_cache_summary": fc_summary,
        "bus": bus_info,
    }
    if not args.quiet:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(json.dumps({"ok": True, "written": out["written"]}, ensure_ascii=False))
    return 0


def main():
    return _cli()


if __name__ == "__main__":
    sys.exit(main())
