# glitchlab/analysis/exporters.py
# -*- coding: utf-8 -*-
"""
Jednolite eksporty artefaktów analizy (graf, metryki, metasoczewki, cache pól) dla GlitchLab.

Nowe API (DRY; zależne od analysis.formats):
  - write_project_graph(graph_or_payload, *, repo_root=None, strict=True) -> Path
  - write_graph_metrics(metrics: dict, *, graph_hash:str|None=None, repo_root=None, strict=True) -> Path
  - write_field_cache(summary: dict, *, graph_hash:str|None=None, metrics_hash:str|None=None, repo_root=None, strict=True) -> Path
  - write_meta_lens(level: str, name: str, subgraph_payload: dict, *,
                    spec: dict|None=None, anchors: list[str]|None=None, metrics: dict|None=None,
                    dot: str|None=None, repo_root=None, strict=True) -> dict[str, Path]

Back-compat:
  - export_hud_bundle(ctx_like) -> dict
  - validate_hud_bundle(bundle) -> list[str]

Konwencje ścieżek (domyślnie w .glx/graphs/):
  - project_graph.json
  - metrics.json
  - field_cache.json
  - meta_<level>_<name>.json (+ .dot jeśli podano)

Uwaga: Funkcje 'write_*' wykonują walidację przez analysis.formats.
       W trybie strict=True rzucą ValueError przy błędach walidacji.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Set, Tuple
from dataclasses import asdict, is_dataclass
from pathlib import Path
from uuid import uuid4
from datetime import datetime, timezone
import re
import json
import os
import tempfile

# ─────────────────────────────────────────────────────────────────────────────
# Formats (walidacje, wersje, hash)
# ─────────────────────────────────────────────────────────────────────────────

try:
    from glitchlab.analysis import formats as F  # preferowany import
except Exception:
    # fallback lokalny (np. w środowisku bez pakietu)
    from . import formats as F  # type: ignore

# ─────────────────────────────────────────────────────────────────────────────
# Artefakty: integracja z glitchlab.io.artifacts (z łagodnym fallbackiem)
# ─────────────────────────────────────────────────────────────────────────────

def _import_artifacts():
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
    """Wywołaj fn() albo fn(arg) — dopasuj do API resolvera katalogu."""
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
        for name in ("ensure_glx_dir", "get_glx_dir", "ensure_artifacts_dir", "artifacts_dir"):
            if hasattr(art, name):
                glx_dir = _call0_or1(getattr(art, name), base)
                p = Path(glx_dir)
                p.mkdir(parents=True, exist_ok=True)
                return p.resolve()
    p = (base / ".glx").resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _graphs_dir(repo_root: Optional[Path] = None) -> Path:
    p = _get_glx_dir(repo_root) / "graphs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=str(path.parent)) as tmp:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def _atomic_write_text(path: Path, text: str) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"))


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    b = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    _atomic_write_bytes(path, b)


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize_name(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("/", "_").replace("\\", "_").replace(":", "_")
    s = re.sub(r"[^A-Za-z0-9_.\-]+", "_", s)
    return s or "auto"

def _as_payload(obj: Any) -> Dict[str, Any]:
    """
    Zamień dowolny wspierany obiekt na dict JSON-ready:
     - dataclass → asdict
     - obiekty mające .to_json() lub .to_dict() → użyj
     - mapping → zwróć
    """
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "to_json") and callable(getattr(obj, "to_json")):  # type: ignore[attr-defined]
        return dict(getattr(obj, "to_json")())  # type: ignore
    if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):  # type: ignore[attr-defined]
        return dict(getattr(obj, "to_dict")())  # type: ignore
    if isinstance(obj, Mapping):
        return dict(obj)
    raise TypeError("Unsupported object type for export; expected mapping/dataclass or JSON-like")

# ─────────────────────────────────────────────────────────────────────────────
# Nowe API: jednolite eksporty
# ─────────────────────────────────────────────────────────────────────────────

def write_project_graph(graph_or_payload: Any, *,
                        repo_root: Optional[Path] = None,
                        strict: bool = True) -> Path:
    """
    Zapisuje .glx/graphs/project_graph.json z wersją i meta.graph_hash.
    Akceptuje dataclass/obj z .to_dict()/.to_json() lub gotowy payload {"graph": {...}}.
    """
    graphs = _graphs_dir(repo_root)
    payload = _as_payload(graph_or_payload)
    # Jeżeli przekazano tylko "graph", opakuj wersją i meta
    if "graph" not in payload and ("nodes" in payload and "edges" in payload):
        payload = {"graph": {"nodes": payload["nodes"], "edges": payload["edges"]}}
    payload.setdefault("version", F.PROJECT_GRAPH_VERSION)
    # policz hash na samej sekcji graph (stabilny)
    graph_obj = payload.get("graph") or {}
    ghash = F.json_hash(graph_obj if isinstance(graph_obj, Mapping) else {})
    meta = dict(payload.get("meta") or {})
    meta.setdefault("graph_hash", ghash)
    meta.setdefault("ts", _iso_now())
    payload["meta"] = meta

    ok, errs = F.validate_project_graph(payload)
    if not ok and strict:
        raise ValueError("project_graph validation errors: " + "; ".join(errs))

    out_p = graphs / "project_graph.json"
    _atomic_write_json(out_p, payload)
    return out_p


def write_graph_metrics(metrics: Mapping[str, Mapping[str, float]], *,
                        graph_hash: Optional[str] = None,
                        repo_root: Optional[Path] = None,
                        strict: bool = True) -> Path:
    """
    Zapisuje .glx/graphs/metrics.json z meta.graph_hash (jeśli podano) i wersją.
    """
    graphs = _graphs_dir(repo_root)
    payload: Dict[str, Any] = {
        "version": F.GRAPH_METRICS_VERSION,
        "metrics": {str(k): {str(n): float(v) for n, v in m.items()} for k, m in metrics.items()},
        "meta": {"ts": _iso_now()},
    }
    if graph_hash:
        payload["meta"]["graph_hash"] = str(graph_hash)

    # opcjonalnie: metrics_hash (pomocne dla cache)
    payload["meta"]["metrics_hash"] = F.json_hash(payload["metrics"])

    ok, errs = F.validate_graph_metrics(payload)
    if not ok and strict:
        raise ValueError("graph_metrics validation errors: " + "; ".join(errs))

    out_p = graphs / "metrics.json"
    _atomic_write_json(out_p, payload)
    return out_p


def write_field_cache(summary: Mapping[str, Mapping[str, Any]], *,
                      graph_hash: Optional[str] = None,
                      metrics_hash: Optional[str] = None,
                      repo_root: Optional[Path] = None,
                      strict: bool = True) -> Path:
    """
    Zapisuje .glx/graphs/field_cache.json – same streszczenia (hash, p50 itd.)
    """
    graphs = _graphs_dir(repo_root)
    payload: Dict[str, Any] = {
        "version": F.FIELD_CACHE_VERSION,
        "fields": {str(k): dict(v) for k, v in summary.items()},
        "meta": {"ts": _iso_now()},
    }
    if graph_hash:
        payload["meta"]["graph_hash"] = str(graph_hash)
    if metrics_hash:
        payload["meta"]["metrics_hash"] = str(metrics_hash)

    ok, errs = F.validate_field_cache(payload)
    if not ok and strict:
        raise ValueError("field_cache validation errors: " + "; ".join(errs))

    out_p = graphs / "field_cache.json"
    _atomic_write_json(out_p, payload)
    return out_p


def write_meta_lens(level: str,
                    name: str,
                    subgraph_payload: Mapping[str, Any],
                    *,
                    spec: Optional[Mapping[str, Any]] = None,
                    anchors: Optional[List[str]] = None,
                    metrics: Optional[Mapping[str, Any]] = None,
                    dot: Optional[str] = None,
                    repo_root: Optional[Path] = None,
                    strict: bool = True) -> Dict[str, Path]:
    """
    Zapisuje meta_<level>_<name>.json (+ .dot jeżeli podano 'dot').
    subgraph_payload powinien zawierać "nodes":[], "edges":[] (lub pełny to_json grafu cząstkowego).
    Sekcja meta zawiera spec/anchors/metrics.
    """
    graphs = _graphs_dir(repo_root)
    lvl = _sanitize_name(level or "custom")
    nm = _sanitize_name(name or "auto")
    out_json = graphs / f"meta_{lvl}_{nm}.json"
    out_dot = graphs / f"meta_{lvl}_{nm}.dot"

    # Zbuduj payload JSON dla meta soczewki
    # Akceptujemy różne wejścia: jeśli subgraph ma już 'nodes/edges' na toplevelu – użyj wprost.
    if "nodes" in subgraph_payload and "edges" in subgraph_payload:
        payload = {"nodes": list(subgraph_payload["nodes"]), "edges": list(subgraph_payload["edges"])}
    elif "graph" in subgraph_payload and isinstance(subgraph_payload["graph"], Mapping):
        g = subgraph_payload["graph"]
        payload = {"nodes": list(g.get("nodes") or []), "edges": list(g.get("edges") or [])}
    else:
        raise ValueError("subgraph_payload must contain 'nodes' and 'edges' (or 'graph' with those)")

    meta_obj = dict(subgraph_payload.get("meta") or subgraph_payload.get("_meta") or {})
    if spec is not None:
        meta_obj["spec"] = dict(spec)
    if anchors is not None:
        meta_obj["anchors"] = list(anchors)
    if metrics is not None:
        meta_obj["metrics"] = dict(metrics)
    meta_obj.setdefault("ts", _iso_now())
    payload["meta"] = meta_obj  # dopuszczamy zarówno 'meta', jak i '_meta'; walidator akceptuje obie formy

    ok, errs = F.validate_meta_lens(payload)
    if not ok and strict:
        raise ValueError("meta_lens validation errors: " + "; ".join(errs))

    _atomic_write_json(out_json, payload)
    result = {"json": out_json}

    if isinstance(dot, str) and dot.strip():
        _atomic_write_text(out_dot, dot)
        result["dot"] = out_dot

    return result

# ─────────────────────────────────────────────────────────────────────────────
# Back-compat: HUD bundle (pozostawione, nie mieszamy z nowymi API)
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    # nowe API
    "write_project_graph",
    "write_graph_metrics",
    "write_field_cache",
    "write_meta_lens",
    # HUD (legacy)
    "export_hud_bundle",
    "validate_hud_bundle",
]

# ===== Legacy HUD exporter (niezależny od nowych funkcji) ====================

def _is_mapping(x: Any) -> bool:
    return isinstance(x, Mapping)


def _collect_stage_indices(cache: Mapping[str, Any]) -> List[int]:
    """
    Przeskanuj klucze cache i wyciągnij unikalne indeksy i "stage/{i}/...".
    """
    pat = re.compile(r"^stage/(\d+)/")
    seen: Set[int] = set()
    for k in cache.keys():
        m = pat.match(k)
        if m:
            try:
                seen.add(int(m.group(1)))
            except Exception:
                pass
    return sorted(seen)


def _first_present(cache: Mapping[str, Any], keys: List[str], default: Any = None) -> Any:
    for k in keys:
        if k in cache:
            return cache[k]
    return default


def _overlay_key_for_stage(cache: Mapping[str, Any], i: int) -> Optional[str]:
    """
    Zwraca istniejący klucz z gotowym overlayem dla etapu i (jeśli jest w cache).
    """
    prefix = f"stage/{i}"
    candidates = [
        f"{prefix}/overlay",
        f"{prefix}/overlay_rgb",
        f"{prefix}/mosaic_overlay",
        f"{prefix}/overlay:rgb",
    ]
    for k in candidates:
        if k in cache:
            return k
    return None


def export_hud_bundle(ctx_like: Mapping[str, Any]) -> Dict[str, Any]:
    """
    (Legacy) Buduje lekki DTO dla HUD/GUI.
    """
    # wyciągnij cache
    if hasattr(ctx_like, "cache"):
        cache = getattr(ctx_like, "cache", {})
    else:
        cache = ctx_like.get("cache", {})

    if not _is_mapping(cache):
        raise TypeError("export_hud_bundle: ctx_like must have mapping 'cache'")

    # Run meta
    run_id = cache.get("run/id") or uuid4().hex
    seed = None
    versions = {}
    source = {}

    # spróbuj z ctx_like.meta
    meta = getattr(ctx_like, "meta", None) if hasattr(ctx_like, "meta") else ctx_like.get("meta")
    if _is_mapping(meta):
        seed = meta.get("seed", seed)
        versions = meta.get("versions", versions) or versions
        source = meta.get("source", source) or source

    # jeśli brak seed, sprawdź też w cfg
    cfg = cache.get("cfg/root")
    if seed is None and _is_mapping(cfg):
        seed = cfg.get("seed")

    # AST JSON
    ast_json = cache.get("ast/json", {})

    # Stages
    indices = _collect_stage_indices(cache)
    stages: List[Dict[str, Any]] = []
    for i in indices:
        prefix = f"stage/{i}"
        name = cache.get(f"{prefix}/name") or f"stage_{i}"
        t_ms = cache.get(f"{prefix}/t_ms")

        metrics_in = cache.get(f"{prefix}/metrics_in") or {}
        metrics_out = cache.get(f"{prefix}/metrics_out") or {}
        diff_stats = cache.get(f"{prefix}/diff_stats") or {}

        # Klucze obrazów/overlay
        k_in = f"{prefix}/in" if f"{prefix}/in" in cache else None
        k_out = f"{prefix}/out" if f"{prefix}/out" in cache else None
        k_diff = f"{prefix}/diff" if f"{prefix}/diff" in cache else None
        k_mosaic = f"{prefix}/mosaic" if f"{prefix}/mosaic" in cache else None
        k_overlay = _overlay_key_for_stage(cache, i)

        stage_entry: Dict[str, Any] = {
            "i": i,
            "name": name,
            "t_ms": float(t_ms) if isinstance(t_ms, (int, float)) else None,
            "metrics_in": metrics_in,
            "metrics_out": metrics_out,
            "diff_stats": diff_stats,
            "keys": {"in": k_in, "out": k_out, "diff": k_diff, "mosaic": k_mosaic, "overlay": k_overlay},
        }

        # Overlay hint z block_stats → core overlay (bez efektów ubocznych)
        has_block_stats = f"{prefix}/block_stats" in cache
        has_mosaic_info = (k_mosaic is not None) or (f"{prefix}/mosaic_meta" in cache) or (f"{prefix}/mosaic_map" in cache)
        if k_overlay is None and has_block_stats and has_mosaic_info:
            stage_entry["overlay_hint"] = {
                "can_build": True,
                "source": "block_stats",
                "map_spec": {
                    "R": ["entropy", [0.0, 0.0]],
                    "G": ["edges",   [0.0, 0.0]],
                    "B": ["mean",    [0.0, 0.0]],
                },
                "inputs": {
                    "block_stats_key": f"{prefix}/block_stats",
                    "mosaic_key": k_mosaic or _first_present(cache, [f"{prefix}/mosaic_meta", f"{prefix}/mosaic_map"]),
                    "image_key": k_out or k_in,
                },
                "suggested_output_key": f"{prefix}/overlay_rgb",
            }

        stages.append(stage_entry)

    # Format forensics (opcjonalne)
    notes = []
    n1 = cache.get("format/notes")
    if isinstance(n1, list):
        notes.extend([str(x) for x in n1])
    has_grid = bool("format/jpg_grid" in cache)

    bundle: Dict[str, Any] = {
        "run": {
            "id": run_id,
            "seed": int(seed) if isinstance(seed, (int,)) else seed,
            "source": source,
            "versions": versions,
        },
        "ast": ast_json if _is_mapping(ast_json) else {},
        "stages": stages,
        "format": {"notes": notes, "has_grid": has_grid},
    }

    return bundle


def validate_hud_bundle(bundle: Mapping[str, Any]) -> List[str]:
    """
    (Legacy) Zwraca listę ostrzeżeń/błędów (strings). Pusta lista = brak problemów krytycznych.
    """
    warns: List[str] = []

    def req(path: str, typ: Any) -> None:
        cur: Any = bundle
        for p in path.split("."):
            if not isinstance(cur, Mapping) or p not in cur:
                warns.append(f"missing key: {path}")
                return
            cur = cur[p]
        if typ == "list":
            if not isinstance(cur, list):
                warns.append(f"invalid type at {path}: expected list")
        elif typ == "dict":
            if not isinstance(cur, Mapping):
                warns.append(f"invalid type at {path}: expected dict")

    # wymagane korzenie
    req("run", "dict")
    req("stages", "list")
    # opcjonalne, ale sprawdzamy format
    if "ast" in bundle and not isinstance(bundle["ast"], Mapping):
        warns.append("invalid type: 'ast' must be a dict")

    # run.* podklucze
    if isinstance(bundle.get("run"), Mapping):
        for sub in ("id", "versions"):
            if sub not in bundle["run"]:
                warns.append(f"missing key: run.{sub}")

    # stages[i] klucze
    stages = bundle.get("stages", [])
    if isinstance(stages, list):
        for idx, st in enumerate(stages):
            if not isinstance(st, Mapping):
                warns.append(f"stage[{idx}]: not a dict")
                continue
            for k in ("i", "name", "keys"):
                if k not in st:
                    warns.append(f"stage[{idx}]: missing '{k}'")
            if "keys" in st and not isinstance(st["keys"], Mapping):
                warns.append(f"stage[{idx}].keys: must be a dict")
            # overlay_hint format (jeśli obecny)
            if "overlay_hint" in st:
                oh = st["overlay_hint"]
                if not isinstance(oh, Mapping):
                    warns.append(f"stage[{idx}].overlay_hint: must be a dict")
                else:
                    if "can_build" in oh and not isinstance(oh["can_build"], bool):
                        warns.append(f"stage[{idx}].overlay_hint.can_build: must be a bool")
                    if "map_spec" in oh and not isinstance(oh["map_spec"], Mapping):
                        warns.append(f"stage[{idx}].overlay_hint.map_spec: must be a dict")
                    if "inputs" in oh and not isinstance(oh["inputs"], Mapping):
                        warns.append(f"stage[{idx}].overlay_hint.inputs: must be a dict")
    else:
        warns.append("stages: not a list")

    # format.*
    if "format" in bundle:
        fmt = bundle["format"]
        if not isinstance(fmt, Mapping):
            warns.append("format: must be a dict")
        else:
            if "notes" in fmt and not isinstance(fmt["notes"], list):
                warns.append("format.notes: must be a list")
            if "has_grid" in fmt and not isinstance(fmt["has_grid"], bool):
                warns.append("format.has_grid: must be a bool")

    return warns
