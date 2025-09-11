# glitchlab/analysis/exporters.py
"""
---
version: 2
kind: module
id: "analysis-exporters"
created_at: "2025-09-11"
name: "glitchlab.analysis.exporters"
author: "GlitchLab v2"
role: "HUD Bundle Exporter & Validator"
description: >
  Składa lekki DTO dla HUD/GUI na podstawie ctx.cache: metadane uruchomienia,
  graf AST (jeśli dostępny) i listę etapów pipeline z kluczami do miniatur/overlayów.
  Nie przenosi obrazów — tylko referencje do nich (cache keys). Zapewnia walidator struktury.
inputs:
  ctx_like:
    type: "Mapping|Ctx"
    fields:
      cache: {type: "dict[str,Any]", required: true}
      meta:  {type: "dict[str,Any]", required: false, fields: ["seed","versions","source"]}
outputs:
  bundle:
    run:    {id: "str", seed: "int|None", source: "dict", versions: "dict"}
    ast:    {type: "dict", desc: "graf z klucza 'ast/json' lub {}"}
    stages: "list[{i:int,name:str,t_ms:float|None,metrics_in:dict,metrics_out:dict,diff_stats:dict,keys:{in|out|diff|mosaic:str|None}}]"
    format: {notes: "list[str]", has_grid: "bool"}
  validate_warnings: {type: "list[str]", desc: "ostrzeżenia z validate_hud_bundle()"}
interfaces:
  exports: ["export_hud_bundle","validate_hud_bundle"]
  depends_on: ["re","uuid"]
  used_by: ["glitchlab.gui","glitchlab.core.pipeline","glitchlab.core.graph"]
policy:
  deterministic: true
  side_effects: false
constraints:
  - "nie dołącza danych binarnych; wyłącznie referencje (cache keys)"
  - "odporne na brakujące pola w cache/meta"
telemetry:
  cache_keys_scanned: "stage/{i}/*, ast/json, format/*, cfg/root, run/id"
hud:
  channels:
    bundle: "export_hud_bundle(ctx)"
license: "Proprietary"
---
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Set, Tuple
from uuid import uuid4
import re


__all__ = ["export_hud_bundle", "validate_hud_bundle"]


def _is_mapping(x: Any) -> bool:
    return isinstance(x, Mapping)


def _collect_stage_indices(cache: Mapping[str, Any]) -> List[int]:
    """
    Przeskanuj klucze cache i wyciągnij unikalne indeksy i z "stage/{i}/...".
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


def export_hud_bundle(ctx_like: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Buduje lekki DTO dla HUD/GUI:
      {
        "run": { "id", "seed", "source", "versions },
        "ast":  <ctx.cache["ast/json"] or {}>,
        "stages": [
           {
             "i", "name", "t_ms",
             "metrics_in", "metrics_out", "diff_stats",
             "keys": {"in","out","diff","mosaic"}
           }, ...
        ],
        "format": {"notes": [...], "has_grid": bool}
      }
    Nie dołącza danych obrazowych — GUI pobiera je po kluczach z ctx.cache.
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

        stages.append({
            "i": i,
            "name": name,
            "t_ms": float(t_ms) if isinstance(t_ms, (int, float)) else None,
            "metrics_in": metrics_in,
            "metrics_out": metrics_out,
            "diff_stats": diff_stats,
            "keys": {"in": k_in, "out": k_out, "diff": k_diff, "mosaic": k_mosaic},
        })

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
    Zwraca listę ostrzeżeń/błędów (strings). Pusta lista oznacza brak problemów krytycznych.
    Waliduje minimalny kontrakt oczekiwany przez HUD.
    """
    warns: List[str] = []

    def req(path: str, typ: Any) -> None:
        cur: Any = bundle
        for p in path.split("."):
            if not _is_mapping(cur) or p not in cur:
                warns.append(f"missing key: {path}")
                return
            cur = cur[p]
        if typ == "list":
            if not isinstance(cur, list):
                warns.append(f"invalid type at {path}: expected list")
        elif typ == "dict":
            if not _is_mapping(cur):
                warns.append(f"invalid type at {path}: expected dict")

    # wymagane korzenie
    req("run", "dict")
    req("stages", "list")
    # opcjonalne, ale sprawdzamy format
    if "ast" in bundle and not _is_mapping(bundle["ast"]):
        warns.append("invalid type: 'ast' must be a dict")

    # run.* podklucze
    if _is_mapping(bundle.get("run")):
        for sub in ("id", "versions"):
            if sub not in bundle["run"]:
                warns.append(f"missing key: run.{sub}")

    # stages[i] klucze
    stages = bundle.get("stages", [])
    if isinstance(stages, list):
        for idx, st in enumerate(stages):
            if not _is_mapping(st):
                warns.append(f"stage[{idx}]: not a dict")
                continue
            for k in ("i", "name", "keys"):
                if k not in st:
                    warns.append(f"stage[{idx}]: missing '{k}'")
            if "keys" in st and not _is_mapping(st["keys"]):
                warns.append(f"stage[{idx}].keys: must be a dict")
    else:
        warns.append("stages: not a list")

    # format.*
    if "format" in bundle:
        fmt = bundle["format"]
        if not _is_mapping(fmt):
            warns.append("format: must be a dict")
        else:
            if "notes" in fmt and not isinstance(fmt["notes"], list):
                warns.append("format.notes: must be a list")
            if "has_grid" in fmt and not isinstance(fmt["has_grid"], bool):
                warns.append("format.has_grid: must be a bool")

    return warns
