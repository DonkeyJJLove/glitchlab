# glitchlab/app/exporter.py
from __future__ import annotations
import json, os, pathlib
from typing import Any, Dict


def export_hud_bundle(ctx_like: Any) -> Dict[str, Any]:
    cache = getattr(ctx_like, "cache", {}) if ctx_like is not None else {}
    stages = []
    i = 0
    while True:
        k = f"stage/{i}/t_ms"
        if k not in cache: break
        stages.append({
            "i": i,
            "t_ms": cache.get(k),
            "keys": {
                "in": f"stage/{i}/in",
                "out": f"stage/{i}/out",
                "diff": f"stage/{i}/diff",
                "mosaic": f"stage/{i}/mosaic",
            },
            "metrics_in": cache.get(f"stage/{i}/metrics_in", {}),
            "metrics_out": cache.get(f"stage/{i}/metrics_out", {}),
            "diff_stats": cache.get(f"stage/{i}/diff_stats", {}),
        })
        i += 1
    run = getattr(ctx_like, "meta", {}).copy() if ctx_like else {}
    run.setdefault("seed", getattr(ctx_like, "seed", None))
    ast_json = cache.get("ast/json")
    fmt = {"notes": cache.get("format/notes", []), "has_grid": bool(cache.get("format/jpg_grid"))}
    return {"run": run, "ast": ast_json, "stages": stages, "format": fmt}


def save_layout(path: str, d: dict) -> None:
    p = pathlib.Path(path);
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, indent=2), encoding="utf-8")


def load_layout(path: str) -> dict:
    p = pathlib.Path(path)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}
