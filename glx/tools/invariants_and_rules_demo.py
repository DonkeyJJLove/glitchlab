# glitchlab/glx/tools/invariants_and_rules_demo.py
from __future__ import annotations
import json, sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# łagodne importy (działamy i wewnątrz pakietu, i „goło”)
try:
    from glitchlab.analysis.field_cache import FieldCache  # type: ignore
    from glitchlab.analysis.fields import resolve_field  # type: ignore
except Exception:
    from analysis.field_cache import FieldCache  # type: ignore
    from analysis.fields import resolve_field  # type: ignore


# BUS publisher (jak w reporting.py – fallback na no-op)
def _get_bus():
    try:
        from glitchlab.src.app.event_bus import publish as _pub  # type: ignore
        return lambda topic, payload: _pub(topic, payload)
    except Exception:
        pass
    try:
        from glitchlab.src.app.event_bus import EventBus  # type: ignore
        bus = EventBus.global_bus() if hasattr(EventBus, "global_bus") else EventBus()
        return lambda topic, payload: bus.publish(topic, payload)  # type: ignore
    except Exception:
        pass
    return lambda topic, payload: None


BUS = _get_bus()


def _glx(repo_root: Path) -> Path:
    d = (repo_root / ".glx").resolve()
    d.mkdir(parents=True, exist_ok=True)
    (d / "graphs").mkdir(parents=True, exist_ok=True)
    return d


def _load_json(p: Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


# --------------------------------------------------------------------
# 1) „próbka dowodu” I1 / I4 na mikrografie
# I1: suma ΔZ na ścieżce = Z_out − Z_in  (konserwacja skoku abstrakcji)
# I4: ruch refaktoryzacyjny nie pogarsza funkcjonału J (tu: J=β̄ na ścieżce)
# --------------------------------------------------------------------

def _proof_I1_I4_demo() -> Dict[str, Any]:
    # mały graf: A( Z=1 ), B( Z=2 ), C( Z=3 ) z krawędziami A->B->C
    nodes = {"A": {"Z": 1, "S": 10, "H": 6}, "B": {"Z": 2, "S": 9, "H": 9}, "C": {"Z": 3, "S": 7, "H": 12}}
    edges = [("A", "B"), ("B", "C")]
    path = ["A", "B", "C"]
    # I1
    dZ_e = {("A", "B"): +1, ("B", "C"): +1}
    sum_dZ = sum(dZ_e[e] for e in edges)
    Z_in, Z_out = nodes["A"]["Z"], nodes["C"]["Z"]
    I1_ok = (sum_dZ == (Z_out - Z_in))

    # I4 — β = H/(S+H); „refactor” B: (S,H) := (10,8) → β spada
    def beta(n):
        S = nodes[n]["S"];
        H = nodes[n]["H"];
        t = max(1, S + H);
        return H / t

    beta_path_before = sum(beta(n) for n in path) / len(path)
    # modyfikacja B
    nodes["B"]["S"], nodes["B"]["H"] = 10, 8
    beta_path_after = sum(beta(n) for n in path) / len(path)
    I4_ok = (beta_path_after <= beta_path_before + 1e-12)

    return {
        "I1": {"ok": bool(I1_ok), "sum_dZ": sum_dZ, "Z_in": Z_in, "Z_out": Z_out},
        "I4": {"ok": bool(I4_ok), "beta_before": beta_path_before, "beta_after": beta_path_after},
        "path": path,
    }


# --------------------------------------------------------------------
# 2) Preset reguł BUS: „hot-bridge consolidation” (β-dominacja)
#   Heurystyka:
#     - mamy community_label (label propagation) i betweenness_approx (z metrics.json),
#     - wyznaczamy „mosty” = węzły łączące różne społeczności (na podstawie krawędzi),
#     - filtr: betweenness w top P% oraz β > T,
#     - publikujemy sugestię konsolidacji (np. scal interfejs / rozdziel).
# --------------------------------------------------------------------

def _hot_bridge_preset(glx_dir: Path, repo_root: Path,
                       percentile: float = 90.0, beta_threshold: float = 0.6) -> Dict[str, Any]:
    metrics = _load_json(glx_dir / "graphs" / "metrics.json") or {}
    pg = _load_json(glx_dir / "graphs" / "project_graph.json") or {}

    # wyjmujemy mapy
    comm = (metrics.get("metrics") or {}).get("community_label") or {}
    btw = (metrics.get("metrics") or {}).get("betweenness_approx") or {}

    # budujemy sąsiedztwo z grafu
    nbrs = {}
    for e in (pg.get("edges") or []):
        u, v = e.get("src"), e.get("dst")
        if not u or not v: continue
        nbrs.setdefault(u, set()).add(v)
        nbrs.setdefault(v, set()).add(u)

    # β z AST snapshotu (H/(S+H))
    ast_idx = _load_json(glx_dir / "graphs" / "ast_index.json") or {}
    # map node_id->beta (dla file:…)
    beta = {}
    files = ast_idx.get("files") or ast_idx
    if isinstance(files, dict):
        for path, rec in files.items():
            S = float(rec.get("S", 0));
            H = float(rec.get("H", 0));
            t = max(1.0, S + H)
            beta[f"file:{Path(path).as_posix()}"] = H / t

    # most = węzeł z sąsiadami w >=2 różnych społecznościach
    candidates = []
    for u, ns in nbrs.items():
        comms = {comm.get(n) for n in ns if n in comm}
        if len([c for c in comms if c is not None]) >= 2:
            candidates.append(u)

    # percentyl betweenness
    vals = sorted([btw.get(u, 0.0) for u in candidates])
    if vals:
        i = int(round((max(0.0, min(100.0, percentile)) / 100.0) * (len(vals) - 1)))
        cut = vals[i]
    else:
        cut = 1e9

    suggestions = []
    for u in candidates:
        if btw.get(u, 0.0) < cut:
            continue
        b = beta.get(u, 0.0)
        if b < beta_threshold:
            continue
        suggestions.append({
            "node": u,
            "betweenness": btw.get(u, 0.0),
            "beta": b,
            "kind": "hot_bridge_consolidation",
            "message": "Węzeł łączy wiele społeczności, β-dominacja wysoka — rozważ konsolidację interfejsu lub separację kontraktów.",
        })

    return {"preset": "hot_bridge_consolidation", "thresholds": {"pct_betweenness": percentile, "beta": beta_threshold},
            "suggestions": suggestions}


# --------------------------------------------------------------------
# 3) Dystans/podobieństwo S/H/Z – przykład użycia na repo (opcjonalny)
# --------------------------------------------------------------------

def _example_shz_field(repo_root: Path, target_percentile: float = 95.0) -> Dict[str, float]:
    fc = FieldCache(repo_root=repo_root) if FieldCache else None
    spec = {"name": "dist_shz", "target_percentile": target_percentile, "weights": {"S": 1, "H": 1, "Z": 1},
            "norm": "l2"}
    return resolve_field(spec, cache=fc, repo_root=repo_root)


# --------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------

def _cli(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="I1/I4 demo + BUS preset (hot-bridge) + SHZ distance example")
    ap.add_argument("--repo-root", "--repo", dest="repo_root", default=".", help="root projektu (z .glx/)")
    ap.add_argument("--target-percentile", type=float, default=95.0,
                    help="cel S/H/Z jako percentyl (gdy brak target_node/vector)")
    ap.add_argument("--publish", action="store_true", help="opublikuj BUS analytics.scope.meta.ready z presetami")
    args = ap.parse_args(argv)

    root = Path(args.repo_root).resolve()
    glx = _glx(root)

    # 1) I1/I4 proof
    proof = _proof_I1_I4_demo()
    print(json.dumps({"I1_I4_demo": proof}, ensure_ascii=False, indent=2))

    # 2) preset reguł
    preset = _hot_bridge_preset(glx, root)
    print(json.dumps({"preset": preset["preset"], "suggestions": preset["suggestions"][:5]}, ensure_ascii=False,
                     indent=2))

    if args.publish:
        try:
            payload = {"presets": [preset], "ts": None}
            BUS("analytics.scope.meta.ready", payload)
            print("[BUS] published analytics.scope.meta.ready (presets)")
        except Exception as e:
            print(f"[BUS] publish failed: {e}")

    # 3) SHZ distance example
    try:
        dist = _example_shz_field(root, args.target_percentile)
        # wypisz top-10 „najbardziej podobnych” (czyli najniższy dystans → po normalizacji: wartość blisko 0)
        sample = sorted(dist.items(), key=lambda kv: kv[1])[:10]
        print(json.dumps({"dist_shz_top10": sample}, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"[dist_shz] skipped: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(_cli())
