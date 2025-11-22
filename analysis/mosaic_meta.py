# glitchlab/analysis/mosaic_meta.py
from __future__ import annotations

import json, math, os, hashlib, tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

__all__ = [
    "MosaicCore", "Cell", "MosaicMeta",
    "build_mosaic_meta", "save_mosaic_meta",
    "cli_main"
]

# ──────────────────────────────────────────────────────────────────────────────
# Modele
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Cell:
    id: int
    center: Tuple[int, int]   # (x, y) px w canvasie
    row: int                  # dedukowane
    col: int                  # dedukowane

@dataclass(frozen=True)
class MosaicCore:
    mode: str                 # "square" | "hex"
    size: Tuple[int, int]     # (W, H)
    cells: List[Cell]         # uporządkowane lewo→prawo, góra→dół
    raster: Optional[Any] = None

@dataclass
class MosaicMeta:
    version: str
    meta: Dict[str, Any]
    cells: Dict[int, Dict[str, Any]]
    edges: List[Tuple[int, int]]


# ──────────────────────────────────────────────────────────────────────────────
# Artefakty (.glx) — łagodna integracja
# ──────────────────────────────────────────────────────────────────────────────

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

def _ensure_glx_dir(root: Optional[Path]) -> Path:
    art = _import_artifacts()
    base = Path.cwd() if root is None else Path(root)
    if art:
        for name in ("ensure_glx_dir", "get_glx_dir", "ensure_artifacts_dir", "artifacts_dir"):
            if hasattr(art, name):
                try:
                    p = Path(getattr(art, name)(base))
                    p.mkdir(parents=True, exist_ok=True)
                    return p.resolve()
                except Exception:
                    pass
    p = (base / ".glx").resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p

def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=str(path.parent)) as tmp:
        tmp.write(data); tmp.flush(); os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


# ──────────────────────────────────────────────────────────────────────────────
# „Coercers” — elastyczne pobranie core/edge/roi z różnych raportów
# (kompaktowa wersja zgodna z mostem Φ/Ψ)
# ──────────────────────────────────────────────────────────────────────────────

def _try_get(d: Mapping[str, Any], *path: str, default=None):
    cur: Any = d
    for k in path:
        if not isinstance(cur, Mapping) or k not in cur:
            return default
        cur = cur[k]
    return cur

def _deduce_grid_xy(cells_raw: List[Mapping[str, Any]]) -> Tuple[List[int], List[int]]:
    xs = sorted({int(c["center"][0]) for c in cells_raw if isinstance(c.get("center"), (list, tuple))})
    ys = sorted({int(c["center"][1]) for c in cells_raw if isinstance(c.get("center"), (list, tuple))})
    return xs, ys

def _coerce_core(delta_or_core: Mapping[str, Any]) -> MosaicCore:
    # 1) bezpośrednio „core”/„mosaic”
    for key in ("core", "mosaic", "repo_mosaic", "mosaic_core"):
        core = _try_get(delta_or_core, key)
        if isinstance(core, Mapping) and "cells" in core and "size" in core:
            mode = str(core.get("mode", "square")).lower()
            size = tuple(core.get("size", (1024, 768)))  # type: ignore
            cells_raw = list(core.get("cells", []))      # type: ignore
            xs, ys = _deduce_grid_xy(cells_raw)
            # zbuduj Cell(row,col)
            # indeks r,c dedukujemy przez najbliższe pozycje w xs, ys
            xs_i = {x: i for i, x in enumerate(xs)}
            ys_i = {y: i for i, y in enumerate(ys)}
            cells: List[Cell] = []
            for c in cells_raw:
                cid = int(c.get("id", len(cells)))
                x, y = int(c["center"][0]), int(c["center"][1])
                r, co = ys_i.get(y, 0), xs_i.get(x, 0)
                cells.append(Cell(id=cid, center=(x, y), row=r, col=co))
            # porządek: r→c
            cells.sort(key=lambda k: (k.row, k.col))
            return MosaicCore(mode=mode, size=size, cells=cells, raster=None)

    # 2) grid rows/cols → równomierne centra
    rows = int(_try_get(delta_or_core, "grid", "rows", default=_try_get(delta_or_core, "layout", "rows", default=0)) or 0)
    cols = int(_try_get(delta_or_core, "grid", "cols", default=_try_get(delta_or_core, "layout", "cols", default=0)) or 0)
    size = tuple(_try_get(delta_or_core, "size", default=(1024, 768)))  # type: ignore
    W, H = int(size[0]), int(size[1])
    cells: List[Cell] = []
    if rows > 0 and cols > 0:
        sx, sy = W / max(1, cols), H / max(1, rows)
        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                cx, cy = int(round((c + 0.5) * sx)), int(round((r + 0.5) * sy))
                cells.append(Cell(id=idx, center=(cx, cy), row=r, col=c))
    return MosaicCore(mode="square", size=(W, H), cells=cells, raster=None)

def _coerce_edges_per_cell(delta: Mapping[str, Any], N: int) -> Optional[List[float]]:
    for key in ("edges_per_cell", "edge", "edges"):
        arr = _try_get(delta, key)
        if isinstance(arr, (list, tuple)) and len(arr) >= N:
            out = [float(x) if isinstance(x, (int, float)) else 0.0 for x in arr[:N]]
            # clamp 0..1
            return [0.0 if (math.isnan(v) or v < 0) else (1.0 if v > 1 else v) for v in out]
    # fallback: files[].edge|impact|dS/dH/dZ
    files = _try_get(delta, "files") or _try_get(delta, "changed_files")
    if isinstance(files, list) and files:
        vals: List[float] = []
        for it in files[:N]:
            try:
                if "edge" in it: v = float(it["edge"])
                elif "impact" in it: v = float(it["impact"])
                else:
                    v = abs(float(it.get("dS", 0))) + abs(float(it.get("dH", 0))) + 0.25 * abs(float(it.get("dZ", 0)))
            except Exception:
                v = 0.0
            vals.append(max(0.0, v))
        mx = max(vals) if vals else 1.0
        vals = [x / mx if mx > 0 else 0.0] + [0.0] * max(0, N - len(vals))
        return vals[:N]
    return None

def _coerce_block_stats(delta: Mapping[str, Any]) -> Dict[Tuple[int, int], Dict[str, float]]:
    obj = _try_get(delta, "block_stats") or _try_get(delta, "blocks", "stats") or _try_get(delta, "mosaic", "block_stats")
    out: Dict[Tuple[int, int], Dict[str, float]] = {}
    if obj is None:
        return out
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            ij = None
            if isinstance(k, (list, tuple)) and len(k) == 2:
                ij = (int(k[0]), int(k[1]))
            elif isinstance(k, str):
                for sep in (",", ";", "|", " "):
                    if sep in k:
                        a, b = k.split(sep, 1)
                        ij = (int(a.strip()), int(b.strip()))
                        break
            if ij and isinstance(v, Mapping):
                out[ij] = {str(kk): float(vv) for kk, vv in v.items() if isinstance(vv, (int, float))}
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            if isinstance(item, (list, tuple)) and len(item) == 3 and isinstance(item[2], Mapping):
                i, j = int(item[0]), int(item[1])
                v = {str(kk): float(vv) for kk, vv in item[2].items() if isinstance(vv, (int, float))}
                out[(i, j)] = v
    return out

def _coerce_roi(delta: Mapping[str, Any], N: int) -> Optional[List[float]]:
    roi = _try_get(delta, "roi") or _try_get(delta, "roi_mask")
    if roi is None:
        idxs = _try_get(delta, "roi_indices") or _try_get(delta, "hot", "indices")
        if isinstance(idxs, (list, tuple)) and idxs:
            out = [0.0] * N
            for x in idxs:
                try:
                    i = int(x)
                    if 0 <= i < N:
                        out[i] = 1.0
                except Exception:
                    pass
            return out
        return None
    if isinstance(roi, (list, tuple)) and len(roi) == N:
        out = []
        for v in roi:
            try:
                x = float(v)
                out.append(0.0 if (math.isnan(x) or x < 0) else (1.0 if x > 1 else x))
            except Exception:
                out.append(0.0)
        return out
    # potraktuj jako indeksy
    out = [0.0] * N
    for x in list(roi):
        try:
            i = int(x)
            if 0 <= i < N:
                out[i] = 1.0
        except Exception:
            pass
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Budowa siatki, sąsiedztwa i cech strukturalnych
# ──────────────────────────────────────────────────────────────────────────────

def _grid_dims(cells: List[Cell]) -> Tuple[int, int]:
    rows = (max(c.row for c in cells) + 1) if cells else 0
    cols = (max(c.col for c in cells) + 1) if cells else 0
    return rows, cols

def _adj_square(rows: int, cols: int, diag: bool=False) -> List[Tuple[int,int]]:
    edges: List[Tuple[int,int]] = []
    def idx(r: int, c: int) -> int: return r*cols + c
    for r in range(rows):
        for c in range(cols):
            u = idx(r,c)
            for dr, dc in ((0,1),(1,0),(-1,0),(0,-1)):
                rr, cc = r+dr, c+dc
                if 0 <= rr < rows and 0 <= cc < cols:
                    v = idx(rr, cc)
                    if u < v: edges.append((u,v))
            if diag:
                for dr, dc in ((1,1),(1,-1),(-1,1),(-1,-1)):
                    rr, cc = r+dr, c+dc
                    if 0 <= rr < rows and 0 <= cc < cols:
                        v = idx(rr, cc)
                        if u < v: edges.append((u,v))
    return edges

def _adj_hex_odd_r(rows: int, cols: int) -> List[Tuple[int,int]]:
    """
    Odd-r offset (częste w bibliotekach). 6-sąsiedztwo.
    """
    edges: List[Tuple[int,int]] = []
    def idx(r: int, c: int) -> int: return r*cols + c
    for r in range(rows):
        for c in range(cols):
            u = idx(r,c)
            parity = r & 1
            neigh = [ (r, c-1), (r, c+1),
                      (r-1, c-parity), (r-1, c+1-parity),
                      (r+1, c-parity), (r+1, c+1-parity) ]
            for rr, cc in neigh:
                if 0 <= rr < rows and 0 <= cc < cols:
                    v = idx(rr,cc)
                    if u < v: edges.append((u,v))
    return edges

def _ring_index_square(r: int, c: int, rows: int, cols: int) -> int:
    cr, cc = (rows-1)/2.0, (cols-1)/2.0
    return int(max(abs(r - cr), abs(c - cc)) + 0.5)

def _ring_index_hex(r: int, c: int, rows: int, cols: int) -> int:
    # przybliżenie: metryka „chebyshev-axial” po odd-r
    cr, cc = (rows-1)/2.0, (cols-1)/2.0
    dr, dc = r - cr, c - cc
    return int(max(abs(dr), abs(dc), abs(dr + dc*0.5)) + 0.5)

def _label_propagation(n: int, edges: List[Tuple[int,int]], weight: Optional[List[float]]=None, iters: int=10) -> List[int]:
    """
    Bardzo proste LP: każdy node przyjmuje najczęstszą etykietę sąsiadów (ważoną, jeśli weight istnieje).
    """
    labels = list(range(n))
    adj: List[List[int]] = [[] for _ in range(n)]
    for u,v in edges:
        adj[u].append(v); adj[v].append(u)
    for _ in range(iters):
        changed = 0
        for u in range(n):
            votes: Dict[int, float] = {}
            for v in adj[u]:
                w = 1.0
                if weight is not None:
                    # waga = 1 + średnia z (w_u, w_v) — promuje „gorące” kafelki
                    w = 1.0 + 0.5*((weight[u] if u < len(weight) else 0.0) + (weight[v] if v < len(weight) else 0.0))
                lab = labels[v]
                votes[lab] = votes.get(lab, 0.0) + w
            if votes:
                best = max(votes.items(), key=lambda kv: (kv[1], -kv[0]))[0]
                if best != labels[u]:
                    labels[u] = best; changed += 1
        if changed == 0:
            break
    # re-numeracja do małych ID
    uniq = {}
    next_id = 0
    out = []
    for lab in labels:
        if lab not in uniq:
            uniq[lab] = next_id; next_id += 1
        out.append(uniq[lab])
    return out

def _quantiles(vals: List[float], qs=(0.25, 0.5, 0.75, 0.9, 0.99)) -> Dict[str, float]:
    v = sorted(x for x in vals if isinstance(x, (int, float)))
    n = len(v)
    if n == 0:
        return {f"q{int(q*100)}": 0.0 for q in qs}
    def qv(q: float) -> float:
        i = min(n-1, max(0, int(round(q*(n-1)))))
        return float(v[i])
    return {f"q{int(q*100)}": qv(q) for q in qs}

# ──────────────────────────────────────────────────────────────────────────────
# Główna budowa meta warstwy
# ──────────────────────────────────────────────────────────────────────────────

def build_mosaic_meta(delta_or_core: Mapping[str, Any]) -> MosaicMeta:
    core = _coerce_core(delta_or_core)
    rows, cols = _grid_dims(core.cells)
    N = rows * cols

    edges_vec = _coerce_edges_per_cell(delta_or_core, N) or [0.0]*N
    roi = _coerce_roi(delta_or_core, N) or [0.0]*N
    block_stats = _coerce_block_stats(delta_or_core)

    # adjacency
    if core.mode.startswith("hex"):
        adj = _adj_hex_odd_r(rows, cols)
        ring_of = lambda r,c: _ring_index_hex(r,c,rows,cols)
        max_deg = 6
    else:
        adj = _adj_square(rows, cols, diag=False)
        ring_of = lambda r,c: _ring_index_square(r,c,rows,cols)
        max_deg = 4

    # map (r,c)->cell.id (przyjęliśmy sort r→c, więc idx==r*cols+c, ale zabezpieczmy)
    rc_to_id: Dict[Tuple[int,int], int] = {}
    for i, cell in enumerate(core.cells):
        rc_to_id[(cell.row, cell.col)] = cell.id if cell.id is not None else i

    # cechy per-cell
    W, H = core.size
    cells_payload: Dict[int, Dict[str, Any]] = {}
    deg = [0]*N
    for u,v in adj:
        deg[u] += 1; deg[v] += 1

    edges_local: List[float] = []
    entr_local: List[float] = []

    # przygotuj szybki dostęp do block_stats
    # zakładamy mapę (bx,by) -> {...}; nasze r,c to by=row, bx=col
    def _bs(r: int, c: int, key: str) -> Optional[float]:
        v = block_stats.get((c, r))
        if v and key in v:
            try: return float(v[key])
            except Exception: return None
        return None

    rings: List[int] = []

    for i, cell in enumerate(core.cells):
        r, c = cell.row, cell.col
        x, y = cell.center
        u, v = (x / max(1, W), y / max(1, H))
        ring = ring_of(r, c)
        rings.append(ring)
        border = (r==0 or c==0 or r==rows-1 or c==cols-1)
        el = _bs(r, c, "edges"); hl = _bs(r, c, "entropy")
        if el is None: el = float(edges_vec[i] if i < len(edges_vec) else 0.0)
        edges_local.append(el if isinstance(el, (int,float)) else 0.0)
        entr_local.append(hl if isinstance(hl, (int,float)) else 0.0)
        cells_payload[i] = dict(
            id=i,
            row=r, col=c, x=x, y=y, u=round(u,6), v=round(v,6),
            degree=deg[i], max_degree=max_deg,
            ring=ring, border=border,
            edge=round(float(el or 0.0),6),
            entropy=round(float(hl or 0.0),6),
            roi=round(float(roi[i] if i < len(roi) else 0.0),6),
        )

    # społeczności (label propagation, ważone „edge”)
    comm = _label_propagation(N, adj, weight=edges_local, iters=10)
    for i in range(N):
        cells_payload[i]["community"] = int(comm[i])

    # globalne metryki
    roi_cov = sum(roi)/max(1, len(roi))
    meta: Dict[str, Any] = {
        "mode": core.mode,
        "size": {"W": W, "H": H},
        "rows": rows, "cols": cols, "cells": N,
        "roi_coverage": round(float(roi_cov), 6),
        "edges_quantiles": _quantiles([float(x) for x in edges_local]),
        "entropy_quantiles": _quantiles([float(x) for x in entr_local]),
        "ring_max": max(rings) if rings else 0,
    }

    # hash struktury (stabilny na zmianę układu wartości)
    hash_src = dict(mode=core.mode, size=core.size, rows=rows, cols=cols)
    h = hashlib.sha256(json.dumps(hash_src, separators=(",",":"), sort_keys=True).encode("utf-8")).hexdigest()

    return MosaicMeta(
        version="v1",
        meta={**meta, "graph_hash": h, "generated_by": "analysis.mosaic_meta"},
        cells=cells_payload,
        edges=adj,
    )

def save_mosaic_meta(m: MosaicMeta, repo_root: Optional[Path]=None) -> Path:
    glx = _ensure_glx_dir(repo_root)
    out = glx / "graphs" / "mosaic_meta.json"
    payload = dict(
        version=m.version,
        meta=m.meta,
        cells=m.cells,
        edges=m.edges,
    )
    _atomic_write_json(out, payload)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _read_json(p: Path) -> Mapping[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))

def cli_main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="analysis.mosaic_meta", description="Buduje warstwę meta mozaiki (struktura + cechy).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("from-delta", help="zbuduj z delta_report.json")
    a.add_argument("--delta", required=True, help="ścieżka do delta_report.json")
    a.add_argument("--repo-root", default=".", help="root repo do zapisu artefaktów (.glx/)")

    b = sub.add_parser("from-core", help="zbuduj ze zserializowanego core/mosaic JSON")
    b.add_argument("--core", required=True, help="ścieżka do JSON zawierającego sekcję core/mosaic/grid")
    b.add_argument("--repo-root", default=".", help="root repo do zapisu artefaktów (.glx/)")

    args = ap.parse_args(argv)
    root = Path(getattr(args, "repo_root", ".")).resolve()

    if args.cmd == "from-delta":
        d = _read_json(Path(args.delta))
        meta = build_mosaic_meta(d)
        out = save_mosaic_meta(meta, repo_root=root)
        print(str(out))
        return 0

    if args.cmd == "from-core":
        core = _read_json(Path(args.core))
        meta = build_mosaic_meta(core)
        out = save_mosaic_meta(meta, repo_root=root)
        print(str(out))
        return 0

    return 2


if __name__ == "__main__":
    import sys
    sys.exit(cli_main())
