# glitchlab/mosaic/hybrid_schema_builder.py
# Budowa HYBRYDOWEJ STRUKTURY (AST ⇄ Mozaika) z pliku YAML/JSON + meta-warstwy
# Cel: stworzyć „drzewo hybrydy struktur” i opisać procesy generacji/opt.
# Python 3.9+ (deps: numpy; opcjonalnie pyyaml do .yaml)

from __future__ import annotations
import json, os, math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Literal, Callable
import numpy as np

# Wykorzystujemy istniejący algorytm
from hybrid_ast_mosaic import (
    EXAMPLE_SRC, EDGE_THR_DEFAULT,
    AstSummary, Mosaic,
    build_mosaic, build_mosaic_grid, build_mosaic_hex, region_ids,
    ast_deltas, compress_ast,
    phi_region_for, phi_region_for_balanced, phi_region_for_entropy,
    phi_cost, psi_feedback,
    distance_ast_mosaic, sweep, run_once, sign_test_phi2_better
)

# ───────────────────────────────────────────────────────────────────────────────
# 1) DEFINICJA SCHEMATU (spec) — co może przyjść w YAML/JSON
# ───────────────────────────────────────────────────────────────────────────────

# Uwaga: projekt świadomie rozdziela „twarde” dane (AST, mozaika) od meta-warstw
# (metryki, topografia metaprzestrzeni, relacje Φ/Ψ, reguły optymalizacji).

@dataclass
class ASTInput:
    # Albo raw_source, albo preliczone statystyki:
    raw_source: Optional[str] = None
    # ew. makra wejściowe do generowania źródła (np. wstrzyknięcia)
    macros: Dict[str, str] = field(default_factory=dict)
    # Jeśli brak raw_source — można podać fallback:
    use_example_if_empty: bool = True


@dataclass
class MosaicInput:
    kind: Literal["grid", "hex"] = "grid"
    rows: int = 12
    cols: int = 12
    seed: int = 7
    edge_thr: float = EDGE_THR_DEFAULT
    # Opcjonalne „narzucenie” ROI lub mapy krawędzi (np. z pliku)
    roi_mask: Optional[List[int]] = None     # indeksy kafli ROI
    edge_override: Optional[List[float]] = None  # wartości edge[0..N-1]


@dataclass
class PhiPsiControl:
    # wybór selektora Φ
    phi: Literal["heur", "balanced", "entropy"] = "balanced"
    # Ψ-feedback:
    delta: float = 0.25
    # λ-kompresja AST:
    lmbd: float = 0.60


@dataclass
class MetaSpace:
    # Topografia metaprzestrzeni: wymiarowanie i priorytety (np. Align/CR)
    axes: List[str] = field(default_factory=lambda: ["Align", "J_phi", "CR_AST", "CR_TO"])
    weights: Dict[str, float] = field(default_factory=lambda: {"Align": 1.0, "J_phi": 1.0, "CR_AST": 0.5, "CR_TO": 0.25})
    # Warstwy meta-opisu (kolejność/przepływ analizy)
    layers: List[str] = field(default_factory=lambda: ["struct", "mosaic", "phi", "psi", "metrics"])
    # Relacje nadrzędne (np. „dokładność > szybkość”)
    policies: Dict[str, Any] = field(default_factory=lambda: {"preference": "accuracy"})


@dataclass
class OptimizationSpec:
    # Przestrzeń przemiatania
    lambdas: List[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75])
    deltas:  List[float] = field(default_factory=lambda: [0.0, 0.25, 0.5])
    edge_thrs: List[float] = field(default_factory=lambda: [0.45, 0.55, 0.60])
    seeds: int = 60
    # Funkcja celu w metaprzestrzeni: mniejsza lepsza (wagowana suma)
    # cost = wJ*J_phi2 + wA*(1-Align) + wC*(-log CR_AST) + wT*(-log(1+CR_TO))
    objective_weights: Dict[str, float] = field(default_factory=lambda: {
        "J_phi": 1.0, "Align": 1.0, "CR_AST": 0.5, "CR_TO": 0.25
    })


@dataclass
class HybridSchema:
    ast: ASTInput = field(default_factory=ASTInput)
    mosaic: MosaicInput = field(default_factory=MosaicInput)
    control: PhiPsiControl = field(default_factory=PhiPsiControl)
    metaspace: MetaSpace = field(default_factory=MetaSpace)
    optimize: OptimizationSpec = field(default_factory=OptimizationSpec)

# ───────────────────────────────────────────────────────────────────────────────
# 2) ŁADOWANIE I WALIDACJA SCHEMATU
# ───────────────────────────────────────────────────────────────────────────────

def load_schema(path_or_dict: Any) -> HybridSchema:
    if isinstance(path_or_dict, dict):
        data = path_or_dict
    else:
        path = str(path_or_dict)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        if path.lower().endswith((".yaml", ".yml")):
            try:
                import yaml  # opcjonalne
            except Exception as e:
                raise RuntimeError("Do obsługi YAML zainstaluj pyyaml") from e
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        else:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
    # spłaszczenie do dataclass
    def dc(cls, key):
        return cls(**data.get(key, {})) if key in data else cls()
    return HybridSchema(
        ast=dc(ASTInput, "ast"),
        mosaic=dc(MosaicInput, "mosaic"),
        control=dc(PhiPsiControl, "control"),
        metaspace=dc(MetaSpace, "metaspace"),
        optimize=dc(OptimizationSpec, "optimize"),
    )

# ───────────────────────────────────────────────────────────────────────────────
# 3) GENERACJA: AST, MOZAJKA, Φ/Ψ, METRYKI — „drzewo hybrydy”
# ───────────────────────────────────────────────────────────────────────────────

@dataclass
class HybridNode:
    """Węzeł w „drzewie hybrydy”: może reprezentować węzeł AST lub region mozaiki."""
    kind: Literal["ast", "tile", "region"] = "ast"
    ref: Optional[int] = None             # id węzła AST albo indeks kafla
    label: Optional[str] = None
    meta: Optional[List[float]] = None    # kopia meta-wektora, jeżeli dotyczy


@dataclass
class HybridGraph:
    """Wynik: pełna hybryda + artefakty wyjściowe i metryki."""
    schema: HybridSchema
    ast_raw: AstSummary
    ast_l: AstSummary
    mosaic: Mosaic
    phi_variant: str
    j_phi1: float
    j_phi2: float
    j_phi3: float
    align: float
    cr_ast: float
    cr_to: float
    alpha: float
    beta: float
    S: int
    H: int
    Z: int
    nodes: List[HybridNode] = field(default_factory=list)
    edges: List[Tuple[int, int, str]] = field(default_factory=list)  # (src, dst, relation)
    # ślad: co liczyliśmy
    trace: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        d = asdict(self)
        # ast_raw/ast_l/mosaic są złożone — serializujemy podstawowe pola
        for k in ("ast_raw", "ast_l"):
            ar = d[k]
            d[k] = {x: ar[x] for x in ("S", "H", "Z", "maxZ", "alpha", "beta")} if isinstance(ar, dict) else {}
        # mozaika — tylko parametry i agregaty
        if isinstance(d.get("mosaic"), dict):
            m = d["mosaic"]
            d["mosaic"] = {x: m.get(x) for x in ("rows", "cols", "kind")}
        return json.dumps(d, indent=2)


def _select_phi(name: str) -> Callable:
    return {
        "heur":     phi_region_for,
        "balanced": phi_region_for_balanced,
        "entropy":  phi_region_for_entropy,
    }.get(name, phi_region_for_balanced)


def _objective_score(res: Dict[str, float], w: Dict[str, float]) -> float:
    # mniejsze lepsze
    J = res["J_phi2"]
    A = res["Align"]
    CA = max(1e-6, res["CR_AST"])
    CT = 1.0 + max(0.0, res["CR_TO"])
    return (
        w.get("J_phi", 1.0) * J +
        w.get("Align", 1.0) * (1.0 - A) +
        w.get("CR_AST", 0.5) * (-math.log(CA)) +
        w.get("CR_TO", 0.25) * (-math.log(CT))
    )

def _apply_overrides(M: Mosaic, spec: MosaicInput) -> Mosaic:
    N = M.rows * M.cols
    if spec.edge_override:
        arr = np.array(spec.edge_override, dtype=float)
        if arr.size != N:
            raise ValueError("edge_override length mismatch")
        M.edge = arr
    if spec.roi_mask:
        mask = np.zeros(N, dtype=float)
        for i in spec.roi_mask:
            if 0 <= i < N: mask[i] = 1.0
        M.roi = mask
    return M

def build_hybrid_from_schema(schema: HybridSchema) -> HybridGraph:
    # 1) AST
    src = schema.ast.raw_source
    if not src:
        src = EXAMPLE_SRC if schema.ast.use_example_if_empty else "def main():\n  return 0\n"
    # proste makra (string replace)
    for k, v in (schema.ast.macros or {}).items():
        src = src.replace(f"${{{k}}}", str(v))

    ast_raw = ast_deltas(src)
    ast_l   = compress_ast(ast_raw, schema.control.lmbd)

    # 2) Mozaika
    M = build_mosaic(schema.mosaic.rows, schema.mosaic.cols,
                     seed=schema.mosaic.seed,
                     kind=schema.mosaic.kind,
                     edge_thr=schema.mosaic.edge_thr)
    M = _apply_overrides(M, schema.mosaic)

    # 3) Φ-koszty
    selector = _select_phi(schema.control.phi)
    J1, _ = phi_cost(ast_l, M, schema.mosaic.edge_thr, phi_region_for)
    J2, _ = phi_cost(ast_l, M, schema.mosaic.edge_thr, selector)
    J3, _ = phi_cost(ast_l, M, schema.mosaic.edge_thr, phi_region_for_entropy)

    # 4) Ψ-feedback + Align/kompresje
    ast_after = psi_feedback(ast_l, M, schema.control.delta, schema.mosaic.edge_thr)
    Align = 1.0 - min(1.0, distance_ast_mosaic(ast_after, M, schema.mosaic.edge_thr))

    p_edge = float(np.mean(M.edge > schema.mosaic.edge_thr))
    CR_AST = (ast_raw.S + ast_raw.H + max(1, ast_raw.Z)) / max(1, ast_l.S + ast_l.H + max(1, ast_l.Z))
    CR_TO  = (1.0 / max(1e-6, min(p_edge, 1 - p_edge))) - 1.0

    # 5) Hybrydowe drzewo (prosty wariant: wszystkie węzły AST + reprezentant ROI/edges)
    nodes: List[HybridNode] = []
    idx_map: Dict[Tuple[str,int], int] = {}

    # AST węzły
    for i, n in ast_l.nodes.items():
        idx_map[("ast", i)] = len(nodes)
        nodes.append(HybridNode(kind="ast", ref=i, label=n.label, meta=list(n.meta)))

    edges: List[Tuple[int,int,str]] = []
    for i, n in ast_l.nodes.items():
        for ch in n.children:
            edges.append((idx_map[("ast", i)], idx_map[("ast", ch)], "ast:child"))

    # reprezentatywne regiony mozaiki jako „region-nodes”
    edges_ids = region_ids(M, "edges", schema.mosaic.edge_thr)[: min(64, M.rows*M.cols)]
    roi_ids   = region_ids(M, "roi",   schema.mosaic.edge_thr)
    # „region” jako logiczny węzeł + kilka przykładów kafli (tile)
    if edges_ids:
        id_region_edges = len(nodes)
        nodes.append(HybridNode(kind="region", ref=None, label="edges"))
        for t in edges_ids[:12]:
            id_tile = len(nodes)
            nodes.append(HybridNode(kind="tile", ref=t, label="tile"))
            edges.append((id_region_edges, id_tile, "region:has_tile"))
    if roi_ids:
        id_region_roi = len(nodes)
        nodes.append(HybridNode(kind="region", ref=None, label="roi"))
        for t in roi_ids[:12]:
            id_tile = len(nodes)
            nodes.append(HybridNode(kind="tile", ref=t, label="tile"))
            edges.append((id_region_roi, id_tile, "region:has_tile"))

    # 6) Ślad i metryki w metaprzestrzeni
    trace = dict(
        J_phi=dict(phi1=J1, phi2=J2, phi3=J3),
        control=asdict(schema.control),
        metaspace=asdict(schema.metaspace),
        objective_preview=_objective_score(
            {"J_phi2": J2, "Align": Align, "CR_AST": CR_AST, "CR_TO": CR_TO},
            schema.optimize.objective_weights
        )
    )

    return HybridGraph(
        schema=schema,
        ast_raw=ast_raw, ast_l=ast_l, mosaic=M,
        phi_variant=schema.control.phi,
        j_phi1=J1, j_phi2=J2, j_phi3=J3,
        align=Align, cr_ast=CR_AST, cr_to=CR_TO,
        alpha=ast_l.alpha, beta=ast_l.beta,
        S=ast_l.S, H=ast_l.H, Z=ast_l.Z,
        nodes=nodes, edges=edges, trace=trace
    )

# ───────────────────────────────────────────────────────────────────────────────
# 4) MINI-BENCH (hooki optymalizacji) — do użycia przez docelowy benchmark
# ───────────────────────────────────────────────────────────────────────────────

def grid_search(schema: HybridSchema) -> Dict[str, Any]:
    """Prosty benchmark/opt: sweep po (λ×Δ×thr) z funkcją celu w metaprzestrzeni."""
    best = None
    best_tuple = None
    hist = []
    for lam in schema.optimize.lambdas:
        for de in schema.optimize.deltas:
            for thr in schema.optimize.edge_thrs:
                res = run_once(lam, de,
                               schema.mosaic.rows, schema.mosaic.cols,
                               thr, mosaic_kind=schema.mosaic.kind)
                score = _objective_score(res, schema.optimize.objective_weights)
                hrow = dict(lambda_=lam, delta_=de, edge_thr=thr, **res, score=score)
                hist.append(hrow)
                tup = (score, -res["Align"], res["J_phi2"])
                if (best is None) or (tup < best_tuple):
                    best, best_tuple = hrow, tup
    return dict(best=best, history=hist)

def sign_test(schema: HybridSchema, n: Optional[int] = None) -> Dict[str, Any]:
    n_runs = n or schema.optimize.seeds
    out = sign_test_phi2_better(n_runs,
                                schema.mosaic.rows, schema.mosaic.cols,
                                schema.mosaic.edge_thr,
                                lam=schema.control.lmbd,
                                mosaic_kind=schema.mosaic.kind)
    return out

# ───────────────────────────────────────────────────────────────────────────────
# 5) ŁADOWANIE/URUCHAMIANIE Z LOKALNEGO PLIKU
# ───────────────────────────────────────────────────────────────────────────────

def load_and_build(path_or_dict: Any) -> HybridGraph:
    schema = load_schema(path_or_dict)
    return build_hybrid_from_schema(schema)

def load_and_benchmark(path_or_dict: Any) -> Dict[str, Any]:
    schema = load_schema(path_or_dict)
    # budowa (artefakt do ewentualnej wizualizacji)
    model = build_hybrid_from_schema(schema)
    # szybkie metryki/opt
    gs = grid_search(schema)
    st = sign_test(schema, n=schema.optimize.seeds)
    return dict(
        setup=asdict(schema),
        hybrid_summary=dict(
            phi=schema.control.phi,
            J=dict(phi1=model.j_phi1, phi2=model.j_phi2, phi3=model.j_phi3),
            Align=model.align, CR_AST=model.cr_ast, CR_TO=model.cr_to,
            alpha=model.alpha, beta=model.beta, S=model.S, H=model.H, Z=model.Z,
            nodes=len(model.nodes), edges=len(model.edges)
        ),
        grid_search=gs,
        sign_test=st
    )

# ───────────────────────────────────────────────────────────────────────────────
# 6) CLI — szybki „generator hybrydy” i mini-benchmark
# ───────────────────────────────────────────────────────────────────────────────

def _cli():
    import argparse, sys
    ap = argparse.ArgumentParser(prog="hybrid-schema-builder",
        description="Budowa hybrydy AST⇄Mozaika z YAML/JSON + mini-benchmark")
    ap.add_argument("schema", help="Ścieżka do .yaml/.json lub '-' dla wbudowanego przykładu")
    ap.add_argument("--benchmark", action="store_true", help="uruchom grid-search + sign-test")
    args = ap.parse_args()

    if args.schema == "-":
        # przykład minimalny
        spec = {
            "ast": {"use_example_if_empty": True},
            "mosaic": {"kind": "hex", "rows": 12, "cols": 12, "edge_thr": 0.55},
            "control": {"phi": "balanced", "lmbd": 0.60, "delta": 0.25},
            "optimize": {"seeds": 40}
        }
    else:
        spec = args.schema

    if args.benchmark:
        out = load_and_benchmark(spec)
        print(json.dumps(out, indent=2))
    else:
        g = load_and_build(spec)
        print(g.to_json())

if __name__ == "__main__":
    _cli()
