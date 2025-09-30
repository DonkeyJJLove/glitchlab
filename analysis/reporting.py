# glitchlab/analysis/reporting.py
# Generacja artefaktów raportowych:
#  - analysis/last/report.json
#  - analysis/last/summary.md
#  - analysis/last/mosaic_map.json
#
# Python 3.9+

from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# ──────────────────────────────────────────────────────────────────────────────
# Opcjonalne zależności na pipeline (żeby uniknąć cykli importów)
# ──────────────────────────────────────────────────────────────────────────────

try:
    # run_from_git(base, head, rows, cols, thr, mosaic_kind, delta, kappa_ab, paths, phi_name, policy_file) -> dict
    from glitchlab.mosaic.hybrid_ast_mosaic import run_from_git as _run_from_git
except Exception:  # pragma: no cover
    _run_from_git = None

DEFAULT_OUTDIR = Path("analysis/last")

__all__ = [
    "emit_artifacts",
    "emit_from_git",
    "emit_from_stdin",
    "build_summary_md",
    "build_mosaic_map",
    "save_json",
    "save_text",
    "_cli",
]

# ──────────────────────────────────────────────────────────────────────────────
# Helpers I/O
# ──────────────────────────────────────────────────────────────────────────────

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def save_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def save_text(path: Path, s: str) -> None:
    path.write_text(s, encoding="utf-8")

def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ──────────────────────────────────────────────────────────────────────────────
# Summary.md (format zgodny z Twoimi sekcjami)
# ──────────────────────────────────────────────────────────────────────────────

def _fmt_align(before: float, after: float) -> str:
    gain = after - before
    return f"{before:.3f} → {after:.3f}  (Δ={gain:+.3f})"

def _any_policy_cost(report: Dict[str, Any]) -> bool:
    for f in report.get("files", []):
        phi = f.get("phi", {}) or {}
        if "J_policy" in phi or phi.get("policy_used"):
            return True
    return False

def _files_table(report: Dict[str, Any]) -> str:
    has_policy = _any_policy_cost(report)

    head = ["Plik", "dS", "dH", "dZ", "Align (before→after)", "Jφ1", "Jφ2", "Jφ3"]
    if has_policy:
        head.append("JφP")

    rows = []
    rows.append("| " + " | ".join(head) + " |")
    rows.append("| " + " | ".join(["----", "--:", "--:", "--:", "---------------------", "---:", "---:", "---:"] + (["---:"] if has_policy else [])) + " |")

    for f in report.get("files", []):
        d = f.get("delta", {}) or {}
        a = f.get("align", {}) or {}
        phi = f.get("phi", {}) or {}
        base_cols = [
            f.get("path", ""),
            str(int(d.get("dS", 0))),
            str(int(d.get("dH", 0))),
            str(int(d.get("dZ", 0))),
            _fmt_align(float(a.get("before", 0.0)), float(a.get("after", 0.0))),
            f"{float(phi.get('J1', 0.0)):.4f}",
            f"{float(phi.get('J2', 0.0)):.4f}",
            f"{float(phi.get('J3', 0.0)):.4f}",
        ]
        if has_policy:
            jp = phi.get("J_policy", None)
            base_cols.append("-" if jp is None else f"{float(jp):.4f}")
        rows.append("| " + " | ".join(base_cols) + " |")
    return "\n".join(rows)

def _totals_block(report: Dict[str, Any]) -> str:
    t = report.get("totals", {}) or {}
    dS = int(t.get("dS", 0))
    dH = int(t.get("dH", 0))
    dZ = int(t.get("dZ", 0))
    ag = float(t.get("align_gain", 0.0))
    return f"- **ΔS**: {dS}, **ΔH**: {dH}, **ΔZ**: {dZ}, **Σ Align gain**: {ag:+.3f}"

def build_summary_md(report: Dict[str, Any]) -> str:
    """
    Buduje summary.md w formacie:
      [Δ] Zakres, [Φ/Ψ], [AST], [Kontekst], [Dokumentacja], [Testy/Ryzyko], Meta
    """
    base = report.get("base", "")
    head = report.get("head", "")
    branch = report.get("branch") or ""
    mz = report.get("mosaic", {}) or {}
    grid = mz.get("grid", {}) or {}
    thr = mz.get("thr", None)
    phi_selector = report.get("phi_selector", None)
    policy_file = report.get("policy_file", None)

    headl = [
        f"# GLX — ΔAST/Mozaika (Φ/Ψ) — {datetime.now().strftime('%Y-%m-%d')}",
        "",
        f"- **BASE**: `{base}`",
        f"- **HEAD**: `{head}`",
        f"- **Branch**: `{branch}`" if branch else "- **Branch**: (brak danych)",
        f"- **Mozaika**: kind={mz.get('kind','grid')}, grid={grid.get('rows','?')}×{grid.get('cols','?')}, thr={thr}",
    ]
    if phi_selector:
        headl.append(f"- **Φ selector**: {phi_selector}")
    if policy_file:
        headl.append(f"- **Policy file**: `{policy_file}`")
    headl.extend([
        f"- **Plików zmienionych**: {len(report.get('files', []))}",
        "",
    ])

    section_delta = [
        "## [Δ] Zakres",
        _totals_block(report),
        "",
        "### Zmiany per plik",
        _files_table(report),
        "",
    ]

    # Sekcja Φ/Ψ — skrótowo: preferuj Jφ2 vs Jφ1, pokaż gdzie zysk
    phi_lines = [
        "## [Φ/Ψ] Projekcja i sprzężenie",
        "- **Φ**: porównujemy koszty Jφ1/Jφ2/Jφ3 (niżej lepiej).",
        "- **Ψ**: feedback na metę + sprzężenie (α/β) → wzrost Align.",
        "",
    ]
    has_policy = _any_policy_cost(report)
    for f in report.get("files", []):
        phi = f.get("phi", {}) or {}
        a = f.get("align", {}) or {}
        gain = float(a.get("after", 0.0)) - float(a.get("before", 0.0))
        line = (
            f"- `{f.get('path','')}`:  "
            f"Jφ1={phi.get('J1',0):.4f}, Jφ2={phi.get('J2',0):.4f}, Jφ3={phi.get('J3',0):.4f}"
        )
        if has_policy:
            jp = phi.get("J_policy", None)
            if jp is not None:
                line += f", JφP={float(jp):.4f}"
        line += f";  Align Δ={gain:+.3f}"
        phi_lines.append(line)
    phi_lines.append("")

    # Sekcja AST — skrót: ΔS/ΔH/ΔZ na plikach
    ast_lines = [
        "## [AST] Skrót metryk",
        "Interpretacja: **S** – masa struktury, **H** – aktywność/heterogeniczność, **Z** – liczba komponentów.",
        "",
    ]
    for f in report.get("files", []):
        d = f.get("delta", {}) or {}
        ast_lines.append(
            f"- `{f.get('path','')}`: ΔS={int(d.get('dS',0))}, ΔH={int(d.get('dH',0))}, ΔZ={int(d.get('dZ',0))}"
        )
    ast_lines.append("")

    # Sekcja Kontekst – placeholder
    ctx_lines = [
        "## [Kontekst]",
        "- Zmiany analizowane względem BASE..HEAD; wpływ lokalny (per-plik).",
        "- Dalsza analiza kontekstowa zależy od domeny projektu (do uzupełnienia przez autora commita).",
        "",
    ]

    # Sekcja Dok – placeholder
    doc_lines = [
        "## [Dokumentacja]",
        "- Jeśli wprowadzono API/param zmiany, zaktualizuj docstringi/README.",
        "- Dodaj link do issue/ADR, jeśli dotyczy.",
        "",
    ]

    # Sekcja Testy/Ryzyko – placeholder z checklistą
    test_lines = [
        "## [Testy / Ryzyko]",
        "- [ ] Testy jednostkowe dla zmienionych funkcji/metod",
        "- [ ] Smoke dla głównych ścieżek",
        "- [ ] Weryfikacja wpływu na I/O (jeśli dotyczy)",
        "",
    ]

    # Meta
    meta_lines = [
        "## Meta",
        f"- Wygenerowano: {_ts()}",
        "- Generator: hybrid AST⇄Mozaika (Φ/Ψ) — reporting",
        "",
    ]

    parts = headl + section_delta + phi_lines + ast_lines + ctx_lines + doc_lines + test_lines + meta_lines
    return "\n".join(parts)

# ──────────────────────────────────────────────────────────────────────────────
# Mosaic map (lekka, zgrubna) — bez ciężkich zależności
# ──────────────────────────────────────────────────────────────────────────────

def build_mosaic_map(report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tworzy lekki opis mozaiki na potrzeby wizualizacji:
    - grid (rows/cols), thr, kind,
    - phi_selector/policy_file (jeśli występują),
    - per-file: Jφ*, JφP (opcjonalnie), align Δ (before/after).
    """
    mz = report.get("mosaic", {}) or {}
    grid = mz.get("grid", {}) or {}
    out = {
        "grid": {
            "rows": int(grid.get("rows", 0) or 0),
            "cols": int(grid.get("cols", 0) or 0),
        },
        "thr": float(mz.get("thr", 0.0) or 0.0),
        "kind": mz.get("kind", "grid"),
        "phi_selector": report.get("phi_selector", None),
        "policy_file": report.get("policy_file", None),
        "files": [],
    }
    for f in report.get("files", []):
        a = f.get("align", {}) or {}
        phi = f.get("phi", {}) or {}
        entry = {
            "path": f.get("path", ""),
            "align": {
                "before": float(a.get("before", 0.0)),
                "after": float(a.get("after", 0.0)),
                "gain": float(a.get("after", 0.0)) - float(a.get("before", 0.0)),
            },
            "costs": {
                "J_phi1": float(phi.get("J1", 0.0)),
                "J_phi2": float(phi.get("J2", 0.0)),
                "J_phi3": float(phi.get("J3", 0.0)),
            },
        }
        if "J_policy" in phi:
            # tylko jeśli pipeline zwrócił koszt polityki
            entry["costs"]["J_phiP"] = float(phi.get("J_policy"))
        out["files"].append(entry)
    return out

# ──────────────────────────────────────────────────────────────────────────────
# Emisja artefaktów
# ──────────────────────────────────────────────────────────────────────────────

def emit_artifacts(report: Dict[str, Any], outdir: Path = DEFAULT_OUTDIR) -> Dict[str, str]:
    """
    Zapisuje trzy artefakty do outdir.
    Zwraca mapę ścieżek: {"report": "...", "summary": "...", "mosaic_map": "..."}.
    """
    _ensure_dir(outdir)
    paths = {
        "report": outdir / "report.json",
        "summary": outdir / "summary.md",
        "mosaic_map": outdir / "mosaic_map.json",
    }

    save_json(paths["report"], report)
    save_text(paths["summary"], build_summary_md(report))
    save_json(paths["mosaic_map"], build_mosaic_map(report))

    return {k: str(v) for k, v in paths.items()}

# ──────────────────────────────────────────────────────────────────────────────
# High-level: FROM-GIT / FROM-STDIN
# ──────────────────────────────────────────────────────────────────────────────

def emit_from_git(
    base: str,
    head: str = "HEAD",
    *,
    rows: int = 6,
    cols: int = 6,
    thr: float = 0.55,
    mosaic_kind: str = "grid",
    delta: float = 0.25,
    kappa_ab: float = 0.35,
    paths: Optional[List[str]] = None,
    outdir: Path = DEFAULT_OUTDIR,
    # NOWE: wybór Φ i plik polityk (przekazywane do run_from_git)
    phi: str = "balanced",
    policy_file: Optional[str] = None,
) -> Dict[str, str]:
    """
    Odpala pipeline from-git i zapisuje artefakty do outdir.
    Wymaga dostępności glitchlab.mosaic.hybrid_ast_mosaic.run_from_git.
    """
    if _run_from_git is None:
        raise RuntimeError("hybrid_ast_mosaic.run_from_git niedostępny — sprawdź instalację modułów.")

    report = _run_from_git(
        base=base, head=head,
        rows=rows, cols=cols, thr=thr,
        mosaic_kind=mosaic_kind,
        delta=delta, kappa_ab=kappa_ab,
        paths=paths or None,
        phi_name=phi,
        policy_file=policy_file,
    )
    return emit_artifacts(report, outdir=outdir)

def emit_from_stdin(outdir: Path = DEFAULT_OUTDIR) -> Dict[str, str]:
    """
    Czyta JSON report z stdin i generuje artefakty.
    Użyteczne, gdy 'from-git' wypisywany jest na stdout i pipowany do tego modułu.
    """
    data = sys.stdin.read()
    try:
        report = json.loads(data)
    except Exception as e:
        raise RuntimeError(f"Niepoprawny JSON na stdin: {e}")
    return emit_artifacts(report, outdir=outdir)

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _cli(argv: Optional[List[str]] = None) -> None:
    import argparse
    p = argparse.ArgumentParser(
        prog="glitchlab.reporting",
        description="Generacja artefaktów raportowych (summary.md, report.json, mosaic_map.json)"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("from-git", help="Odpala analizę BASE..HEAD i zapisuje artefakty")
    g.add_argument("--base", required=True)
    g.add_argument("--head", default="HEAD")
    g.add_argument("--rows", type=int, default=6)
    g.add_argument("--cols", type=int, default=6)
    g.add_argument("--thr", type=float, default=0.55)
    g.add_argument("--mosaic", choices=["grid", "hex"], default="grid")
    g.add_argument("--delta", type=float, default=0.25)
    g.add_argument("--kappa", type=float, default=0.35)
    g.add_argument("--paths", nargs="*")
    g.add_argument("--out", default=str(DEFAULT_OUTDIR))
    # NOWE: wybór Φ i plik polityk
    g.add_argument("--phi", choices=["basic", "balanced", "entropy", "policy"], default="balanced")
    g.add_argument("--policy-file", default=None)
    g.set_defaults(cmd="from-git")

    s = sub.add_parser("from-stdin", help="Czyta JSON raport z stdin i zapisuje artefakty")
    s.add_argument("--out", default=str(DEFAULT_OUTDIR))
    s.set_defaults(cmd="from-stdin")

    args = p.parse_args(argv)

    if args.cmd == "from-git":
        out_map = emit_from_git(
            base=args.base,
            head=args.head,
            rows=args.rows,
            cols=args.cols,
            thr=args.thr,
            mosaic_kind=args.mosaic,
            delta=args.delta,
            kappa_ab=args.kappa,
            paths=args.paths or None,
            outdir=Path(args.out),
            phi=args.phi,
            policy_file=args.policy_file,
        )
        print(json.dumps({"ok": True, "artifacts": out_map}, ensure_ascii=False, indent=2))
        return

    if args.cmd == "from-stdin":
        out_map = emit_from_stdin(outdir=Path(args.out))
        print(json.dumps({"ok": True, "artifacts": out_map}, ensure_ascii=False, indent=2))
        return


if __name__ == "__main__":
    _cli()
