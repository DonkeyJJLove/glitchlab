# -*- coding: utf-8 -*-
"""
tests/hybrid_mosaic_test_report.py
Runner + wizualizacja wyników testów dla hybrydowego algorytmu AST⇄Mozaika.

Funkcje:
- uruchomienie PyTest i zebranie podsumowania,
- zrzut metryk demo (baseline, sweep, test znaku),
- render Markdown z tabelami i minigrafami ASCII,
- opcjonalny zapis do pliku (--out report.md) i/lub JSON (--json-out results.json)

Uruchom przykładowo:
  python tests/hybrid_mosaic_test_report.py --lam 0.60 --delta 0.25 --rows 12 --cols 12 --kind hex --seeds 60 --edge-thr 0.55 --out report.md --json-out results.json

Wymaga:
  - glitchlab.gui.mosaic.hybrid_ast_mosaic (moduł algorytmu)
  - pytest (do uruchomienia testów)
  - numpy (w algorytmie)
"""

from __future__ import annotations
import argparse
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from typing import Dict, List

import numpy as np

# Import modułu algorytmu
import glitchlab.gui.mosaic.hybrid_ast_mosaic as hma


# ─────────────────────────────────────────────────────────────────────────────
# Pomocnicze: ascii-wykresy i tabele
# ─────────────────────────────────────────────────────────────────────────────

def _ascii_bar(vals: List[float], width: int = 32, fill: str = "█") -> str:
    """Prosty pasek (0..1) w ASCII dla listy wartości — rysuje średnią."""
    if not vals:
        return ""
    v = max(0.0, min(1.0, float(np.mean(vals))))
    n = int(round(v * width))
    return fill * n + " " * (width - n) + f"  ({v:.3f})"

def _ascii_sparkline(vals: List[float], height: int = 6, width: int = 40) -> str:
    """Mini-wykres linii w ASCII (skalowanie 0..1)."""
    if not vals:
        return ""
    lo, hi = min(vals), max(vals)
    rng = max(1e-12, hi - lo)
    norm = [(v - lo) / rng for v in vals]
    cols = min(width, len(norm))
    # równomierne próbkowanie do 'cols'
    idxs = np.linspace(0, len(norm)-1, cols).astype(int)
    grid = [[" "] * cols for _ in range(height)]
    for ci, i in enumerate(idxs):
        y = height - 1 - int(round(norm[i] * (height - 1)))
        grid[y][ci] = "•"
    return "\n".join("".join(row) for row in grid)

def _table(headers: List[str], rows: List[List[str]]) -> str:
    widths = [max(len(h), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    def line(cols): return " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(cols))
    sep = "-+-".join("-" * w for w in widths)
    out = [line(headers), sep]
    out.extend(line(r) for r in rows)
    return "\n".join(out)


# ─────────────────────────────────────────────────────────────────────────────
# Uruchamianie pytest i parsowanie wyników
# ─────────────────────────────────────────────────────────────────────────────

def run_pytests(pytest_path: str = "pytest", test_file: str = "tests/test_hybrid_mosaic_algo.py") -> Dict:
    """
    Uruchamia pytest w subprocessie. Zwraca słownik z podsumowaniem:
    {"ok": bool, "returncode": int, "stdout": "...", "summary": {"passed": int, "failed": int, "skipped": int, "xpassed": int}}
    """
    cmd = [pytest_path, "-q", test_file]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = proc.stdout + proc.stderr

    # prymitywny parser podsumowania z ostatniej linii pytest
    # przykłady: "8 passed in 0.25s" / "7 passed, 1 skipped in 0.21s" / "6 passed, 1 failed in 0.30s"
    summary = {"passed": 0, "failed": 0, "skipped": 0, "xpassed": 0, "xfailed": 0, "errors": 0}
    last_lines = [ln.strip() for ln in out.strip().splitlines()[-5:]]
    for ln in last_lines:
        if "in " in ln and any(k in ln for k in ["passed", "failed", "skipped", "xpassed", "xfailed", "error"]):
            # Rozbij po przecinkach, policz tokeny
            parts = [p.strip() for p in ln.split(" in ")[0].split(",")]
            for p in parts:
                toks = p.split()
                if len(toks) >= 2 and toks[0].isdigit():
                    n = int(toks[0]); tag = toks[1].lower()
                    if tag.startswith("passed"):  summary["passed"] = n
                    elif tag.startswith("failed"): summary["failed"] = n
                    elif tag.startswith("skipped"):summary["skipped"] = n
                    elif tag.startswith("xpassed"):summary["xpassed"] = n
                    elif tag.startswith("xfailed"):summary["xfailed"] = n
                    elif tag.startswith("error"): summary["errors"] = n
            break

    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": out,
        "summary": summary,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Zbieranie metryk demo (baseline, sweep, sign-test)
# ─────────────────────────────────────────────────────────────────────────────

def collect_demo(rows: int, cols: int, kind: str, lam: float, delta: float, edge_thr: float, seeds: int) -> Dict:
    # baseline
    base = hma.run_once(lam, delta, rows, cols, edge_thr, mosaic_kind=kind)

    # sweep
    sw = hma.sweep(rows, cols, edge_thr, mosaic_kind=kind)
    align_vals = [r["Align"] for r in sw]
    j2_vals = [r["J_phi2"] for r in sw]
    cr_vals = [r["CR_AST"] for r in sw]

    # sign-test
    sign = hma.sign_test_phi2_better(n_runs=seeds, rows=rows, cols=cols, thr=edge_thr,
                                     lam=lam, mosaic_kind=kind)

    return dict(
        baseline=base,
        sweep=sw,
        sign=sign,
        aggregates=dict(
            align_mean=float(np.mean(align_vals)),
            align_median=float(np.median(align_vals)),
            jphi2_mean=float(np.mean(j2_vals)),
            cr_ast_mean=float(np.mean(cr_vals)),
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Render Markdown
# ─────────────────────────────────────────────────────────────────────────────

def render_markdown(cfg: Dict, pytest_res: Dict, demo: Dict) -> str:
    dt = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")
    base = demo["baseline"]
    sign = demo["sign"]
    sw = demo["sweep"]
    agg = demo["aggregates"]

    # Tabela sweep (skrócona)
    hdr = ["λ", "Δ", "Align", "J_phi2", "CR_AST", "CR_TO", "α", "β", "S", "H", "Z"]
    rows = []
    for r in sw:
        rows.append([
            f"{r['lambda_']:.2f}", f"{r['delta_']:.2f}", f"{r['Align']:.3f}",
            f"{r['J_phi2']:.4f}", f"{r['CR_AST']:.3f}", f"{r['CR_TO']:.3f}",
            f"{r['alpha']:.2f}", f"{r['beta']:.2f}", str(int(r['S'])), str(int(r['H'])), str(int(r['Z'])),
        ])

    # Mini wykresy
    align_vals = [r["Align"] for r in sw]
    j2_vals = [r["J_phi2"] for r in sw]
    cr_vals = [r["CR_AST"] for r in sw]

    md = []
    md.append(f"# Hybrid AST⇄Mosaic — Test Report\n")
    md.append(f"_generated: {dt}_\n")
    md.append("## Setup\n")
    md.append("```json\n" + json.dumps(cfg, indent=2) + "\n```\n")

    md.append("## PyTest Summary\n")
    md.append("```text\n" + pytest_res["stdout"].strip() + "\n```\n")
    md.append("**Parsed summary:** " + json.dumps(pytest_res["summary"]) + "\n")

    md.append("## Baseline\n")
    md.append("```json\n" + json.dumps(base, indent=2) + "\n```\n")

    md.append("## Sweep λ×Δ (skrócone)\n")
    md.append("```\n" + _table(hdr, rows) + "\n```\n")

    md.append("### Mini-wykresy (ASCII)\n")
    md.append("**Align (trend):**\n\n```\n" + _ascii_sparkline(align_vals) + "\n```\n")
    md.append("**J_phi2 (średnia — mniejsze lepsze):**\n\n```\n" + _ascii_bar(j2_vals) + "\n```\n")
    md.append("**CR_AST (średnia — większa kompresja lepsza):**\n\n```\n" + _ascii_bar(cr_vals) + "\n```\n")

    md.append("## Sign test: Φ2 vs Φ1\n")
    md.append("```json\n" + json.dumps(sign, indent=2) + "\n```\n")

    md.append("## Wnioski (skrót)\n")
    md.append(f"- Średni Align (sweep): **{agg['align_mean']:.3f}**; mediana **{agg['align_median']:.3f}**.\n")
    md.append(f"- Średni J_phi2 (sweep): **{agg['jphi2_mean']:.3f}** (niżej lepiej).\n")
    md.append(f"- Średni CR_AST (sweep): **{agg['cr_ast_mean']:.3f}** (wyżej = większa kompresja).\n")
    md.append(f"- Sign test (Φ2 lepsze od Φ1): **wins={sign['wins']}**, **losses={sign['losses']}**, **ties={sign['ties']}**, p≈**{sign['p_sign']:.3g}**.\n")

    md.append("\n---\n")
    md.append("_End of report._\n")
    return "\n".join(md)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Runner + wizualizacja testów dla Hybrid AST⇄Mosaic")
    ap.add_argument("--rows", type=int, default=12)
    ap.add_argument("--cols", type=int, default=12)
    ap.add_argument("--kind", type=str, choices=["grid", "hex"], default="grid")
    ap.add_argument("--lam", type=float, default=0.60)
    ap.add_argument("--delta", type=float, default=0.25)
    ap.add_argument("--edge-thr", type=float, default=hma.EDGE_THR_DEFAULT)
    ap.add_argument("--seeds", type=int, default=60)
    ap.add_argument("--pytest-path", type=str, default="pytest")
    ap.add_argument("--test-file", type=str, default="test_hybrid_mosaic_algo.py")
    ap.add_argument("--out", type=str, default="raport.md")        # ścieżka do pliku .md (opcjonalnie)
    ap.add_argument("--json-out", type=str, default="")   # ścieżka do pliku .json (opcjonalnie)
    args = ap.parse_args()

    cfg = dict(rows=args.rows, cols=args.cols, kind=args.kind, lam=args.lam, delta=args.delta,
               edge_thr=args.edge_thr, seeds=args.seeds, test_file=args.test_file)

    print("\n=== Hybrid AST⇄Mosaic — Test Runner ===\n")

    # 1) PyTest
    print("[1/3] Running PyTest…")
    pytest_res = run_pytests(pytest_path=args.pytest_path, test_file=args.test_file)
    print(f"    -> returncode={pytest_res['returncode']} summary={pytest_res['summary']}")

    # 2) Demo metrics
    print("[2/3] Collecting demo metrics…")
    demo = collect_demo(rows=args.rows, cols=args.cols, kind=args.kind,
                        lam=args.lam, delta=args.delta, edge_thr=args.edge_thr,
                        seeds=args.seeds)
    print("    -> baseline:", {k: round(v, 4) if isinstance(v, float) else v for k, v in demo["baseline"].items()})

    # 3) Render report (Markdown + opcjonalny zapis)
    print("[3/3] Rendering report…")
    md = render_markdown(cfg, pytest_res, demo)
    print("\n" + md)  # pokaż w konsoli

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"\n[Saved] Markdown report -> {args.out}")

    if args.json_out:
        bundle = dict(config=cfg, pytest=pytest_res, demo=demo)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(bundle, f, indent=2)
        print(f"[Saved] JSON results -> {args.json_out}")


if __name__ == "__main__":
    main()
