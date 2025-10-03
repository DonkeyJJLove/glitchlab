# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
import json
import os
from typing import Dict, Any

# Backend bez wyświetlacza (ważne pod pytest/CI)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _safe_get(d: Dict[str, Any], key: str, default: float = 0.0) -> float:
    v = d.get(key, default)
    try:
        return float(v) if v is not None else default
    except Exception:
        return default


def _ratio(pass_at_1: float, total: float) -> float:
    if total and total > 0:
        return pass_at_1 / float(total)
    return 0.0


def plot_results(json_path: str, out_dir: str) -> None:
    """
    Czyta artefakt ab.json i zapisuje:
      - accuracy.png       (Pass@1 / Total)
      - timings.png        (Mean time)
      - align_vs_ast.png   (Mean align vs. mean_cr_ast)
      - comparison.txt     (krótki werdykt tekstowy)
    """
    # --- I/O
    with open(json_path, "r", encoding="utf-8") as f:
        data: Dict[str, Dict[str, Any]] = json.load(f)

    os.makedirs(out_dir, exist_ok=True)

    # --- Przygotowanie danych
    agents = ["A1", "A2", "B"]
    # Upewnijmy się, że brakujące sekcje nie wywalą importu
    for a in agents:
        data.setdefault(a, {})

    acc_vals = []
    t_vals = []
    align_vals = []
    cr_ast_vals = []

    for a in agents:
        d = data[a]
        acc = _ratio(_safe_get(d, "pass_at_1", 0.0), _safe_get(d, "total", 0.0))
        acc_vals.append(acc)
        t_vals.append(_safe_get(d, "mean_time", 0.0))
        align_vals.append(_safe_get(d, "mean_align", 0.0))
        cr_ast_vals.append(_safe_get(d, "mean_cr_ast", 0.0))

    # --- 1) accuracy.png
    plt.figure()
    plt.bar(agents, acc_vals)
    plt.ylim(0, 1)
    plt.ylabel("Accuracy (Pass@1 / Total)")
    plt.title("Accuracy by Agent")
    for i, v in enumerate(acc_vals):
        plt.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy.png"))
    plt.close()

    # --- 2) timings.png
    plt.figure()
    plt.bar(agents, t_vals)
    plt.ylabel("Mean time [s]")
    plt.title("Mean runtime by Agent")
    for i, v in enumerate(t_vals):
        plt.text(i, v, f"{v:.3f}s", ha="center", va="bottom", rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "timings.png"))
    plt.close()

    # --- 3) align_vs_ast.png (tworzymy ZAWSZE, nawet jeśli 0 lub brak danych)
    x = range(len(agents))
    width = 0.35

    plt.figure()
    plt.bar([i - width/2 for i in x], align_vals, width=width, label="mean_align")
    plt.bar([i + width/2 for i in x], cr_ast_vals, width=width, label="mean_cr_ast")
    plt.xticks(list(x), agents)
    plt.ylabel("Score")
    plt.title("Alignment vs. AST compression (mean)")
    plt.legend()
    # adnotacje
    for i, v in enumerate(align_vals):
        plt.text(i - width/2, v, f"{v:.3f}", ha="center", va="bottom", rotation=90)
    for i, v in enumerate(cr_ast_vals):
        plt.text(i + width/2, v, f"{v:.3f}", ha="center", va="bottom", rotation=90)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "align_vs_ast.png"))
    plt.close()

    # --- 4) comparison.txt (krótki, deterministyczny werdykt)
    # Reguła werdyktu (jak w README):
    # 1) max accuracy, 2) wyższy align + cr_ast bliżej 1, 3) krótszy czas
    def distance_to_one(v: float) -> float:
        return abs(1.0 - v)

    scores = []
    for i, a in enumerate(agents):
        acc = acc_vals[i]
        align = align_vals[i]
        crast = cr_ast_vals[i]
        t = t_vals[i]
        score = (
            acc * 1000.0                        # priorytet 1: mocno ważymy accuracy
            + align * 10.0                      # priorytet 2a
            - distance_to_one(crast) * 10.0     # priorytet 2b (bliżej 1 lepiej)
            - t                                  # priorytet 3 (krótszy czas)
        )
        scores.append((score, a))

    scores_sorted = sorted(scores, reverse=True)
    winner = scores_sorted[0][1]

    # zapis raportu
    lines = []
    lines.append("=== Comparison summary ===")
    lines.append(f"Agents: {', '.join(agents)}")
    lines.append("")
    lines.append("Accuracy (Pass@1/Total):")
    for a, v in zip(agents, acc_vals):
        lines.append(f"  {a}: {v:.3f}")
    lines.append("")
    lines.append("Mean time [s]:")
    for a, v in zip(agents, t_vals):
        lines.append(f"  {a}: {v:.3f}")
    lines.append("")
    lines.append("mean_align:")
    for a, v in zip(agents, align_vals):
        lines.append(f"  {a}: {v:.3f}")
    lines.append("")
    lines.append("mean_cr_ast:")
    for a, v in zip(agents, cr_ast_vals):
        lines.append(f"  {a}: {v:.3f}")
    lines.append("")
    lines.append(f"Winner (rule-based): {winner}")

    rep_path = os.path.join(out_dir, "comparison.txt")
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[INFO] Report written to {rep_path}")


def _make_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--json", required=True, help="Path to artifacts/ab.json")
    p.add_argument("--out", required=True, help="Output directory for plots")
    return p


def main() -> None:
    args = _make_argparser().parse_args()
    plot_results(json_path=args.json, out_dir=args.out)


if __name__ == "__main__":
    main()
