# -*- coding: utf-8 -*-
from __future__ import annotations
import json, argparse, glob
from typing import Dict, Any, List

from agents import agent_mozaika as A
from agents import agent_ms_like as B
from judge import run_tests
from backup.analysis.metrics import summarize_pairs


def load_task(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_agent_on_tasks(agent, task_paths: List[str], mode: str, rows=12, cols=12, thr=0.55):
    results = []
    for p in task_paths:
        t = load_task(p)
        gen = agent.generate_code(t, mode=mode, rows=rows, cols=cols, thr=thr)
        tests = run_tests(gen["code"], t)
        results.append(dict(
            id=t["id"], entry=t["entrypoint"],
            pass_at_1=tests["pass_at_1"],
            pass_cnt=tests["pass_cnt"], total=tests["total"],
            time_s=gen["time_s"],
            align=gen["metrics"].get("Align", None) if "metrics" in gen else None,
            j_phi2=gen["metrics"].get("J_phi2", None) if "metrics" in gen else None,
            cr_to=gen["metrics"].get("CR_TO", None) if "metrics" in gen else None,
            cr_ast=gen["metrics"].get("CR_AST", None) if "metrics" in gen else None,
            errors=tests["errors"]
        ))
    return results


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    acc = dict(
        pass_at_1=sum(r["pass_at_1"] for r in rows),
        total=len(rows),
        mean_time=sum(r["time_s"] for r in rows) / max(1, len(rows)),
    )
    # uśrednij metryki mapowania (tam gdzie są)
    for k in ("align", "j_phi2", "cr_to", "cr_ast"):
        vals = [r[k] for r in rows if r[k] is not None]
        acc["mean_" + k] = sum(vals) / len(vals) if vals else None
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="bench/tasks/*.json")
    ap.add_argument("--rows", type=int, default=12)
    ap.add_argument("--cols", type=int, default=12)
    ap.add_argument("--edge-thr", type=float, default=0.55)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    task_paths = sorted(glob.glob(args.tasks))
    assert task_paths, "No tasks found."

    A1 = run_agent_on_tasks(A, task_paths, mode="A1", rows=args.rows, cols=args.cols, thr=args.edge_thr)
    A2 = run_agent_on_tasks(A, task_paths, mode="A2", rows=args.rows, cols=args.cols, thr=args.edge_thr)
    Bx = run_agent_on_tasks(B, task_paths, mode="B", rows=args.rows, cols=args.cols, thr=args.edge_thr)

    # Porównania parowe pass@1
    sum_A2_A1 = summarize_pairs([r["pass_at_1"] for r in A2], [r["pass_at_1"] for r in A1])
    sum_A2_B = summarize_pairs([r["pass_at_1"] for r in A2], [r["pass_at_1"] for r in Bx])

    agg = dict(
        A1=aggregate(A1),
        A2=aggregate(A2),
        B=aggregate(Bx),
        A2_vs_A1=sum_A2_A1,
        A2_vs_B=sum_A2_B
    )

    print("\n=== A/B PILOT — Hybrid AST⇄Mosaic vs Baselines ===\n")
    print("[Summary accuracy]")
    for k in ("A1", "A2", "B"):
        s = agg[k]
        print(f" {k}: pass@1={s['pass_at_1']}/{s['total']}  mean_time={s['mean_time']:.3f}s"
              + (f"  Align={s['mean_align']:.3f}  J_phi2={s['mean_j_phi2']:.2f}" if s[
                                                                                        'mean_align'] is not None else ""))

    print("\n[A2 vs A1] pass@1 diffs (sign test):")
    s = agg["A2_vs_A1"]
    print(f" wins={s['wins']} losses={s['losses']} ties={s['ties']} p≈{s['p_sign']:.3g}  "
          f"meanΔ={s['mean_diff']:.3f}  medianΔ={s['median_diff']:.3f}  Cliffδ={s['cliffs_delta']:.3f}")

    print("\n[A2 vs B ] pass@1 diffs (sign test):")
    s = agg["A2_vs_B"]
    print(f" wins={s['wins']} losses={s['losses']} ties={s['ties']} p≈{s['p_sign']:.3g}  "
          f"meanΔ={s['mean_diff']:.3f}  medianΔ={s['median_diff']:.3f}  Cliffδ={s['cliffs_delta']:.3f}")

    if args.json:
        print("\n[JSON]")
        print(json.dumps(dict(per_task=dict(A1=A1, A2=A2, B=Bx), summary=agg), indent=2))


if __name__ == "__main__":
    main()
