# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
import json
import glob
import os
import sys
import time

from tqdm import tqdm

from glitchlab.gui.bench import judge, stats
from glitchlab.gui.bench.agents import agent_mozaika, agent_ms_like


def load_tasks(pattern: str) -> dict:
    files = glob.glob(pattern)
    if not files:
        print(f"[ERROR] No task files match: {pattern}", file=sys.stderr)
        sys.exit(1)

    tasks = {}
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            tdef = json.load(f)
            tasks[os.path.basename(path)] = tdef
    return tasks


def run_agent(agent_name: str, tasks: dict) -> dict:
    results = {}
    total = len(tasks)

    for tname, tdef in tqdm(tasks.items(), total=total, desc=f"Running {agent_name}"):
        try:
            if agent_name == "A1":
                gen = agent_mozaika.generate_code(tdef, mode="A1")
            elif agent_name == "A2":
                gen = agent_mozaika.generate_code(tdef, mode="A2")
            elif agent_name == "B":
                gen = agent_ms_like.generate_code(tdef, mode="B")
            else:
                raise ValueError(f"Unknown agent {agent_name}")

            code = gen["code"]
            metrics = gen.get("metrics", {})
            time_s = gen.get("time_s", 0.0)

            tests = judge.run_tests(code, tdef)

            results[tname] = {
                **tests,
                **metrics,
                "time_s": time_s,
            }
        except Exception as e:
            results[tname] = {"error": str(e)}

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True, help="Pattern to JSON task files")
    parser.add_argument("--out", required=True, help="Output JSON file")
    args = parser.parse_args()

    print(time.strftime("%H:%M:%S"), "[INFO] Loading tasks...")
    tasks = load_tasks(args.tasks)
    print(time.strftime("%H:%M:%S"), f"[INFO] Loaded {len(tasks)} tasks")

    print(time.strftime("%H:%M:%S"), "[INFO] Running agents...")

    A1 = run_agent("A1", tasks)
    A2 = run_agent("A2", tasks)
    B = run_agent("B", tasks)

    summary = stats.summarize(A1, A2, B)

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(time.strftime("%H:%M:%S"), f"[INFO] Report saved to {args.out}")


if __name__ == "__main__":
    main()
