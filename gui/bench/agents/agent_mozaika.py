# -*- coding: utf-8 -*-
from __future__ import annotations
import time, types
from typing import Dict, Any, List
import numpy as np

# Nasz protokół
from glitchlab.gui.mosaic import hybrid_ast_mosaic as hma

# Proste „szablony” – agent generuje minimalny kod funkcji z docstringiem.
# (To miejsce, gdzie normalnie wpinamy LLM. Tu trzymamy deterministykę.)
_TEMPLATES = {
    "reverse_str": "def reverse_str(s: str) -> str:\n    return s[::-1]\n",
    "fib": "def fib(n: int) -> int:\n    a,b=0,1\n    for _ in range(n): a,b=b,a+b\n    return a\n",
    "sum_csv_numbers": "def sum_csv_numbers(csv_text: str) -> int:\n    parts=[p.strip() for p in csv_text.split(',') if p.strip()]\n    return sum(int(p) for p in parts)\n",
    "moving_avg": "def moving_avg(x: list, k: int) -> list:\n    if k<=0 or len(x)<k: return []\n    out=[]\n    for i in range(len(x)-k+1):\n        out.append(sum(x[i:i+k])/k)\n    return out\n",
    "count_calls": "import ast\n\ndef count_calls(src: str) -> int:\n    t=ast.parse(src)\n    return sum(isinstance(n, ast.Call) for n in ast.walk(t))\n",
}


def _align_metrics(lmbd: float, delta: float, rows: int, cols: int, thr: float, kappa_ab: float = hma.KAPPA_AB_DEFAULT):
    res = hma.run_once(lmbd, delta, rows, cols, thr, mosaic_kind="grid", kappa_ab=kappa_ab)
    return dict(Align=res["Align"], J_phi2=res["J_phi2"], CR_TO=res["CR_TO"], CR_AST=res["CR_AST"])


def generate_code(task: Dict[str, Any], mode: str = "A1", rows: int = 12, cols: int = 12, thr: float = 0.55) -> Dict[
    str, Any]:
    """
    Zwraca: dict(code:str, metrics:dict, time_s:float)
    mode: "A1" (λ=0.60, Δ=0.25) | "A2" (λ=0.00, Δ=0.50)
    """
    t0 = time.time()
    if mode == "A2":
        lmbd, delta = 0.0, 0.5
    else:
        lmbd, delta = 0.60, 0.25

    m = _align_metrics(lmbd, delta, rows, cols, thr)
    name = task["entrypoint"]
    code = _TEMPLATES.get(name, f"def {name}(*args, **kwargs):\n    raise NotImplementedError\n")
    return dict(code=code, metrics=m, time_s=time.time() - t0)
