# -*- coding: utf-8 -*-
from __future__ import annotations
import math, time, json
from typing import List, Dict, Any
import numpy as np


def sign_test_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0: return 1.0
    from math import comb
    k = max(wins, losses)
    return sum(comb(n, t) for t in range(k, n + 1)) / (2 ** n)


def cliffs_delta(diffs: List[float]) -> float:
    x = np.asarray(diffs, float)
    gt = np.count_nonzero(x > 0);
    lt = np.count_nonzero(x < 0)
    n = max(1, x.size)
    return float((gt - lt) / n)


def bootstrap_ci(data: List[float], iters=2000, seed=17, kind="mean"):
    rng = np.random.default_rng(seed)
    x = np.asarray(data, float)
    if x.size == 0: return (0.0, 0.0)
    vals = []
    for _ in range(iters):
        s = rng.choice(x, size=x.size, replace=True)
        vals.append(float(np.mean(s) if kind == "mean" else np.median(s)))
    a, b = np.quantile(vals, [0.025, 0.975])
    return float(a), float(b)


def summarize_pairs(a: List[float], b: List[float]) -> Dict[str, Any]:
    diffs = list(np.asarray(a) - np.asarray(b))
    wins = int(np.sum(np.asarray(diffs) > 0))
    losses = int(np.sum(np.asarray(diffs) < 0))
    ties = len(diffs) - wins - losses
    p = sign_test_p(wins, losses)
    m = float(np.mean(diffs)) if diffs else 0.0
    med = float(np.median(diffs)) if diffs else 0.0
    ci_m = bootstrap_ci(diffs, kind="mean")
    ci_med = bootstrap_ci(diffs, kind="median")
    delta = cliffs_delta(diffs)
    return dict(
        wins=wins, losses=losses, ties=ties, p_sign=p,
        mean_diff=m, mean_ci_low=ci_m[0], mean_ci_high=ci_m[1],
        median_diff=med, median_ci_low=ci_med[0], median_ci_high=ci_med[1],
        cliffs_delta=delta
    )
