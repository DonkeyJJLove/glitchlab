# scratch.py — minimalny prototyp metryki mozaiki (D_M)

import numpy as np
import itertools
import math

# ---- mozaika 5x5 ----
R, C = 5, 5
rng = np.random.default_rng(0)
edge = rng.random(R * C)  # edge density per tile in [0,1]
roi = np.zeros(R * C, dtype=float)
for r in range(1, 4):
    for c in range(1, 4):
        roi[r * C + c] = 1.0  # środkowy kwadrat jako ROI


def tile_dist(i: int, j: int, alpha=1.0, beta=1.0, gamma=1.0) -> float:
    """Pseudodystans między kaflami: geo + różnica cech + kara za różne etykiety."""
    x1, y1 = i % C, i // C
    x2, y2 = j % C, j // C
    geo = math.hypot(x1 - x2, y1 - y2)
    feat_diff = abs(float(edge[i]) - float(edge[j]))
    label_pen = 1.0 if ((edge[i] > 0.5) != (edge[j] > 0.5)) else 0.0
    return alpha * geo + beta * feat_diff + gamma * label_pen


def D_M(S, S2, alpha=1.0, beta=1.0, gamma=1.0) -> float:
    """Earth-mover-like dopasowanie dla dwóch zbiorów o tej samej liczności."""
    if len(S) != len(S2):
        raise ValueError("Zbiory muszą mieć tę samą liczność.")
    best = float("inf")
    for perm in itertools.permutations(S2):
        cost = 0.0
        for i, j in zip(S, perm):
            cost += tile_dist(i, j, alpha, beta, gamma)
        best = min(best, cost)
    return best


# ---- przykładowe regiony ----
ROI = [6, 7, 8]  # 3 kafle ze środka (r=1..3, c=1..3) – uproszczony przykład
TOP = [0, 1, 2]  # górny rząd

print("D_M(ROI, ROI) =", D_M(ROI, ROI))
print("D_M(ROI, TOP) =", D_M(ROI, TOP))
