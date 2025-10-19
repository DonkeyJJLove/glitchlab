# glitchlab/app/state.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class UiState:
    image: Any | None = None  # PIL.Image
    preset_cfg: Dict[str, Any] | None = None
    single_filter: str | None = None
    filter_params: Dict[str, Any] = field(default_factory=dict)
    seed: int = 7
    hud_mapping: Dict[str, list[str]] = field(default_factory=lambda: {
        "slot1": ["stage/0/in", "stage/0/metrics_in", "format/jpg_grid"],
        "slot2": ["stage/0/out", "stage/0/metrics_out", "stage/0/fft_mag"],
        "slot3": ["stage/0/diff", "stage/0/diff_stats", "ast/json"],
    })
