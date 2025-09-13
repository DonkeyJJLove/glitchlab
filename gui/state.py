"""
---
version: 3
kind: module
id: "gui-state"
created_at: "2025-09-13"
name: "glitchlab.gui.state"
author: "GlitchLab v3"
role: "UI State, Reducer & Selectors (unidirectional flow)"
description: >
Kanoniczny stan aplikacji GUI z czystym reduktorem zdarzeń i selektorami do odczytu.
Zapewnia niezmienniczy model danych, wspiera mapowanie HUD i integrację z Services.
inputs:
events:
- "ui.image.set {image}"
- "ui.seed.set {seed}"
- "ui.filter.set {name, params?}"
- "ui.filter.params.merge {params}"
- "ui.preset.set {cfg}"
- "ui.hud.map.set {mapping}"
- "ui.layout.set {layout}"
- "run.request {}"
- "run.progress {percent, message?}"
- "run.done {out, ctx}"
- "run.error {error}"
state:
image_in: "PIL.Image|np.ndarray|None"
image_out: "PIL.Image|np.ndarray|None"
preset_cfg: "dict|None"
single_filter: "str|None"
filter_params: "dict"
seed: "int"
last_ctx: "Any|None # oczekuje .cache lub {'cache':...}"
hud_mapping: "dict[str, list[str]] # slot→priorytety kluczy"
layout: "dict"
in_progress: "bool"
progress_pct: "float"
progress_msg: "str|None"
last_error: "str|None"
outputs:
selectors:
current_image(): "image_out || image_in"
cache(): "dict # płytka kopia ctx.cache"
cache_get(key, default): "Any"
hud_slot(slot): "{slot, selected_key, value, candidates}"
running(): "(in_progress, progress_pct, progress_msg)"
error(): "str|None"
filter_selection(): "(single_filter, filter_params)"
preset(): "dict|None"
layout(): "dict"
interfaces:
exports:
- "UiState # dataclass (frozen)"
- "reduce(state, (topic, payload)) -> UiState"
- "Selectors"
- "initial_state() -> UiState"
- "make_event(topic, **payload) -> (topic, payload)"
- "DEFAULT_HUD_MAPPING"
depends_on: ["dataclasses","typing"]
used_by: ["glitchlab.gui.app_shell","glitchlab.gui.views.","glitchlab.gui.services."]
policy:
deterministic: true
side_effects: false
purity: "reducer bez I/O i bez wywołań GUI"
constraints:

"brak odwołań do Tkintera"

"ctx.cache traktowany jako słownik (bez interpretacji macierzy)"
hud:
mapping_defaults:
slot1: ["stage/0/in","stage/0/metrics_in","format/jpg_grid"]
slot2: ["stage/0/out","stage/0/metrics_out","stage/0/fft_mag"]
slot3: ["stage/0/diff","stage/0/diff_stats","ast/json"]
license: "Proprietary"
---
"""
# glitchlab/gui/state.py
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Mapping, Optional, Tuple

# Domyślne mapowanie slotów HUD → lista preferowanych kluczy (pierwszy istniejący wygrywa)
DEFAULT_HUD_MAPPING: Dict[str, List[str]] = {
    "slot1": ["stage/0/in", "stage/0/metrics_in", "format/jpg_grid"],
    "slot2": ["stage/0/out", "stage/0/metrics_out", "stage/0/fft_mag"],
    "slot3": ["stage/0/diff", "stage/0/diff_stats", "ast/json"],
}


@dataclass(frozen=True)
class UiState:
    """
    Kanoniczny stan GUI (unidirectional data flow).
    """
    # Obrazy
    image_in: Any | None = None          # wejściowy obraz (np. PIL.Image lub np.ndarray uint8 RGB)
    image_out: Any | None = None         # ostatni wynik przetwarzania

    # Preset / Filtr
    preset_cfg: Dict[str, Any] | None = None
    single_filter: str | None = None
    filter_params: Dict[str, Any] = field(default_factory=dict)
    seed: int = 7

    # Kontekst ostatniego uruchomienia (cache/HUD)
    last_ctx: Any | None = None          # obiekt posiadający .cache (dict) lub dict z kluczem "cache"

    # HUD i layout
    hud_mapping: Dict[str, List[str]] = field(default_factory=lambda: {k: v[:] for k, v in DEFAULT_HUD_MAPPING.items()})
    layout: Dict[str, Any] = field(default_factory=dict)

    # Status uruchomień
    in_progress: bool = False
    progress_pct: float = 0.0
    progress_msg: str | None = None
    last_error: str | None = None


# --------------------------------------------------------------------------------------
# Reducer — jedyna droga modyfikacji UiState
# --------------------------------------------------------------------------------------

def reduce(state: UiState, event: Tuple[str, Mapping[str, Any]]) -> UiState:
    """
    Reduktor stanu (czysty, bez I/O).
    Przyjmuje (topic, payload), zwraca nowy UiState (immutability).
    Obsługiwane tematy (skrót):
      - ui.image.set {image}
      - ui.seed.set {seed}
      - ui.filter.set {name, params?}
      - ui.filter.params.merge {params}
      - ui.preset.set {cfg}
      - ui.hud.map.set {mapping}
      - ui.layout.set {layout}
      - run.request {}
      - run.progress {percent, message?}
      - run.done {out, ctx}
      - run.error {error}
    """
    topic, payload = event
    p = dict(payload or {})

    # UI: obraz wejściowy
    if topic == "ui.image.set":
        return replace(state, image_in=p.get("image"), image_out=None, last_error=None)

    # UI: seed
    if topic == "ui.seed.set":
        seed = int(p.get("seed", state.seed))
        return replace(state, seed=seed)

    # UI: wybór filtra
    if topic == "ui.filter.set":
        name = p.get("name")
        params = dict(p.get("params", {})) if isinstance(p.get("params"), Mapping) else {}
        return replace(state, single_filter=name, filter_params=params, last_error=None)

    # UI: scalenie parametrów filtra
    if topic == "ui.filter.params.merge":
        merge_params = dict(p.get("params", {})) if isinstance(p.get("params"), Mapping) else {}
        new_params = {**state.filter_params, **merge_params}
        return replace(state, filter_params=new_params)

    # UI: preset
    if topic == "ui.preset.set":
        cfg = dict(p.get("cfg", {})) if isinstance(p.get("cfg"), Mapping) else None
        return replace(state, preset_cfg=cfg, last_error=None)

    # UI: HUD mapping
    if topic == "ui.hud.map.set":
        mapping = p.get("mapping")
        if isinstance(mapping, Mapping):
            # tylko listy str
            clean = {str(k): [str(x) for x in list(v)] for k, v in mapping.items() if isinstance(v, (list, tuple))}
            return replace(state, hud_mapping=clean)
        return state

    # UI: layout (dock/float, geometrii itp.)
    if topic == "ui.layout.set":
        layout = p.get("layout")
        if isinstance(layout, Mapping):
            return replace(state, layout=dict(layout))
        return state

    # RUN: start/progress/done/error
    if topic == "run.request":
        return replace(state, in_progress=True, progress_pct=0.0, progress_msg=None, last_error=None)

    if topic == "run.progress":
        pct = float(p.get("percent", state.progress_pct))
        msg = p.get("message", state.progress_msg)
        return replace(state, progress_pct=max(0.0, min(100.0, pct)), progress_msg=msg)

    if topic == "run.done":
        out = p.get("out", state.image_out)
        ctx = p.get("ctx", state.last_ctx)
        return replace(state, image_out=out, last_ctx=ctx, in_progress=False, progress_pct=100.0, progress_msg=None)

    if topic == "run.error":
        err = p.get("error")
        return replace(state, in_progress=False, last_error=str(err) if err is not None else "unknown error")

    # Nieznane — bez zmian
    return state


# --------------------------------------------------------------------------------------
# Selectors — czytelny dostęp do stanu dla widoków
# --------------------------------------------------------------------------------------

class Selectors:
    """
    Zestaw metod ułatwiających czytanie UiState przez widoki.
    """

    @staticmethod
    def current_image(s: UiState) -> Any | None:
        """
        Priorytet: image_out (jeśli istnieje), inaczej image_in.
        """
        return s.image_out if s.image_out is not None else s.image_in

    @staticmethod
    def cache(s: UiState) -> Dict[str, Any]:
        """
        Zwraca referencję cache z last_ctx (jeśli dostępna), w przeciwnym razie pusty dict.
        Akceptuje kontekst w dwóch wariantach:
          - obiekt z atrybutem `.cache` (dict)
          - dict z kluczem "cache"
        """
        ctx = s.last_ctx
        if ctx is None:
            return {}
        if isinstance(ctx, Mapping) and isinstance(ctx.get("cache"), Mapping):
            return dict(ctx["cache"])  # płytka kopia
        cache = getattr(ctx, "cache", None)
        return dict(cache) if isinstance(cache, Mapping) else {}

    @staticmethod
    def cache_get(s: UiState, key: str, default: Any = None) -> Any:
        return Selectors.cache(s).get(key, default)

    @staticmethod
    def hud_slot(s: UiState, slot_id: int | str) -> Dict[str, Any]:
        """
        Zwraca dane do renderu slotu HUD:
          {
            "slot": "slot1|slot2|slot3|<custom>",
            "selected_key": "<klucz z cache>",
            "value": <wartość z cache lub None>,
            "candidates": [lista kluczy branych pod uwagę]
          }
        """
        if isinstance(slot_id, int):
            slot_key = f"slot{slot_id}"
        else:
            slot_key = str(slot_id)
        mapping = s.hud_mapping.get(slot_key, [])
        cache = Selectors.cache(s)

        selected_key = None
        val = None
        for k in mapping:
            if k in cache:
                selected_key = k
                val = cache[k]
                break

        return {
            "slot": slot_key,
            "selected_key": selected_key,
            "value": val,
            "candidates": list(mapping),
        }

    @staticmethod
    def running(s: UiState) -> Tuple[bool, float, Optional[str]]:
        """
        Zwraca (in_progress, progress_pct, progress_msg).
        """
        return s.in_progress, s.progress_pct, s.progress_msg

    @staticmethod
    def error(s: UiState) -> Optional[str]:
        return s.last_error

    @staticmethod
    def filter_selection(s: UiState) -> Tuple[Optional[str], Dict[str, Any]]:
        return s.single_filter, dict(s.filter_params)

    @staticmethod
    def preset(s: UiState) -> Optional[Dict[str, Any]]:
        return dict(s.preset_cfg) if isinstance(s.preset_cfg, Mapping) else None

    @staticmethod
    def layout(s: UiState) -> Dict[str, Any]:
        return dict(s.layout)


# --------------------------------------------------------------------------------------
# Fabryka stanu początkowego
# --------------------------------------------------------------------------------------

def initial_state() -> UiState:
    return UiState()


# --------------------------------------------------------------------------------------
# Proste narzędzia do budowy eventów (opcjonalnie)
# --------------------------------------------------------------------------------------

def make_event(topic: str, **payload: Any) -> Tuple[str, Dict[str, Any]]:
    return topic, dict(payload)

