# glitchlab/core/pipeline.py
"""
---
version: 2
kind: module
id: "core-pipeline"
created_at: "2025-09-11"
name: "glitchlab.core.pipeline"
author: "GlitchLab v2"
role: "Pipeline Orchestrator"
description: >
  Wykonuje sekwencję kroków (steps) na obrazie jako liniowy DAG,
  mierzy metryki in/out, zapisuje różnice i telemetrię do ctx.cache.
  Zapewnia deterministykę względem seed i jednolity kontrakt filtrów v2.
inputs:
  image: {dtype: "uint8", shape: "(H,W,3)", colorspace: "RGB"}
  steps: "list[Step{name:str, params:dict}]"
  ctx:   "Ctx{rng, amplitude, masks, cache, meta}"
outputs:
  image_out: {dtype: "uint8", shape: "(H,W,3)"}
  cache_keys:
    - "stage/{i}/in"
    - "stage/{i}/out"
    - "stage/{i}/diff"
    - "stage/{i}/diff_stats"
    - "stage/{i}/metrics_in"
    - "stage/{i}/metrics_out"
    - "stage/{i}/t_ms"
    - "debug/log"
record_model:
  StageRecord: ["i","name","params","t_ms","metrics_in","metrics_out","diff_stats"]
  RunRecord:   ["run_id","seed","source","stages[]","warnings[]","versions"]
interfaces:
  exports: ["normalize_preset","build_ctx","apply_pipeline"]
  depends_on: ["glitchlab.core.registry","glitchlab.analysis.metrics","glitchlab.analysis.diff","Pillow"]
  used_by: ["glitchlab.gui","glitchlab.analysis.exporters","glitchlab.core.graph"]
policy:
  fail_fast: true
  gather_metrics: true
constraints:
  - "brak SciPy/OpenCV"
  - "filtry muszą mieć sygnaturę (img:uint8 RGB, ctx:Ctx, **params)->np.ndarray"
telemetry:
  metrics: ["entropy","edge_density","contrast_rms"]
  diffs: ["heatmap","stats{mean,p95,max}"]
hud:
  channels:
    thumbs: {in: "stage/{i}/in", out: "stage/{i}/out", diff: "stage/{i}/diff"}
    metrics: {in: "stage/{i}/metrics_in", out: "stage/{i}/metrics_out", diff: "stage/{i}/diff_stats"}
    debug: "debug/log"
license: "Proprietary"
---

Jak powstaje ten front-matter:

1) Określamy KONTEKST: version=2, kind=module — to standard v2-mosaic dla plików kodu.
2) Identyfikacja: id (stały lub UUID), name (pełna ścieżka modułu), author, created_at.
3) Rola i opis: 'role' i 'description' syntetycznie mówią, co moduł robi w architekturze.
4) Interfejs I/O: 'inputs' i 'outputs' to kontrakt wykonawczy — bezpośrednio mapują się
   na typy w kodzie (dtype/shape) i klucze w ctx.cache.
5) Model rekordów: 'record_model' wyrównuje pipeline z warstwą Analysis/HUD (Stage/Run).
6) Interfaces: kto z nas korzysta (used_by) i od czego zależymy (depends_on).
7) Policy/constraints: zasady wykonania (fail_fast, gather_metrics) i ograniczenia (bez SciPy).
8) Telemetry/HUD: jakie kanały zasila pipeline — spójne z AN-1..AN-10 (klucze ctx.cache).

Jak stworzyć własny nagłówek:

a) Skopiuj blok --- ... --- i uzupełnij pola: id, name, role, description.
b) Dopasuj 'inputs/outputs' do realnych typów i kluczy cache które moduł zapisuje.
c) Zaktualizuj 'interfaces.depends_on/used_by' wg realnych importów i konsumentów.
d) Jeżeli moduł publikuje mozaikę lub AST, dodaj w 'hud.channels' odpowiednie klucze
   (np. 'mosaic: {image: "stage/{i}/mosaic", meta: "stage/{i}/mosaic_meta"}').
e) 'created_at' trzymaj w ISO-8601 (YYYY-MM-DD) lub z czasem (YYYY-MM-DDThh:mm:ssZ).
f) 'policy' możesz nadpisać na poziomie pliku (np. domyślnie fail_fast:false dla narzędzia batch).
"""
# glitchlab/core/pipeline.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Tuple, TypedDict
import time
import uuid

import numpy as np
from PIL import Image, ImageFilter

from glitchlab.core.registry import get as registry_get, meta as registry_meta
from glitchlab.core.metrics.basic import (
    compute_entropy,
    edge_density,
    contrast_rms,
    to_gray_f32 as _gray01,  # dla metryk
)
# Graf jest opcjonalny — jeśli nie ma modułu, pipeline i tak działa
try:
    from glitchlab.core.graph import build_and_export_graph  # type: ignore
except Exception:
    build_and_export_graph = None  # type: ignore


# -------------------------------------------------------------------------------------------------
# Typy publiczne
# -------------------------------------------------------------------------------------------------

class Step(TypedDict):
    name: str
    params: Dict[str, Any]


@dataclass
class Ctx:
    rng: np.random.Generator
    amplitude: Optional[np.ndarray]          # (H,W) float32 [0..1] lub None
    masks: Dict[str, np.ndarray]             # np.ndarray (H,W) float32 [0..1]
    cache: Dict[str, Any]
    meta: Dict[str, Any]                     # np. {"source": {...}, "versions": {...}}


# -------------------------------------------------------------------------------------------------
# Presety v2 — normalizacja
# -------------------------------------------------------------------------------------------------

def normalize_preset(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Ujednolica preset do schematu v2.
    Wymaga co najmniej: version, name?, steps: list[{name,params}]
    Dopuszcza (opcjonalnie): amplitude{kind,strength,...}, edge_mask{thresh,ksize,dilate}
    """
    if not isinstance(cfg, Mapping):
        raise ValueError("normalize_preset: cfg must be a mapping/dict")
    out: Dict[str, Any] = dict(cfg)

    # Obsłuż warianty „starego” stylu (root: steps) lub {preset_name: { ... }}
    if "version" not in out and "steps" in out and isinstance(out["steps"], list):
        out["version"] = 2
    if "version" not in out and len(out) == 1:
        # {name: {...}}
        k = next(iter(out))
        body = out[k]
        if isinstance(body, Mapping):
            body = dict(body)
            body.setdefault("version", 2)
            body.setdefault("name", k)
            out = body

    # Wymuś v2
    out.setdefault("version", 2)
    if out["version"] != 2:
        raise ValueError("normalize_preset: only version: 2 is supported")

    # Struktury opcjonalne
    amp = out.get("amplitude") or {"kind": "none", "strength": 1.0}
    if not isinstance(amp, Mapping):
        amp = {"kind": "none", "strength": 1.0}
    amp = dict(amp)
    amp.setdefault("kind", "none")
    amp.setdefault("strength", 1.0)
    out["amplitude"] = amp

    edge = out.get("edge_mask") or {"thresh": 60, "dilate": 0, "ksize": 3}
    if not isinstance(edge, Mapping):
        edge = {"thresh": 60, "dilate": 0, "ksize": 3}
    edge = dict(edge)
    edge.setdefault("thresh", 60)
    edge.setdefault("dilate", 0)
    edge.setdefault("ksize", 3)
    out["edge_mask"] = edge

    # Kroki
    steps = out.get("steps", [])
    if not isinstance(steps, list):
        raise ValueError("normalize_preset: steps must be a list")
    fixed_steps: List[Step] = []
    for i, st in enumerate(steps):
        if not isinstance(st, Mapping):
            raise ValueError(f"normalize_preset: step[{i}] must be a mapping")
        nm = st.get("name")
        pr = st.get("params", {})
        if not isinstance(nm, str) or not nm:
            raise ValueError(f"normalize_preset: step[{i}] missing/invalid 'name'")
        if not isinstance(pr, Mapping):
            raise ValueError(f"normalize_preset: step[{i}].params must be a mapping")
        fixed_steps.append({"name": nm, "params": dict(pr)})
    out["steps"] = fixed_steps

    # Nazwa preset
    if "name" not in out:
        out["name"] = "Preset v2"
    return out


# -------------------------------------------------------------------------------------------------
# Budowa kontekstu (RNG, amplitude, edge mask)
# -------------------------------------------------------------------------------------------------

def _resize_float01(arr: np.ndarray, size_wh: Tuple[int, int]) -> np.ndarray:
    """Resize float [0,1] → float [0,1], bicubic via Pillow."""
    W, H = int(size_wh[0]), int(size_wh[1])
    im = Image.fromarray((np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), mode="L")
    im = im.resize((W, H), resample=Image.BICUBIC)
    return np.asarray(im, dtype=np.float32) / 255.0


def _build_amplitude(shape_hw: Tuple[int, int], spec: Mapping[str, Any]) -> Optional[np.ndarray]:
    H, W = shape_hw
    kind = str(spec.get("kind", "none")).lower()
    strength = float(spec.get("strength", 1.0))
    strength = max(0.0, strength)
    if kind == "none" or strength == 0.0:
        return None

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    if kind == "linear_x":
        amp = (xx / max(1, W - 1))
    elif kind == "linear_y":
        amp = (yy / max(1, H - 1))
    elif kind == "radial":
        cx = (W - 1) * 0.5
        cy = (H - 1) * 0.5
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        r /= max(1e-6, np.sqrt(cx * cx + cy * cy))
        amp = 1.0 - np.clip(r, 0.0, 1.0)
    elif kind == "mask":
        key = spec.get("mask_key")
        # Zostanie wpięte w build_ctx po stworzeniu maski — tutaj tylko sygnalizujemy None
        # bo maska wymaga ctx.masks.
        return None
    elif kind == "perlin":
        scale = float(spec.get("scale", 96.0))
        octaves = int(spec.get("octaves", 4))
        persistence = float(spec.get("persistence", 0.5))
        lacunarity = float(spec.get("lacunarity", 2.0))
        base = int(spec.get("base", 0))
        try:
            from noise import pnoise2  # type: ignore
            amp = np.zeros((H, W), dtype=np.float32)
            fx = max(1e-6, scale)
            fy = max(1e-6, scale)
            for j in range(H):
                for i in range(W):
                    amp[j, i] = pnoise2(i / fx, j / fy, octaves=octaves,
                                        persistence=persistence, lacunarity=lacunarity, base=base)
            # pnoise2 → [-1,1], normalizuj do [0,1]
            amp = (amp - amp.min()) / max(1e-6, (amp.max() - amp.min()))
        except Exception:
            # Fallback: „value-noise” przez ziarno + blur (szybki)
            rng = np.random.default_rng(base or None)
            # mapa ziaren co ~scale px
            grid = max(4, int(round(scale)))
            gh = max(1, H // grid + 2)
            gw = max(1, W // grid + 2)
            coarse = rng.random((gh, gw), dtype=np.float32)
            amp = _resize_float01(coarse, (W, H))
            # wygładzamy lekkim blur
            im = Image.fromarray((amp * 255.0 + 0.5).astype(np.uint8), mode="L")
            im = im.filter(ImageFilter.BoxBlur(radius=max(1, int(round(scale * 0.1)))))
            amp = np.asarray(im, dtype=np.uint8).astype(np.float32) / 255.0
    else:
        return None

    amp = np.clip(amp * strength, 0.0, 1.0)
    return amp.astype(np.float32, copy=False)


def _build_edge_mask(img_u8: np.ndarray, spec: Mapping[str, Any]) -> np.ndarray:
    """Szybka maska krawędzi: |∇x|+|∇y|, próg 0..255; opcjonalna dylacja (MaxFilter)."""
    thresh = int(spec.get("thresh", 60))
    dilate = int(spec.get("dilate", 0))
    _ = int(spec.get("ksize", 3))  # pozostawione dla kompatybilności interfejsu (tu niewykorzystywane)
    g = _gray01(img_u8)  # [0,1]
    gx = np.zeros_like(g, dtype=np.float32); gy = np.zeros_like(g, dtype=np.float32)
    gx[:, 1:] = g[:, 1:] - g[:, :-1]
    gy[1:, :] = g[1:, :] - g[:-1, :]
    mag = np.abs(gx) + np.abs(gy)
    m = (mag * 255.0) >= float(thresh)
    m_u8 = (m.astype(np.uint8) * 255)
    if dilate > 0:
        # MaxFilter kernel ~ (2*dilate+1)
        k = max(1, 2 * dilate + 1)
        im = Image.fromarray(m_u8, mode="L").filter(ImageFilter.MaxFilter(k))
        m_u8 = np.asarray(im, dtype=np.uint8)
    return (m_u8.astype(np.float32) / 255.0).astype(np.float32)


def build_ctx(img_u8: np.ndarray, *, seed: Optional[int], cfg: Optional[Mapping[str, Any]]) -> Ctx:
    """
    Buduje kontekst przetwarzania:
      - deterministyczny RNG,
      - amplitude (opcjonalna) z cfg['amplitude'],
      - maski (w tym 'edge') z cfg['edge_mask'],
      - cache/meta + run_id.
    """
    if img_u8.ndim != 3 or img_u8.shape[-1] != 3:
        raise ValueError("build_ctx: expected uint8 RGB (H,W,3)")

    H, W, _ = img_u8.shape
    rng = np.random.default_rng(seed)
    cache: Dict[str, Any] = {}
    meta: Dict[str, Any] = {}

    cfg_n = normalize_preset(cfg or {"version": 2, "steps": []})
    cache["cfg/preset"] = cfg_n
    cache["cfg/amplitude"] = dict(cfg_n.get("amplitude", {}))
    cache["cfg/edge_mask"] = dict(cfg_n.get("edge_mask", {}))

    amp = _build_amplitude((H, W), cfg_n.get("amplitude", {}))
    masks: Dict[str, np.ndarray] = {}

    # Jeżeli amplitude=mask: przepisz z mask_key (zbudujemy poniżej)
    amp_spec = cfg_n.get("amplitude", {})
    if str(amp_spec.get("kind", "")).lower() == "mask":
        key = amp_spec.get("mask_key")
        if isinstance(key, str) and key:
            # utworzymy po zbudowaniu edge; na razie pusta — pipeline użyje ctx.amplitude None i use_amp
            amp = None
            # oznacz dla build_ctx końcowego do przepisania
            meta["amplitude_mask_key"] = key

    # Maska krawędzi
    masks["edge"] = _build_edge_mask(img_u8, cfg_n.get("edge_mask", {}))

    # Jeżeli amplitude miała kind=mask → przypisz ampurowi wskazaną maskę
    if meta.get("amplitude_mask_key"):
        mk = meta["amplitude_mask_key"]
        if mk in masks:
            amp = masks[mk].astype(np.float32, copy=False)
        del meta["amplitude_mask_key"]

    # run_id
    cache.setdefault("run/id", uuid.uuid4().hex)

    return Ctx(rng=rng, amplitude=amp, masks=masks, cache=cache, meta=meta)


# -------------------------------------------------------------------------------------------------
# Pipeline — wykonanie kroków + telemetria HUD
# -------------------------------------------------------------------------------------------------

def _thumb_rgb(u8: np.ndarray, max_side: int = 1024) -> np.ndarray:
    H, W = u8.shape[:2]
    m = max(H, W)
    if m <= max_side:
        return u8
    scale = max_side / float(m)
    new_size = (max(1, int(round(W * scale))), max(1, int(round(H * scale))))
    im = Image.fromarray(u8, mode="RGB").resize(new_size, resample=Image.BICUBIC)
    return np.asarray(im, dtype=np.uint8)


def _compute_diff(a_u8: np.ndarray, b_u8: np.ndarray, max_side: int = 1024) -> Dict[str, Any]:
    """Lekki diff: |Δ| gray, stats mean/p95/max oraz per-channel |Δ| (0..1)."""
    if a_u8.shape != b_u8.shape:
        raise ValueError("diff: shapes must match")
    a = _thumb_rgb(a_u8, max_side)
    b = _thumb_rgb(b_u8, max_side)
    ga = _gray01(a)
    gb = _gray01(b)
    d = np.abs(ga - gb)
    # statystyki
    flat = d.reshape(-1)
    mean = float(flat.mean()) if flat.size else 0.0
    p95 = float(np.percentile(flat, 95.0)) if flat.size else 0.0
    mx = float(d.max()) if flat.size else 0.0
    # per channel abs
    pa = np.abs(a.astype(np.float32) / 255.0 - b.astype(np.float32) / 255.0)
    return {
        "abs": d.astype(np.float32),
        "per_channel": (pa[..., 0], pa[..., 1], pa[..., 2]),
        "heatmap": d.astype(np.float32),
        "stats": {"mean": mean, "p95": p95, "max": mx},
    }


def _gather_metrics(u8: np.ndarray) -> Dict[str, float]:
    g = _gray01(u8)
    return {
        "entropy": float(compute_entropy(g)),
        "edge_density": float(edge_density(g)),
        "contrast_rms": float(contrast_rms(g)),
    }


def _apply_wrapper_mask_amp(
    src_u8: np.ndarray,
    fx_u8: np.ndarray,
    *,
    ctx: Ctx,
    mask_key: Optional[str],
    use_amp: Any,
    clamp: bool,
) -> np.ndarray:
    """Zewnętrzny wrapper: blend efektu z oryginałem wg mask_key i amplitude*use_amp."""
    H, W = src_u8.shape[:2]
    base = src_u8.astype(np.float32) / 255.0
    eff = fx_u8.astype(np.float32) / 255.0
    m: Optional[np.ndarray] = None

    if mask_key:
        m_candidate = ctx.masks.get(mask_key)
        if m_candidate is not None:
            if m_candidate.shape != (H, W):
                m = _resize_float01(m_candidate, (W, H))
            else:
                m = np.clip(m_candidate.astype(np.float32, copy=False), 0.0, 1.0)

    amp = ctx.amplitude
    if isinstance(use_amp, (float, int)) and amp is not None:
        scale = float(use_amp)
        if scale > 0.0:
            a = np.clip(amp.astype(np.float32, copy=False) * scale, 0.0, 1.0)
            m = a if m is None else np.clip(m * a, 0.0, 1.0)
        else:
            # 0.0 → brak efektu
            m = np.zeros((H, W), dtype=np.float32)
    elif use_amp is False:
        m = np.zeros((H, W), dtype=np.float32)

    if m is None:
        out_f = eff
    else:
        out_f = base * (1.0 - m[:, :, None]) + eff * m[:, :, None]

    if clamp:
        out_f = np.clip(out_f, 0.0, 1.0)
    return (out_f * 255.0 + 0.5).astype(np.uint8)


def apply_pipeline(
    img_u8: np.ndarray,
    ctx: Ctx,
    steps: List[Step],
    *,
    fail_fast: bool = True,
    debug_log: Optional[List[str]] = None,
    metrics: bool = True,
) -> np.ndarray:
    """
    Wykonuje kroki pipeline, zapisując telemetrię dla HUD:
      - stage/{i}/in|out (miniatury RGB)
      - stage/{i}/metrics_in|metrics_out (entropy/edge_density/contrast_rms)
      - stage/{i}/diff (abs/heatmap/per_channel) + stage/{i}/diff_stats
      - stage/{i}/t_ms (czas kroku)
    Po zakończeniu: (jeśli dostępny core.graph) zapisuje JSON grafu pod 'ast/json'.
    """
    if img_u8.ndim != 3 or img_u8.shape[-1] != 3 or img_u8.dtype != np.uint8:
        raise ValueError("apply_pipeline: expected uint8 RGB (H,W,3)")
    out = img_u8
    cache = ctx.cache
    dbg = debug_log if debug_log is not None else []

    # minimalna walidacja
    if not isinstance(steps, list):
        raise ValueError("apply_pipeline: steps must be a list")

    for i, step in enumerate(steps):
        name = step.get("name")
        params_in = dict(step.get("params", {}))
        if not isinstance(name, str) or not name:
            raise ValueError(f"apply_pipeline: step[{i}] invalid 'name'")
        try:
            fn = registry_get(name)
        except KeyError:
            msg = f"[pipeline] step[{i}] '{name}' not found in registry"
            dbg.append(msg)
            if fail_fast:
                raise KeyError(msg)
            else:
                # telemetria brakującego kroku
                cache[f"stage/{i}/in"] = _thumb_rgb(out)
                cache[f"stage/{i}/t_ms"] = 0.0
                cache[f"stage/{i}/metrics_in"] = _gather_metrics(out) if metrics else {}
                cache[f"stage/{i}/metrics_out"] = {}
                cache[f"stage/{i}/diff_stats"] = {"mean": 0.0, "p95": 0.0, "max": 0.0}
                continue

        # Zbierz defaults i odfiltruj nieznane parametry (loguj ostrzeżenia)
        defs = registry_meta(name)["defaults"]
        eff_params = {**defs, **params_in}
        unknown = [k for k in params_in.keys() if k not in defs]
        if unknown:
            dbg.append(f"[pipeline] step[{i}] '{name}': unknown params {unknown}")

        # Wyciągnij wspólne i usuń z wywołania filtra (wrapper zastosujemy zewnętrznie)
        mask_key = eff_params.pop("mask_key", None)
        use_amp = eff_params.pop("use_amp", 1.0)
        clamp = bool(eff_params.pop("clamp", True))

        # Telemetria: in + metrics_in
        cache[f"stage/{i}/in"] = _thumb_rgb(out)
        m_in = _gather_metrics(out) if metrics else {}

        t0 = time.perf_counter()
        try:
            fx = fn(out, ctx, **eff_params)  # filtr v2 przyjmuje (img_u8, ctx, **params)
            if not (isinstance(fx, np.ndarray) and fx.ndim == 3 and fx.shape[-1] == 3 and fx.dtype == np.uint8):
                # zabezpieczenie: jeśli filtr zwrócił float, spróbuj skonwertować
                if isinstance(fx, np.ndarray) and fx.ndim == 3 and fx.shape[-1] == 3:
                    fx_u8 = np.clip(fx.astype(np.float32), 0.0, 1.0)
                    fx = (fx_u8 * 255.0 + 0.5).astype(np.uint8)
                else:
                    raise TypeError(f"filter '{name}' must return uint8 RGB (H,W,3)")
            # Zastosuj wspólny wrapper (mask/amplitude/clamp) na wyniku filtra
            out_next = _apply_wrapper_mask_amp(out, fx, ctx=ctx, mask_key=mask_key, use_amp=use_amp, clamp=clamp)
            t_ms = (time.perf_counter() - t0) * 1000.0
        except Exception as ex:
            msg = f"[pipeline] step[{i}] '{name}' failed: {ex!r}"
            dbg.append(msg)
            if fail_fast:
                raise
            # przy kontynuacji: pozostaw obraz bez zmian, ale zanotuj metryki
            cache[f"stage/{i}/t_ms"] = 0.0
            cache[f"stage/{i}/metrics_in"] = m_in
            cache[f"stage/{i}/metrics_out"] = {}
            cache[f"stage/{i}/diff_stats"] = {"mean": 0.0, "p95": 0.0, "max": 0.0}
            continue

        # Telemetria: out + metrics_out + diff
        cache[f"stage/{i}/out"] = _thumb_rgb(out_next)
        m_out = _gather_metrics(out_next) if metrics else {}
        cache[f"stage/{i}/metrics_in"] = m_in
        cache[f"stage/{i}/metrics_out"] = m_out
        cache[f"stage/{i}/t_ms"] = float(t_ms)

        d = _compute_diff(out, out_next)
        cache[f"stage/{i}/diff"] = d["abs"]          # Gray 0..1 (float32, miniatura)
        cache[f"stage/{i}/diff_stats"] = d["stats"]  # {"mean","p95","max"}

        # przejście do kolejnego etapu
        out = out_next

    # Zapisz debuglog
    if dbg is not debug_log and dbg:
        # jeżeli przekazano listę, już została uzupełniona; w przeciwnym razie wrzuć do cache
        ctx.cache["debug/log"] = list(dbg)

    # Export grafu (jeśli dostępna biblioteka core.graph)
    if build_and_export_graph is not None:
        try:
            build_and_export_graph(steps, ctx.cache, attach_delta=True, cache_key="ast/json")  # type: ignore
        except Exception as ex:
            dbg.append(f"[pipeline] graph export failed: {ex!r}")

    return out

