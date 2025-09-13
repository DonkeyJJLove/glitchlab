# core/ARCHITECTURE.md

# GlitchLab Core v2 — Architektura (warstwa `core/`)

Ten dokument jest **źródłem prawdy** dla warstwy `core/` — opisuje **kontrakty**, **przepływy**, **moduły** oraz **telemetrię HUD** generowaną przez rdzeń. Ustalony tu model jest spójny z plikiem „GlitchLab v2 — System Architecture” (poziom ogólny) oraz z dokumentacją GUI/filters. Wszystkie przykłady i nazwy odpowiadają **rzeczywistym** interfejsom widocznym w kodzie (zob. `/core/*.py`).

---

## 1) Cel i zakres

Warstwa `core/`:

* rejestruje i uruchamia **filtry v2** (jeden kontrakt wywołania),
* normalizuje **presety v2** (amplitude/edge/steps),
* wykonuje **pipeline** kroków i zbiera **telemetrię** (metryki, diff, czasy),
* buduje **lekki graf** procesu (DAG → JSON do HUD),
* udostępnia „soczewkę” **mozaiki** (blokowa projekcja metryk/AST),
* zawiera podstawowe **metryki** (entropy/edges/contrast, PSNR/SSIM),
* dostarcza proste **ROI/maski** i **symbole** (bitmap→mask, proceduralne).

**Założenia technologiczne**

* Python 3.9+, **NumPy**, **Pillow**; *(opcjonalnie)* `noise` (Perlin).
* **Bez** SciPy/OpenCV.
* Obraz wejściowy/wyjściowy: **uint8 RGB (H,W,3)**.
* Wnętrze filtrów: `float32 [0,1]` (zalecenie), ale pipeline akceptuje też `uint8`.

---

## 2) Przegląd modułów (`core/`)

* `registry.py` — rejestr filtrów (kanon nazw, aliasy, defaults/doc).
* `pipeline.py` — normalizacja presetów, `Ctx`, wrapper mask/amplitude/clamp, metryki/diff/czasy, debug log, opcjonalny eksport grafu.
* `graph.py` — budowa liniowego **DAG** i eksport **JSON** (`ast/json`).
* `mosaic.py` — siatki square/hex, raster etykiet, projekcja blokowych metryk na **overlay**.
* `astmap.py` — Python AST → graf semantyczny → projekcja na mozaikę (wspólny format).
* `metrics/basic.py` — entropia, gęstość krawędzi, kontrast RMS, blokowe statystyki.
* `metrics/compare.py` — PSNR i SSIM (box).
* `utils.py` — konwersje (gray/u8/rgb), stabilne clip, resize via Pillow, box blur.
* `roi.py` — prymitywy masek (polygon/rect/circle) + `merge_masks`.
* `symbols.py` — bitmap→mask, symbole proceduralne, `stamp_mask`.

> **Stan w repo**: wszystkie powyższe pliki istnieją i odpowiadają opisanym interfejsom. `pipeline.py` warunkowo importuje `core.graph` (gdy brak — pipeline nadal działa).

---

## 3) Kontrakty publiczne

### 3.1 Filter API v2 (twarde)

```python
# registry-decorated callable
def my_filter(img_u8: np.ndarray, ctx: Ctx, **params) -> np.ndarray
# Wejście: uint8 RGB (H,W,3)
# Wnętrze (rekomend.): float32 [0,1]
# Wyjście: uint8 RGB (H,W,3)

# Wspólne parametry (obsługiwane przez pipeline-wrapper, NIE przekazuj do body filtra):
#   mask_key: str|None  → blend z maską ctx.masks[mask_key] (float32 [0,1])
#   use_amp: float|bool → modulacja siły efektem ctx.amplitude (float32 [0,1])
#   clamp:   bool       → końcowy clip [0,1] (po blendzie)
#
# RNG: tylko ctx.rng; diagnostyki: ctx.cache["diag/<filter>/..."] (swobodne klucze)
```

> Pipeline zdejmuje `mask_key|use_amp|clamp` z `params`, uruchamia filtr, a następnie **zewnętrznie** stosuje blend mask/amplitude i clamp (zob. `_apply_wrapper_mask_amp` w `pipeline.py`).

### 3.2 Registry (`core/registry.py`)

```python
@register(name: str, defaults: dict | None = None, doc: str | None = None) -> decorator
get(name: str) -> callable                   # rozwiązuje aliasy
available() -> list[str]                     # kanoniczne, posortowane
canonical(name: str) -> str                  # alias→kanon (KeyError, gdy brak)
alias(alias_name: str, target_name: str) -> bool
meta(name: str) -> dict                      # {"name","defaults","doc","aliases"}
```

* Nazwy **case-insensitive**.
* Aliasowanie **bez pętli** (ochrona); alias nie może nadpisać istniejącej kanonicznej funkcji (innej niż target).
* `defaults/doc` utrzymywane w rejestrze — GUI buduje panele na ich bazie.

### 3.3 Preset v2 (schema na wejściu pipeline)

```yaml
version: 2
name: "<NAME>"
seed: 7
amplitude: { kind: none|linear_x|linear_y|radial|perlin|mask, strength: 1.0, ... }
edge_mask: { thresh: 60, dilate: 0, ksize: 3 }
steps:
  - name: <canonical_filter_name>
    params: { ...only filter params (bez mask_key/use_amp/clamp)... }
```

`pipeline.normalize_preset(cfg)` wymusza powyższy kształt, uzupełnia brakujące pola domyślne i pilnuje typów.

### 3.4 Kontekst wykonania (`Ctx`, `pipeline.py`)

```python
@dataclass
class Ctx:
    rng: np.random.Generator                 # deterministyczny RNG
    amplitude: np.ndarray | None             # (H,W) f32 [0,1] lub None
    masks: dict[str, np.ndarray]             # {"edge": ...} f32 [0,1]
    cache: dict[str, Any]                    # kanały HUD + meta
    meta: dict[str, Any]                     # dane pomocnicze (tymczasowe)
```

**Kanały HUD (min.) per etap `i`:**

* `stage/{i}/in` (miniatura `u8`), `stage/{i}/out` (`u8`),
  `stage/{i}/diff` (gray `f32 [0,1]`), `stage/{i}/diff_stats` (`{"mean","p95","max"}`).
* `stage/{i}/metrics_in`, `stage/{i}/metrics_out` (`{"entropy","edge_density","contrast_rms"}`).
* `stage/{i}/t_ms` (czas kroku, `float` ms).

**Kanały globalne:**

* `cfg/preset`, `cfg/amplitude`, `cfg/edge_mask`, `run/id`, `debug/log`.
* *(opcjonalnie)* `ast/json` — graf procesu (jeżeli dostępny `core.graph`).

---

## 4) Pipeline: od preset do telemetrii

### 4.1 Normalizacja presetów

`normalize_preset(cfg)`:

* wymusza `version:2`, listę kroków `{name, params}`,
* ustawia domyślne `amplitude` i `edge_mask`,
* toleruje warianty legacy (np. `{ preset_name: {...}}`).

### 4.2 Budowa kontekstu

`build_ctx(img_u8, seed, cfg)`:

* RNG (`np.random.default_rng(seed)`),
* `amplitude` (kind: `none|linear_x|linear_y|radial|mask|perlin`); Perlin: próba `noise.pnoise2`, fallback „value-noise” + blur,
* `masks["edge"]` — szybka maska krawędzi: `|∇x| + |∇y|` z progiem i opcjonalną dylacją (`Pillow.ImageFilter.MaxFilter`),
* zapisuje `cfg/*`, generuje `run/id`.

> `amplitude=mask` — jeśli `mask_key` wskazuje istniejącą maskę (np. `"edge"`), amplitude zostanie z nią **zsynchronizowana** po zbudowaniu masek.

### 4.3 Wykonanie i wrapper mask/amplitude/clamp

`apply_pipeline(img_u8, ctx, steps, fail_fast=True, metrics=True)`:

1. Dla każdego kroku:

   * rozwiązuje `fn = registry.get(name)`,
   * łączy `defaults` z `params`, ostrzega o nieznanych parametrach (do `debug/log`),
   * **wycina** `mask_key|use_amp|clamp` z `params` (wrapper zewnętrzny),
   * zapisuje `stage/{i}/in` i `metrics_in` (opcjonalnie),
   * woła `fn(img_u8, ctx, **eff_params)`.

2. Po zwrocie z filtra:

   * stosuje `_apply_wrapper_mask_amp(...)`:

     * `mask_key` → blend z `ctx.masks[mask_key]` (auto-resize),
     * `use_amp` (float/bool) → modulacja przez `ctx.amplitude`,
     * `clamp` → końcowy clip `[0,1]`,
   * zapisuje `stage/{i}/out`, `metrics_out`, `diff` (|Δ| w gray i statystyki), `t_ms`.

3. Na końcu:

   * dopisuje `debug/log` (jeśli lokalny bufor użyty),
   * *opcjonalnie* buduje `ast/json` przez `core.graph.build_and_export_graph(...)`.

> **Fail-fast**: przy błędzie kroku, gdy `fail_fast=True` — wyjątek; gdy `False` — pipeline loguje błąd i przechodzi dalej, wypełniając kanały minimalnie.

---

## 5) Graf procesu (DAG → JSON)

`core/graph.py`:

* **Wejście**: `steps` + `ctx.cache["stage/{i}/*"]` (metryki/diff/t\_ms).
* **Wyjście**: `Graph{nodes[], edges[], meta{...}}` (wyłącznie prymitywy JSON).
* `build_graph_from_cache(...)` agreguje:

  * `metrics_in/out`, `diff_stats`, `t_ms`, status `"ok"|"missing"`,
  * `delta = metrics_out - metrics_in` (wspólne klucze).
* `export_ast_json(graph, ctx_like, cache_key="ast/json")` — zapis do `ctx.cache`.
* `build_and_export_graph(...)` — skrót: buduje i zapisuje.

**HUD**: GUI odczytuje `ctx.cache["ast/json"]` i rysuje mini-graf w slocie diagnostycznym.

---

## 6) Soczewka mozaiki i mapa AST

### 6.1 `mosaic.py`

* `mosaic_map(shape_hw, mode="square"| "hex", cell_px:int)` → definicja siatki (komórki, centrum, sąsiedzi) + **raster etykiet** `(H,W) int32`.
* `mosaic_label_raster(mosaic)` → sam raster `(H,W)`.
* `mosaic_project_blocks(block_stats, mosaic, map_spec?)` → **RGB u8 overlay** z projekcją statystyk kafelkowych na komórki (normalizacja i paleta wbudowana).
* `mosaic_overlay(img_u8, overlay_u8, alpha=0.5)` → półprzezroczysta nakładka.

**Źródło danych**: zwykle `metrics.basic.block_stats(...)` (np. wariancja/entropia per blok). Wynik overlay trafia do HUD jako `stage/{i}/mosaic` (+ `stage/{i}/mosaic_meta`).

### 6.2 `astmap.py`

* `build_ast(source:str)` → `ast.AST` Pythona (parsowanie deterministyczne).
* `ast_to_graph(tree)` → graf semantyczny `{"nodes":[...], "edges":[...], "meta":{...}}`.
* `project_ast_to_mosaic(graph, mosaic, map_spec?)` → **RGB u8 overlay** (ta sama ścieżka renderu co przy metrykach obrazowych).
* `export_ast_json(graph, ctx_like, cache_key="ast/json")` — współdzieli klucz z grafem pipeline (GUI ma jeden punkt wejścia).

---

## 7) Metryki (`metrics/`)

### 7.1 `metrics/basic.py`

* `to_gray_f32(arr)` — RGB/uint8 → `f32 [0,1]`.
* `downsample_max_side(arr, max_side=1024)` — thumbnail bez aliasingu (Pillow\.BICUBIC).
* `compute_entropy(arr, bins=256)` — Shannon (globalna).
* `edge_density(arr)` — udział silnych gradientów (prosty operator różnicowy).
* `contrast_rms(arr)` — wariancja/odchylenie (globalny kontrast).
* `block_stats(arr, block=16, ...)` — statystyki kafelkowe (słownik `(bx,by)->{...}`).

### 7.2 `metrics/compare.py`

* `psnr(a,b)` — 8-bit, kanały łączone.
* `ssim_box(a,b, win=7)` — uproszczone SSIM (okno pudełkowe).

> Pipeline używa **globalnych** metryk (entropy/edge\_density/contrast\_rms) per `stage/{i}/metrics_*`. `diff` to **lekka** różnica obliczana lokalnie w `pipeline.py` (nie korzysta z `analysis.diff`).

---

## 8) ROI i symbole

### 8.1 `roi.py`

* `mask_polygon(shape_hw, points, feather=0)`
* `mask_rect(shape_hw, xyxy, feather=0)`
* `mask_circle(shape_hw, center, radius, feather=0)`
* `merge_masks(op, *masks)` — `max|min|mean|mul` (float32 `[0,1]`).

Maski są **czyste** i deterministyczne; feather przez `ImageFilter.BoxBlur`.

### 8.2 `symbols.py`

* `bitmap_to_mask(img_u8, channel=0, invert=False, thresh=127)` — binarna maska `f32 [0,1]`.
* `load_symbol(name, size_hw)` — wbudowane symbole (`circle|ring|square|triangle|plus|cross|diamond|hex`).
* `stamp_mask(dst, mask, xy, mode="max|min|mean|mul")` — nakładanie na płótno (wyjście przycinane do kadrów).

---

## 9) Telemetria HUD — słownik kluczy

**Per etap `i`:**

* `stage/{i}/in` — miniatura `RGB u8`.
* `stage/{i}/out` — miniatura `RGB u8`.
* `stage/{i}/diff` — `f32 [0,1]` (abs gray); statystyki w `stage/{i}/diff_stats`.
* `stage/{i}/metrics_in` / `stage/{i}/metrics_out` — `{"entropy","edge_density","contrast_rms"}`.
* `stage/{i}/t_ms` — czas etapu (float).

**Opcjonalnie (gdy używane):**

* `stage/{i}/mosaic`, `stage/{i}/mosaic_meta` — overlay mozaiki i metadane.
* `ast/json` — graf procesu i/lub AST (JSON, bez macierzy).
* `debug/log` — lista ostrzeżeń/zdarzeń pipeline.

GUI jest **czytnikiem cache** — nie liczy metryk; renderuje na podstawie kluczy.

---

## 10) Zasady jakości, błędy i deterministyka

* **Deterministyczność**: wszystkie losowości przez `ctx.rng`; amplitude Perlin ma deterministyczny `base`; fallback value-noise inicjowany RNG bazującym na `base`.
* **Fail-fast** domyślnie `True`; gdy `False`, telemetria etapu jest minimalna, pipeline idzie dalej.
* **Parametry nadmiarowe**: **nie zrywają** wykonania — logowane do `debug/log`, ignorowane przy wywołaniu filtra.
* **Walidacja kształtów**: pipeline sprawdza `uint8 RGB (H,W,3)`; wrapper skaluje maskę do rozmiaru wejścia.

---

## 11) Wydajność

* Miniatury/metryki liczone na **thumbie** (max side \~1024).
* Unikanie pętli per-piksel (NumPy, operacje blokowe; wyjątek: fallback Perlin, lecz tylko przy braku `noise.pnoise2`).
* Resize i proste operacje przez **Pillow** (BICUBIC / BoxBlur / MaxFilter).
* Bufory: `float32` wewnątrz, **konwersja do `uint8`** na granicach.

---

## 12) Rozszerzanie — wzorce

### 12.1 Nowy filtr

```python
from glitchlab.core.registry import register

DEFAULTS = {
    "strength": 1.0,
    "mask_key": None,   # obsłuży wrapper
    "use_amp": 1.0,     # obsłuży wrapper
    "clamp": True       # obsłuży wrapper
}
DOC = "Krótki opis techniczny."

@register("my_filter", defaults=DEFAULTS, doc=DOC)
def my_filter(img_u8, ctx, **p):
    x = img_u8.astype(np.float32) / 255.0
    strength = float(p.get("strength", 1.0))
    # ... oblicz efekt w [0,1] ...
    out = np.clip(x * (1.0 + 0.25 * strength), 0.0, 1.0)
    # diagnostyki (dowolne klucze, np.):
    ctx.cache["diag/my_filter/strength"] = strength
    return (out * 255.0 + 0.5).astype(np.uint8)
```

**Zalecenia**: zero pętli per-piksel; `ctx.rng` do losowości; diagnostyki nazwij stabilnie (`diag/<filter>/...`).

### 12.2 Nowa metryka

Dodaj do `metrics/basic.py` i podłącz w `pipeline._gather_metrics`. Zachowaj ≤50 ms @ 1K.

### 12.3 Nowy overlay mozaiki

Wykorzystaj `mosaic_map` + `mosaic_project_blocks` i zapisz wynik do `ctx.cache["stage/{i}/mosaic"]`.

---

## 13) Testy i smoke

* **Smoke pipeline**: `zeros(128,128,3)` + pierwszy dostępny filtr → brak wyjątków; sprawdź obecność kluczy `stage/{i}/*`.
* **Preset walidacja**: `normalize_preset` z minimalnym YAML-em i z legacy.
* **ROI/symbols**: kształt i zakres `[0,1]` (float32); `merge_masks` zgodność kształtów.
* **Graph**: `build_and_export_graph` na sztucznym `ctx.cache` (bez obrazów).

Przykładowy smoke (offline):

```python
import numpy as np
from glitchlab.core.pipeline import normalize_preset, build_ctx, apply_pipeline
from glitchlab.core.registry import available

cfg = {
  "version": 2,
  "seed": 7,
  "amplitude": {"kind":"none","strength":1.0},
  "edge_mask": {"thresh":60,"dilate":0,"ksize":3},
  "steps": [{"name": available()[0], "params": {}}]
}
img = np.zeros((96,128,3), np.uint8) + 40
ctx = build_ctx(img, seed=cfg.get("seed", 7), cfg=normalize_preset(cfg))
out = apply_pipeline(img, ctx, cfg["steps"], fail_fast=True, metrics=True)
assert out.shape == img.shape and out.dtype == np.uint8
assert any(k.startswith("stage/0/") for k in ctx.cache)
```

---

## 14) Mozaikowa meta-struktura (ontologia rdzenia)

Warstwa `core/` jest **mozaiką** współpracujących soczewek:

* **Soczewka czasu** — `pipeline`: porządkuje transformacje w liniowym DAG; każdy **etap** to komórka mozaiki procesu z metrykami `in/out`, `diff` i `t_ms`.
* **Soczewka przestrzeni** — `mosaic`: agreguje właściwości obszarów (bloki/celle) obrazu do nakładek wizualnych; współdzielony raster etykiet umożliwia porównania między różnymi widokami (metryki, AST).
* **Soczewka semantyki** — `astmap`: odwzorowuje strukturę kodu (AST) na tę samą geometrię mozaiki (np. „gorące” węzły → komórki).

To **wspólne pole współrzędnych** (czas × przestrzeń × semantyka) pozwala spinać diagnostykę: HUD może w tym samym układzie odniesienia wyświetlać **delta-metrcy** per etap, **blokowe** mapy wariancji oraz **projekcje AST**, a użytkownik ma jeden „metajęzyk” odczytu efektów.

---

## 15) Zgodność i wersjonowanie

* Schema presetów i kontrakty API oznaczone jako **v2** (utrzymywane kompatybilnie do momentu ogłoszenia v3).
* Zmiany w publicznych sygnaturach → wpis do `CHANGELOG.md` i inkrement `glitchlab.__version__`.
* Wyłącznie **aliasy rejestru** obsługują nazwy legacy; brak prywatnych mapowań w filtrach.

---

## 16) Checklist release (core)

* [ ] `registry.available()` kompletne; `meta()` zwraca poprawne `defaults/doc/aliases`.
* [ ] `pipeline` zapisuje pełny zestaw kanałów per etap + `debug/log` przy ostrzeżeniach.
* [ ] `graph` eksportuje poprawny JSON do `ast/json` (bez macierzy).
* [ ] `mosaic` działa (overlay + legenda); brak zależności zewnętrznych.
* [ ] `metrics` < 50 ms @ 1K; brak NaN/Inf.
* [ ] `utils/roi/symbols` — czyste funkcje (bez mutacji wejść).
* [ ] Smoke/fuzz presetów przechodzi lokalnie.

---

## 17) Załączniki (konkretne API)

### 17.1 `pipeline.py` — funkcje publiczne

```python
normalize_preset(cfg: Mapping[str, Any]) -> Dict[str, Any]
build_ctx(img_u8: np.ndarray, *, seed: Optional[int], cfg: Optional[Mapping[str, Any]]) -> Ctx
apply_pipeline(img_u8: np.ndarray, ctx: Ctx, steps: List[Step],
               *, fail_fast: bool = True, debug_log: Optional[List[str]] = None,
               metrics: bool = True) -> np.ndarray
```

### 17.2 `graph.py` — funkcje publiczne

```python
build_graph_from_cache(steps: List[Mapping[str, Any]], cache: Dict[str, Any], *,
                       attach_delta: bool = True) -> Graph
export_ast_json(graph: Graph, ctx_like: Optional[Mapping[str, Any]] = None, *,
                cache_key: str = "ast/json") -> Dict[str, Any]
build_and_export_graph(steps: List[Mapping[str, Any]], ctx_like: Any, *,
                       attach_delta: bool = True, cache_key: str = "ast/json") -> Dict[str, Any]
```

### 17.3 `mosaic.py` — funkcje publiczne

```python
mosaic_map(shape_hw: Tuple[int,int], *, mode: str = "square", cell_px: int = 32) -> Mosaic
mosaic_label_raster(mosaic: Mapping[str, Any]) -> np.ndarray                  # (H,W) int32
mosaic_project_blocks(block_stats: Mapping[Tuple[int,int], Mapping[str,float]], mosaic: Mosaic,
                      *, map_spec: Optional[Mapping[str, Tuple[str, Tuple[float,float]]]] = None) -> np.ndarray  # RGB u8
mosaic_overlay(img_u8: np.ndarray, overlay_u8: np.ndarray, *, alpha: float = 0.5,
               clamp: bool = True) -> np.ndarray
```

### 17.4 `metrics/basic.py`

```python
to_gray_f32(arr) -> np.ndarray
downsample_max_side(arr, max_side: int = 1024) -> np.ndarray
compute_entropy(arr, bins: int = 256) -> float
edge_density(arr) -> float
contrast_rms(arr) -> float
block_stats(arr, block: int = 16, max_side: int = 1024, bins: int = 64) -> Dict[Tuple[int,int], Dict[str,float]]
```

### 17.5 ROI i Symbole

```python
# roi.py
mask_polygon(shape_hw, points, *, feather=0) -> FloatMask
mask_rect(shape_hw, xyxy, *, feather=0) -> FloatMask
mask_circle(shape_hw, center, radius, *, feather=0) -> FloatMask
merge_masks(op: str, *masks: FloatMask) -> FloatMask

# symbols.py
bitmap_to_mask(img_u8, *, channel=0, invert=False, thresh=127) -> FloatMask
load_symbol(name: str, size_hw: Tuple[int,int]) -> FloatMask
stamp_mask(dst: FloatMask, mask: FloatMask, xy: Tuple[int,int], *, mode="max") -> FloatMask
```

---

## 18) FAQ (skrót)

* **Czy filtr może sam robić clamp/mask/amp?** — Może, ale **nie powinien**. Standaryzujemy to w wrapperze pipeline, by GUI i diagnostyka były spójne.
* **Co jeśli filtr odda `float32 [0,1]` zamiast `uint8`?** — Pipeline spróbuje bezpiecznej konwersji do `uint8`. Inne kształty/dtype → błąd.
* **Czy `core.graph` jest obowiązkowy?** — Nie. Pipeline wykona się bez niego; po prostu nie powstanie `ast/json`.

---


