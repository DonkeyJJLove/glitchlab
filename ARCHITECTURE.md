# GlitchLab v2 — System Architecture

## 0) Po co ten dokument

Zamiast wielu rozproszonych opisów, ten plik opisuje **architekturę całej aplikacji** end-to-end: od presetów i filtrów, przez rdzeń (core) i analizę, po GUI/HUD. Służy jako **źródło prawdy** dla interfejsów i praktyk w projekcie.

---

## 1) Zasady projektowe (cały system)

* **Deterministyczność:** seed RNG, brak ukrytych źródeł losowości.
* **Jedna sygnatura filtra:** `fn(img_u8, ctx, **params) -> np.ndarray`.
* **Diagnostyka jako produkt:** każdy krok odkłada dane do `ctx.cache` (HUD je czyta).
* **Lekkość zależności:** Python 3.9+, NumPy, Pillow. Bez SciPy/OpenCV.
* **Spójne nazewnictwo:** kanoniczne nazwy filtrów, preset v2 (jedno schema).
* **Wektoryzacja:** brak pętli per-piksel, gdy niepotrzebne.
* **Modułowość GUI:** panele per filtr + stałe sekcje HUD, dokowanie, powiększenia.

---

## 2) Warstwy i ich role

### 2.1 Core (`glitchlab/core/`)

* **registry.py** — rejestr filtrów, aliasy, defaults/doc.
* **pipeline.py** — normalizacja presetów v2, budowa `Ctx`, uruchamianie kroków, metryki, diff, logi.
* **graph.py** — DAG procesu z deltami metryk i eksportem JSON do HUD.
* **mosaic.py** — siatki square/hex, projekcja metryk na overlay.
* **astmap.py** — Python AST → graf semantyczny → projekcja na mozaikę.
* **metrics/** — `basic.py` (entropy/edges/contrast/bloki), `compare.py` (PSNR/SSIM box).
* **utils.py / roi.py / symbols.py** — konwersje, maski ROI, symbole→maski.
* *(opcjonalnie)* **artifact.py / operator.py** — runtime operatórów (node-style), równoległy do klasycznego registry.

### 2.2 Analysis (`glitchlab/analysis/`)

* Implementacje metryk i wizualizacji wykorzystywane przez core/HUD:

  * **metrics.py** — globalne i kafelkowe metryki.
  * **diff.py** — różnice wizualne + statystyki.
  * **spectral.py** — FFT, ring/sector, histogram.
  * **formats.py** — JPEG/PNG forensics (grid 8×8, noty).
  * **exporters.py** — bundling telemetrii dla HUD (DTO bez obrazów).

> Core może być „cienkim wrapperem” na część analysis, by nie tworzyć cykli zależności. Interfejsy muszą być identyczne (sygnatury, zakresy wartości).

### 2.3 Filters (`glitchlab/filters/`)

* Moduły z dekoratorem `@register(...)`.
* Wspólne parametry: `mask_key`, `use_amp`, `clamp` (obsłużone w filtrze **lub** na wejściu/wyjściu w pipeline).
* Każdy filtr odkłada co najmniej 1–2 mapy diagnostyczne do `ctx.cache["diag/<name>/..."]`.

### 2.4 Presets (`glitchlab/presets/`)

* YAML v2: `version/name/amplitude/edge_mask/steps`.
* Migrator legacy → v2 (P2) i walidator/dry-run (P3).

### 2.5 GUI (`glitchlab/gui/`)

* **Stały layout:** top-bar, lewy podgląd (z overlay/fullscreen), prawy „Parameters”, dół: 3 duże sloty diagnostyk.
* **Docking/pływające panele:** per filtr + stałe panele (Amplitude, Edge, Mosaic, Antifragility).
* **HUD:** czyta `ctx.cache` (obrazy po kluczach, DTO bez obrazów).
* **Mini-graf procesu** (DAG) + zakładka „Mozaika/AST”.

---

## 3) Kontrakty między warstwami

### 3.1 Filter API v2 (twarde)

```python
def my_filter(img: np.ndarray, ctx: Ctx, **params) -> np.ndarray
# Wejście: uint8 RGB (H,W,3); Wnętrze: float32 [0,1]; Wyjście: uint8
# Wspólne parametry: mask_key: str|None, use_amp: float|bool, clamp: bool=True
# RNG wyłącznie z ctx.rng; diagnostyki do ctx.cache["diag/<name>/..."]
```

### 3.2 Registry (odczytywane przez pipeline/GUI)

```python
@register(name: str, defaults: dict|None = None, doc: str|None = None)
get(name) -> callable
available() -> list[str]
canonical(name) -> str
alias(src, dst) -> bool
meta(name) -> {"name","defaults","doc","aliases"}
```

### 3.3 Preset schema v2

```yaml
version: 2
name: "<NAME>"
seed: 7
amplitude: { kind: none|linear_x|linear_y|radial|perlin|mask, strength: 1.0, ... }
edge_mask: { thresh: 60, dilate: 0, ksize: 3 }
steps:
  - name: <canonical_filter_name>
    params: { ... only filter params ... }
```

### 3.4 Ctx i kanały HUD (wspólne dla core↔GUI↔analysis)

```python
@dataclass
class Ctx:
    rng: np.random.Generator
    amplitude: np.ndarray      # (H,W) f32 [0,1]
    masks: Dict[str, np.ndarray]
    cache: Dict[str, Any]      # obrazy i metadane dla HUD
    meta: Dict[str, Any]       # {source, versions, ...}
```

**Kanały standardowe (min.):**

* `stage/{i}/in|out|diff`, `stage/{i}/t_ms`
* `stage/{i}/metrics_in|metrics_out|diff_stats`
* `stage/{i}/fft_mag`, `stage/{i}/hist`
* `stage/{i}/mosaic`, `stage/{i}/mosaic_meta`
* `diag/<filter>/...` (dowolne mapy wewnętrzne)
* `ast/json`, `format/jpg_grid`, `format/notes`, `cfg/*`, `run/id`, `run/snapshot`

### 3.5 DTO dla HUD (`export_hud_bundle(ctx)`)

```json
{
  "run":   {"id": "...", "seed": 7, "source": {...}, "versions": {...}},
  "ast":   { ... contents of cache["ast/json"] ... },
  "stages":[{"i":0,"name":"...","t_ms":..., "metrics_in":{...},"metrics_out":{...},"diff_stats":{...},
             "keys":{"in":"stage/0/in","out":"stage/0/out","diff":"stage/0/diff","mosaic":"stage/0/mosaic"}}],
  "format":{"notes":[...], "has_grid": true }
}
```

> Obrazy nie są w DTO — GUI pobiera je po kluczach z `ctx.cache`.

---

## 4) Przepływy (E2E)

### 4.1 Uruchomienie pipeline

```
load image → normalize_preset(cfg)
→ build_ctx(img, seed, cfg)
→ apply_pipeline(img, ctx, cfg["steps"])
    ↳ per step: defaults ⨝ params → fn(img, ctx, **params)
    ↳ metryki in/out + diff + miniatury → ctx.cache
→ build_and_export_graph(cfg["steps"], ctx) → ctx.cache["ast/json"]
→ GUI: read bundle + obrazy po kluczach → render HUD/panele
```

### 4.2 Edycja parametrów w GUI

* GUI pyta `registry.meta(name)["defaults"]` → generuje panel.
* Zmiana parametru → natychmiastowy rerun kroku lub całego pipeline (wg trybu).
* HUD aktualizuje sloty diagnostyczne (`stage/{i}/...`, `diag/...`).

### 4.3 Mozaika/AST

* `analysis.metrics.block_stats` → `core.mosaic.mosaic_project_blocks` → overlay do `stage/{i}/mosaic`.
* `core.astmap.ast_to_graph` → projekcja na mozaikę (równy format nakładki).

---

## 5) GUI — zasady i layout

* **Stały layout** (bez „znikających” przycisków):

  * Top-bar: Open/Save, Preset, Filter, Seed, tryby HUD.
  * Left: duży viewer (toggle overlay/fullscreen).
  * Right: sekcje **Filter info**, **Parameters**, **Amplitude/Edge/Mosaic**.
  * Bottom: 3 sloty diagnostyczne (Masks\&Amplitude, Filter Diagnostics, Graph\&Log).
* **Pływające panele / dokowanie:** dowolne panele filtrów + narzędzi (Antifragility, Mosaic/AST).
* **Antykruchość:** szybkie perturbacje i wykres delty metryk.
* **Stabilne ID kluczy** w `ctx.cache` → panele nie „łamie” wymiana filtrów.

---

## 6) Testy, walidacja i jakość

* **Preset P3** — walidator schema v2 + dry-run smoke (bez I/O).
* **Smoke pipeline** — `zeros(128,128,3)` + losowy filtr → brak wyjątków, metryki w cache.
* **compare.py** — `psnr/ssim_box` testy elementarne (tożsamość, szum).

**Checklist release:**

* [ ] `registry.available()` kompletne; `meta()` zawiera defaults/doc.
* [ ] `pipeline` zapisuje pełen zestaw kanałów per stage.
* [ ] DAG json (`ast/json`) zgodny i bez macierzy.
* [ ] Mozaika działa (overlay i legenda).
* [ ] Metryki <50 ms @ 1K; brak NaN/Inf.
* [ ] GUI: panele nie znikają; overlay full-frame działa.

---

## 7) Rozszerzanie / wzorce pracy

### 7.1 Nowy filtr

1. Stwórz moduł w `filters/` z `@register(...)`.
2. Dodaj mapy diagnostyczne do `ctx.cache["diag/<name>/..."]`.
3. (Jeśli potrzeba) panel w `gui/panels/panel_<name>.py`.

### 7.2 Nowy preset

1. Użyj **Prompt P1** (poniżej).
2. Waliduj **Prompt P3**.
3. Jeśli legacy, migruj **Prompt P2**.

### 7.3 Nowe metryki

* Dodaj do `core/metrics/basic.py` i (opcjonalnie) `analysis/metrics.py`.
* Rejestr w pipeline/graph, opis do HUD.

---

## 8) Ograniczenia i decyzje technologiczne

* Brak SciPy/OpenCV; Pillow do resize/blur.
* Obowiązkowe `uint8 RGB` na wejściu/wyjściu core.
* Tolerancja nadmiarowych parametrów (ignoruj + loguj).
* Aliasowanie tylko API rejestru (bez prywatnych słowników).

---

## 9) Appendix A — Prompt kit (Presets)

### P1 — Nowy PRESET v2 (YAML)

Wymagania i wzorzec dokładnie jak poniżej — zgodny z registry + GUI:

```yaml
version: 2
name: "{PRESET_NAME}"
amplitude:
  kind: {none|linear_x|linear_y|radial|perlin|mask}
  strength: 1.0
  # perlin: {scale:96, octaves:4, persistence:0.5, lacunarity:2.0}
  # mask:   {mask_key:"<ctx.masks key>"}
edge_mask: {thresh:60, dilate:0, ksize:3}
steps:
  - name: <KANONICZNA_NAZWA_FILTRA>
    params: { ... }
```

**Mapowanie aliasów (kanon):**
`conture_flow|anisotropic_contour_flow→anisotropic_contour_warp`,
`block_mosh→block_mosh_grid`,
`spectral_shaper_lab|spectral_ring→spectral_shaper`,
`perlin_grid|nosh_perlin_grid→noise_perlin_grid`.

### P2 — Migracja legacy → v2

Zasady: przepisz strukturę, zmapuj nazwy filtrów i parametry (`amp_px|amp_strength→use_amp`, `edgeMask→edge_mask→global`), usuń nieznane, wymuś typy, odetnij per-step amplitude/edge do globalnych.

### P3 — Walidacja + dry-run

Skrypt: wczytuje YAML inline, sprawdza schema/availability filtrów/nieznane parametry vs `registry.meta(name)["defaults"]`, uruchamia dry-run na `zeros(96×128)+40`, wypisuje kształt/dtype/min-max oraz listę kluczy `ctx.cache`.

---

## 10) Appendix B — Prompt kit (Filters)

### #1 — Nowy filtr zgodny z v2

* Sygnatura: `def {FILTER_NAME}(img: np.ndarray, ctx: Ctx, **params) -> np.ndarray`
* Rejestracja: `@register("{FILTER_NAME}", defaults=DEFAULTS, doc=DOC)`
* Wejście `uint8 RGB`; praca `float32 [0,1]`; zwrot `uint8`.
* Obsłuż `mask_key|use_amp|clamp`, RNG z `ctx.rng`; diagnostyki do `ctx.cache[f"diag/{FILTER_NAME}/..."]`.
* Zero pętli per-piksel; sensowne DEFAULTS + walidacja.

### #2 — Refaktor starego filtra do v2

* Zmień sygnaturę, wykonaj mapowanie legacy→v2 (w tym `amp_px→use_amp`, `edgeMask→mask_key="edge"` gdy dostępna maska), rejestruj, obsłuż wspólne parametry, zapisuj diagnostyki, ignoruj nadmiarowe parametry z logiem ostrzeżeń.

### #3 — Alias adapter + testy „smoke”

* Aliasy tylko przez `glitchlab.core.registry.alias`.
* Smoke: obraz `48×64`, kontekst z `amplitude=ones`, przypadki `use_amp=1.0`, `use_amp=0.0`, `mask_key="edge"`; asercje kształtu/dtype/zakresu, krótkie metryki różnic.

---

## 11) Appendix C — Słownik kanałów HUD (skrót)

* `stage/{i}/in|out|diff|t_ms`
* `stage/{i}/metrics_in|metrics_out|diff_stats`
* `stage/{i}/fft_mag|hist|mosaic|mosaic_meta`
* `diag/<filter>/...`
* `ast/json`, `format/jpg_grid`, `format/notes`, `cfg/*`, `run/id`, `run/snapshot`

---

## 12) Co utrzymujemy „na bieżąco”

* Zmiany w filtrach → aktualizacja `defaults/doc` w rejestrze.
* Nowe kanały diagnostyczne → wpis do sekcji 11 (HUD).
* Zmiany schema presetów → sekcja 3.3 + Appendix A.
* GUI: dodanie nowego panelu → rejestr w loaderze + opis w sekcji 5.

---


