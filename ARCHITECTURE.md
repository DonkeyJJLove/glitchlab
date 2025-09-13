# glitchlab/ARCHITECTURE.md

# GlitchLab v2 — System Architecture (scalony)

> **Zakres:** najwyższy poziom dokumentacji. Ten plik łączy i porządkuje kontrakty między warstwami `core/`, `analysis/`, `filters/`, `gui/`, wskazuje ontologię „mozaiki” oraz przepływy E2E.
> **Źródła szczegółów:**
> – `core/ARCHITECTURE.md` (runtime, DAG, metryki podstawowe)
> – `analysis/ARCHITECTURE.md` (metryki rozszerzone, FFT, forensyka, eksport DTO) — **wchłania dawny `analysis/analysis.md`**
> – `filters/ARCHITECTURE.md` (konwencje i wzorce filtrów)
> – `gui/ARCHITECTURE.md` (layout, HUD, panele)

---

## 0) Cel dokumentu

Zamiast wielu rozproszonych opisów, ten plik opisuje **architekturę całej aplikacji** end-to-end: od presetów i filtrów, przez rdzeń (core) i warstwę analityczną, po GUI/HUD. Służy jako **źródło prawdy** dla interfejsów, kontraktów i praktyk w projekcie.

---

## 1) Zasady projektowe (cały system)

* **Deterministyczność:** globalny seed RNG, brak ukrytej losowości.
* **Jedna sygnatura filtra:** `fn(img_u8, ctx, **params) -> np.ndarray`.
* **Diagnostyka jako produkt:** każdy krok zapisuje telemetrię do `ctx.cache` (HUD czyta po kluczach).
* **Lekkość zależności:** Python 3.9+, **NumPy**, **Pillow**. **Bez** SciPy/OpenCV.
* **Spójne nazewnictwo:** kanony nazw filtrów + aliasy, **preset v2** (jedno schema).
* **Wektoryzacja:** zero pętli per-piksel, gdy niepotrzebne.
* **Modułowość GUI:** panele per filtr + stałe sekcje HUD, **dock/float**, podglądy full-frame.
* **Stabilne klucze HUD:** komponenty GUI nie łamią się przy wymianie filtrów.
* **Budżety czasu:** metryki i diagnostyki ≤ \~50 ms @ max side 1K.

---

## 2) Warstwy i odpowiedzialności

```
glitchlab/
  core/        # registry, pipeline, graph, mosaic, astmap, metrics (basic/compare), utils, roi, symbols
  analysis/    # metrics (global/kafelkowe), diff, spectral (FFT), formats (JPEG/PNG), exporters (thin DTO)
  filters/     # filtry z @register(...) zgodne z API v2 (+ aliasy)
  presets/     # YAML v2: version/name/seed/amplitude/edge_mask/steps
  gui/         # Tkinter HUD: viewer, panele parametrów, mozaika, mini-DAG, sloty diagnostyk
```

**Rola warstw:**

* **core/** — orkiestracja: rejestr filtrów, pipeline z telemetrią, graf procesu (DAG), mozaika, AST→mozaika.
* **analysis/** — metryki i wizualizacje do HUD, FFT, forensyka formatu, eksport **thin-DTO** dla GUI.
* **filters/** — implementacje efektów (glitch/analiza), kanały diagnostyczne.
* **gui/** — podgląd i panele, **wyłącznie czyta** kanały HUD po kluczach (bez własnych obliczeń metryk).

---

## 3) Meta-struktura: „mozaika” jako rdzeń ontologiczny

**Mozaika** to wspólna siatka (square/hex), na którą rzutujemy:

* **metryki blokowe** obrazu (`analysis.metrics.block_stats` → `core.mosaic.mosaic_project_blocks`),
* **strukturę kodu/pipeline** (AST/DAG z `core.astmap`/`core.graph`).

Dzięki temu:

* HUD może pokazać **porównywalne nakładki** (obraz ↔ AST) bez interpretacji po stronie GUI,
* otrzymujemy „meta-mapę” spójności: te same komórki mozaiki niosą różne aspekty (np. entropia vs. stopień sprzężenia kroków w DAG).

**Pseudometryki meta-poziomu** (na potrzeby wnioskowania i porównań):

* `Δ_stage(i) = 1 − SSIM(out_i, in_i)` — siła artefaktu na etapie `i`.
* `Δ_AB = 1 − SSIM(A,B)` — *niekomutatywność* pary filtrów.
* `R_ring = energy_sel / energy_total` — udział energii w wybranym paśmie (pierścień/sektor).
* `L_ROI = mean(|out−in| in ROI) / mean(|out−in| outside ROI)` — lokalność efektu.

Wszystkie powyższe są **deterministyczne** i stabilne w kluczach HUD.

---

## 4) Kontrakty między warstwami

### 4.1 Filter API v2 (twarde)

```python
def my_filter(img: np.ndarray, ctx: Ctx, **params) -> np.ndarray
# Wejście: uint8 RGB (H,W,3); wnętrze: float32 [0,1]; wyjście: uint8
# Wspólne parametry: mask_key: str|None, use_amp: float|bool, clamp: bool=True
# RNG wyłącznie z ctx.rng; diagnostyki → ctx.cache["diag/<name>/..."]
```

### 4.2 Registry (odczytywane przez pipeline/GUI)

```python
@register(name: str, defaults: dict|None = None, doc: str|None = None)
get(name) -> callable
available() -> list[str]
canonical(name) -> str
alias(src, dst) -> bool
meta(name) -> {"name","defaults","doc","aliases"}
```

### 4.3 Preset schema v2

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

**Alias → kanon (skrót):**
`conture_flow|anisotropic_contour_flow → anisotropic_contour_warp`
`block_mosh → block_mosh_grid`
`spectral_shaper_lab|spectral_ring → spectral_shaper`
`perlin_grid|nosh_perlin_grid → noise_perlin_grid`

### 4.4 Ctx i kanały HUD (wspólne dla core↔analysis↔gui)

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

### 4.5 DTO dla HUD (analysis.exporters)

```json
{
  "run":   {"id":"...","seed":7,"source":{...},"versions":{...}},
  "ast":   { ... echo cache["ast/json"] ... },
  "stages":[
    {"i":0,"name":"...","t_ms":...,
     "metrics_in":{...},"metrics_out":{...},"diff_stats":{...},
     "keys":{"in":"stage/0/in","out":"stage/0/out","diff":"stage/0/diff","mosaic":"stage/0/mosaic"}}
  ],
  "format":{"notes":[...], "has_grid": true}
}
```

> Obrazy **nie** są w DTO — GUI pobiera je po kluczach z `ctx.cache`.

---

## 5) Przepływy (E2E)

### 5.1 Uruchomienie pipeline

```
load image
→ core.pipeline.normalize_preset(cfg)
→ core.pipeline.build_ctx(img, seed, cfg)
→ core.pipeline.apply_pipeline(img, ctx, cfg["steps"])
    ↳ per step: defaults ⨝ params → fn(img, ctx, **params)
    ↳ metrics/diff/fft/hist/mosaic → ctx.cache (klucze HUD)
→ core.graph.build_and_export_graph(cfg["steps"], ctx) → cache["ast/json"]
→ analysis.exporters.export_hud_bundle(ctx) → DTO
→ GUI: wczytuje DTO + obrazy po kluczach → render HUD/panele
```

### 5.2 Edycja parametrów (GUI)

* GUI pyta `registry.meta(name)["defaults"]` → generuje panel (lub panel dedykowany).
* Zmiana parametru → natychmiastowy rerun kroku lub całego pipeline (tryb zależny od GUI).
* HUD aktualizuje sloty (`stage/{i}/...`, `diag/...`).

### 5.3 Mozaika/AST

* `analysis.metrics.block_stats` → `core.mosaic.mosaic_project_blocks` → `stage/{i}/mosaic`.
* `core.astmap.ast_to_graph` → projekcja AST na mozaikę (wspólna geometria).

---

## 6) GUI — ramy (streszczenie)

* **Stały layout:** top-bar, lewy **viewer** (overlay/fullscreen), prawy **Parameters**, dół: 3 sloty **HUD**.
* **Dock/float:** panele mogą być odłączane do `Toplevel`.
* **Mini-DAG:** w HUD (lub jako pływające).
* **HUD:** wyłącznie **czyta** klucze z `ctx.cache` (brak obliczeń po stronie GUI).

Szczegóły: `gui/ARCHITECTURE.md`.

---

## 7) Analysis — tezy i protokoły (streszczenie)

> **Teza:** artefakt to sygnał diagnostyczny; sterując **gdzie** (maski) i **jak mocno** (amplitude) występuje, wnioskujemy o strukturze i relacjach między filtrami.

**Protokoły referencyjne:**

* **Komutacja (A/B):** `Δ_AB = 1 − SSIM(A,B)` — wrażliwość na kolejność.
* **ROI-scan:** porównanie intensywności artefaktu w ROI vs. tło (`L_ROI`).
* **Sweep parametrów:** progi/przejścia (kolana krzywych entropii/SSIM).
* **Seed sweep:** stabilność efektu względem RNG (mała wariancja ⇒ struktura obrazu).

Szczegóły: `analysis/ARCHITECTURE.md`.

---

## 8) Testy, walidacja i jakość

* **Preset P3 (walidator + dry-run):** sprawdza schema v2, dostępność filtrów, nieznane parametry vs. `registry.meta().defaults`; wykonuje dry-run (np. `96×128`).
* **Smoke pipeline:** wejście `zeros(128,128,3)` + losowy filtr → brak wyjątków, pełny zestaw HUD.
* **compare.py:** testy elementarne `psnr/ssim_box` (tożsamość, szum).
* **Budżety:** metryki < 50 ms @ 1K, brak NaN/Inf.
* **Stabilność GUI:** panele nie znikają; overlay full-frame działa.

**Checklist release:**

* [ ] `registry.available()` kompletne; `meta()` zawiera `defaults/doc`.
* [ ] `pipeline` zapisuje pełny zestaw kanałów per stage.
* [ ] `ast/json` zgodny z HUD, bez macierzy.
* [ ] Mozaika działa (overlay + legenda).
* [ ] Metryki mieszczą się w budżetach, brak NaN/Inf.
* [ ] GUI: dock/float, skróty, render slotów — OK.

---

## 9) Rozszerzanie / wzorce pracy

### 9.1 Nowy filtr

1. Moduł w `filters/` z `@register(...)`.
2. Obsłuż `mask_key|use_amp|clamp`, RNG z `ctx.rng`.
3. Wstaw mapy diagnostyczne do `ctx.cache["diag/<name>/..."]`.
4. (Opcjonalnie) panel w `gui/panels/panel_<name>.py`.

### 9.2 Nowy preset

1. Użyj **Prompt P1** (niżej).
2. Waliduj **Prompt P3**.
3. Migracje legacy wg **P2**.

### 9.3 Nowe metryki

* Dodaj do `core/metrics/basic.py` i/lub `analysis/metrics.py`.
* Zarejestruj w pipeline/graph, opisz w HUD.

---

## 10) Ograniczenia i decyzje technologiczne

* **Bez** SciPy/OpenCV; Pillow do resize/blur.
* Wejście/wyjście **uint8 RGB** w core.
* Parametry spoza `defaults` → ignoruj + loguj ostrzeżenie.
* Aliasowanie wyłącznie przez `core.registry.alias` (bez prywatnych map).
* Downsample do 1K w metrykach/FFT/hist (spójność i szybkość).

---

## 11) Prompt kit — **Presets**

### P1 — Nowy PRESET v2 (YAML, wzorzec)

```yaml
version: 2
name: "{PRESET_NAME}"
seed: 7
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

### P2 — Migracja legacy → v2 (reguły)

* Struktura: przepisać do v2; nazwy filtrów zmapować do kanonów.
* Parametry: `amp_px|amp_strength → use_amp`; `edgeMask → edge_mask (global)`.
* Nieznane pola usunąć; wymusić typy; usunąć per-step amplitude/edge.

### P3 — Walidacja + dry-run

* Sprawdź schema, dostępność filtrów, nieznane parametry vs. `registry.meta().defaults`.
* Dry-run na `zeros(96×128)+40`; wypisz kształt/dtype/min-max oraz listę kluczy `ctx.cache`.

---

## 12) Prompt kit — **Filters**

### #1 — Nowy filtr zgodny z v2

* Sygnatura: `def {FILTER_NAME}(img: np.ndarray, ctx: Ctx, **params) -> np.ndarray`
* Rejestr: `@register("{FILTER_NAME}", defaults=DEFAULTS, doc=DOC)`
* Wejście `uint8 RGB`; praca `float32 [0,1]`; wyjście `uint8`
* Obsłuż `mask_key|use_amp|clamp`; RNG z `ctx.rng`; diagnostyki → `ctx.cache[f"diag/{FILTER_NAME}/..."]`
* Zero pętli per-piksel; sensowne DEFAULTS + walidacja.

### #2 — Refaktor starego filtra do v2

* Zmiana sygnatury, mapowanie legacy→v2 (`amp_px→use_amp`, `edgeMask→mask_key="edge"` gdy maska dostępna), rejestracja, diagnostyki, ignorowanie nadmiarowych parametrów z logiem.

### #3 — Alias adapter + smoke

* Aliasy tylko przez `glitchlab.core.registry.alias`.
* Smoke: obraz `48×64`, kontekst `amplitude=ones`, przypadki `use_amp∈{0.0,1.0}`, `mask_key="edge"`; asercje kształtu/dtype/zakresu + krótkie metryki różnic.

---

## 13) Słownik kanałów HUD (skrót)

* `stage/{i}/in|out|diff|t_ms`
* `stage/{i}/metrics_in|metrics_out|diff_stats`
* `stage/{i}/fft_mag|hist|mosaic|mosaic_meta`
* `diag/<filter>/...`
* `ast/json`, `format/jpg_grid`, `format/notes`
* `cfg/*`, `run/id`, `run/snapshot`

---

## 14) Wersjonowanie i zmiany

* Zmiany API zapisywać w `CHANGELOG.md`; inkrement `glitchlab.__version__`.
* Etykieta zmian w warstwach: **\[core] \[analysis] \[filters] \[gui]**.
* Każda zmiana **klucza HUD** wymaga: migratora + aktualizacji w `gui/` i dokumentacji.

---

## 15) Roadmap (wysoki poziom)

1. **widgets/** w GUI: `image_canvas`, `hud`, `graph_view`, `mosaic_view`.
2. **docking** + integracja w `app.py`.
3. **state** + „Run” → pipeline + odświeżanie HUD.
4. **exporters** (thin-DTO) — zakończone w `analysis/`.
5. Panele brakujące (`depth_*`, `rgb_offset`).
6. Hotkeys + full-frame preview.
7. `main.py` (uruchamiacz).

---

## 16) Słownik (operacyjny)

* **HUD** — część GUI, która wyświetla diagnostykę z `ctx.cache`.
* **Mozaika** — wspólna siatka projekcyjna (square/hex) dla metryk obrazu i AST/DAG.
* **DAG** — graf kroków pipeline z metrykami i deltami.
* **Artefakt** — lokalna zmiana (intencjonalna) używana jako sygnał diagnostyczny.
* **Komutacja** — zamiana kolejności filtrów nie zmienia wyniku (rzadkie).
* **Próg/przejście** — wartość parametru, po której pojawia się jakościowo nowa struktura artefaktu.

---

### Załączniki i odwołania

* Szczegóły warstwy rdzeniowej: `core/ARCHITECTURE.md`
* Szczegóły metryk/FFT/forensyki/DTO: `analysis/ARCHITECTURE.md`
* Zasady i wzorce implementacji filtrów: `filters/ARCHITECTURE.md`
* Layout GUI i system paneli: `gui/ARCHITECTURE.md`


