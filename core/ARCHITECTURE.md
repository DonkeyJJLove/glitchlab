# glitchlab — controlled glitch for analysis

![Interfejs](backup/screen.png)

GlitchLab to narzędzie do **kontrolowanej generacji artefaktów** w obrazach (2D), projektowane pod **analizę** i **wnioskowanie**. Błąd traktujemy jako **sygnał diagnostyczny**: poprzez maski (ROI), pola amplitudy (siła lokalna) i deterministyczne filtry można **wywołać** i **ukształtować** artefakty tak, by ujawniały informacje o strukturze danych i relacjach między transformacjami.

---

## Co nowego w v2

* **Spójne API filtrów:** `fn(img_u8, ctx, **params) -> np.ndarray` + wspólne parametry `mask_key | use_amp | clamp`.
* **Jedno schema presetów (v2):** `version/name/amplitude/edge_mask/steps` + walidator i dry-run.
* **Pipeline z metrykami:** metryki wejścia/wyjścia, diff, czasy, telemetria do `ctx.cache` (HUD).
* **DAG procesu:** lekki graf (nodes/edges/delta) eksportowany jako JSON dla GUI.
* **Mozaika/AST:** wspólna „soczewka” wizualizacji metryk blokowych i struktury kodu (AST→mosaic).
* **GUI/HUD v2:** stały layout, panele parametrów nie „znikają”, powiększany podgląd, pływające panele.

---

## Instalacja

Wymagania: Python 3.9+, NumPy, Pillow, Tkinter (systemowe).

```bash
git clone https://github.com/you/glitchlab
cd glitchlab
pip install -r requirements.txt
```

---

## Uruchomienie (GUI)

```bash
python -m glitchlab.gui.main
```

**Workflow (skrót):**

1. **Open image…** (PNG/JPG/WEBP)
2. Ustaw **Amplitude & Edge** (prawy panel)
3. (Opcjonalnie) **Load mask…** (ROI) — maska od razu w HUD
4. Wybierz **Preset** lub **Filter** → **Apply**
5. Odczytaj mapy w **Filter Diagnostics** i iteruj parametry
6. **Save result…**

---

## Po co to (analitycznie)?

Artefakty są wytwarzane **intencjonalnie** i **lokalnie**. Dzięki temu możesz:

* **weryfikować hipotezy** o strukturze (anizotropia, kontury),
* **wykrywać rezonanse** (siatka bloków, ślady kompresji/skalowania),
* **badać relacje filtrów** (test komutacji A/B, czułość kolejności).

Więcej w **ANALYSIS.md** (protokół A/B, sweepy, SSIM/PSNR/entropia).

---

## Struktura projektu

```
glitchlab/
  core/        # registry, pipeline, graph, mosaic, astmap, metrics, utils, roi, symbols
  analysis/    # metrics, diff, spectral, formats, exporters (warstwa badawcza)
  filters/     # filtry rejestrowane dekoratorem (API v2)
  presets/     # YAML v2 (schemat poniżej)
  gui/         # aplikacja i panele (HUD, graf, mozaika, parametry)
```

---

## Architektura (skrót)

* **Core**: rejestr filtrów, pipeline z metrykami/diff i DAG, mozaika/AST, narzędzia.
* **Analysis**: metryki globalne/kafelkowe, FFT/histogram, forensyka formatu, bundling DTO.
* **GUI**: stały układ (viewer + parameters + 3 sloty diagnostyk), panele per filtr, pływające/dokowane okna, mini-graf.

---

## Standardy API

### Filtr (v2)

```python
def my_filter(img: np.ndarray, ctx: Ctx, **params) -> np.ndarray
# Wejście: uint8 RGB (H,W,3); praca: float32 [0,1]; zwrot: uint8
# Wspólne parametry: mask_key: str|None, use_amp: float|bool, clamp: bool=True
# RNG: wyłącznie ctx.rng; diagnostyki → ctx.cache[f"diag/<name>/..."]
```

### Rejestr

```python
@register(name: str, defaults: dict|None = None, doc: str|None = None)
get(name) -> callable
available() -> list[str]
canonical(name) -> str
alias(src, dst) -> bool
meta(name) -> {"name","defaults","doc","aliases"}
```

### Kontekst (`Ctx`) i HUD-channels

```python
@dataclass
class Ctx:
    rng: np.random.Generator                # deterministyczny seed
    amplitude: np.ndarray                   # (H,W) f32 [0..1]
    masks: Dict[str, np.ndarray]            # maski (H,W) f32 [0..1]
    cache: Dict[str, Any]                   # telemetria dla HUD
    meta: Dict[str, Any]                    # {source, versions, ...}
```

**Kanały min.:**
`stage/{i}/in|out|diff|t_ms`,
`stage/{i}/metrics_in|metrics_out|diff_stats`,
`stage/{i}/fft_mag|hist|mosaic|mosaic_meta`,
`diag/<filter>/...`, `ast/json`, `format/jpg_grid`, `format/notes`, `cfg/*`, `run/id`, `run/snapshot`.

---

## Presety (YAML v2)

```yaml
version: 2
name: "example"
seed: 7
amplitude:
  kind: perlin        # none|linear_x|linear_y|radial|perlin|mask
  strength: 1.0
  scale: 96
  octaves: 4
  persistence: 0.5
  lacunarity: 2.0
edge_mask:
  thresh: 60
  dilate: 0
  ksize: 3
steps:
  - name: anisotropic_contour_warp
    params: { strength: 1.2, iters: 2, edge_bias: 0.5, use_amp: 1.0, clamp: true }
  - name: block_mosh_grid
    params: { size: 24, p: 0.45, max_shift: 32, mix: 0.9 }
```

**Alias → kanon (skrót):**
`conture_flow|anisotropic_contour_flow → anisotropic_contour_warp`
`block_mosh → block_mosh_grid`
`spectral_shaper_lab|spectral_ring → spectral_shaper`
`perlin_grid|nosh_perlin_grid → noise_perlin_grid`

---

## Filtry referencyjne (skrót merytoryczny)

### `anisotropic_contour_warp`

* **Cel:** przemieszcza piksele **wzdłuż konturów** (tangent do ∇I).
* **Użycie:** test **anizotropii** (stabilność semantyki na tangencie).
* **Parametry:** `strength`, `iters`, `ksize`, `smooth`, `edge_bias`, `mask_key`, `use_amp`.
* **Diag:** `acw_mag` (|∇I|), `acw_tx/ty` (tangenty).

### `block_mosh_grid`

* **Cel:** przestawianie/rotacja bloków (siatka).
* **Użycie:** **skala i blokowość** (rezonans rozmiaru bloku).
* **Parametry:** `size`, `p`, `max_shift`, `mode`, `wrap`, `mask_key`, `amp_influence`, `channel_jitter`, `posterize_bits`, `mix`.
* **Diag:** `bmg_select`, `bmg_dx/dy`.

---

## Szybki smoke-test (offline)

```python
import numpy as np
from glitchlab.core.pipeline import normalize_preset, build_ctx, apply_pipeline
from glitchlab.core.registry import available

cfg = {
  "version": 2,
  "seed": 7,
  "amplitude": {"kind":"none","strength":1.0},
  "edge_mask": {"thresh":60,"dilate":0,"ksize":3},
  "steps": [{"name": available()[0], "params": {}}]  # pierwszy dostępny filtr
}
cfg = normalize_preset(cfg)

img = np.zeros((96,128,3), np.uint8) + 40
ctx = build_ctx(img, seed=cfg.get("seed", 7), cfg=cfg)
out = apply_pipeline(img, ctx, cfg["steps"], fail_fast=True, metrics=True)

print(out.shape, out.dtype)         # (96,128,3) uint8
print(sorted(k for k in ctx.cache)) # sprawdź kanały HUD
```

---

## Rozszerzanie

### Nowy filtr

```python
from glitchlab.core.registry import register
import numpy as np

DEFAULTS = {"strength":1.0, "mask_key":None, "use_amp":1.0, "clamp":True}
DOC = "Przykładowy filtr v2; pracuje w f32 [0..1], zwraca u8."

@register("my_filter", defaults=DEFAULTS, doc=DOC)
def my_filter(img: np.ndarray, ctx, **p) -> np.ndarray:
    # wewnątrz f32 [0,1]
    x = img.astype(np.float32) / 255.0
    strength = float(p.get("strength", 1.0))
    amp = float(p.get("use_amp", 1.0))
    mkey = p.get("mask_key")
    m = None
    if mkey and mkey in ctx.masks:
        m = ctx.masks[mkey].astype(np.float32)
        if m.shape != x.shape[:2]:
            # tu zwykle resize do (H,W)
            m = np.clip(m, 0, 1)
    # przykładowy efekt: lekkie rozjaśnienie
    eff = np.clip(x * (1.0 + strength*0.2*amp), 0, 1)
    if m is not None:
        eff = (1 - m[...,None]) * x + m[...,None] * eff
    out = np.clip(eff, 0, 1)
    ctx.cache["diag/my_filter/amp"] = amp
    if m is not None:
        ctx.cache["diag/my_filter/mask"] = m
    return (out * 255.0 + 0.5).astype(np.uint8)
```

### Nowy preset

Skorzystaj z **Prompt P1** (patrz **SYSTEM\_ARCHITECTURE.md** / **ARCHITECTURE.md**), a następnie zweryfikuj **Prompt P3** (walidacja + dry-run).

### Panele GUI

* Dodaj panel do `glitchlab/gui/panels/panel_<filter>.py`.
* Pobierz `defaults/doc` z `registry.meta(name)`.
* Emituj parametry jako dict → GUI odpala rerun; mapy diagnostyczne pokażą się automatycznie (klucze `diag/<filter>/...`).

---

## Maski i amplitude (praktyka)

* **Maski**: grayscale (0..255) → \[0..1]; wczytywane w GUI; dostępne jako `ctx.masks["<key>"]`.
* **Amplitude**: `linear_x/y`, `radial`, `perlin`, `mask`; zwykle mnoży siłę efektu (`use_amp`).
* Dla stabilności zalecana baza **>0** (np. `0.25 + 0.75*A`), by uniknąć „dziur”.

---

## Rozwiązywanie problemów

* **„Unknown filter '…'”** — upewnij się, że moduł filtra jest jawnie importowany w `filters/__init__.py` i/lub alias istnieje w rejestrze.
* **„Maska nie w HUD”** — sprawdź rozmiar maski (musi równać się obrazowi) i klucz w `ctx.masks`.
* **„Brak efektu”** — podnieś `strength/p/mix` lub `use_amp`; sprawdź mapy diagnostyczne.

---

## Licencja i autorzy

Open Source — D2J3 aka Cha0s (for test and fun)

---

## Dalsza lektura
* [**ARCHITECTURE**](ARCHITECTURE.md)
* [**GUI ARCHITECTURE**](gui/ARCHITECTURE.md)
* [**CORE ARCHITECTURE**](core/ARCHITECTURE.md)
* [**ANALYSIS**](analysis/analysis.md)

---

To wszystko — gotowe do wklejenia jako README.md. Po tym przechodzimy do **systemu GUI**: rozpiszę skeleton stałego layoutu, kontrakty paneli, loader paneli, oraz widżety `graph_view` i `mosaic_view` (z kluczami HUD już obsłużonymi w core).
