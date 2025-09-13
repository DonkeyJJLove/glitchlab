# analysis/ARCHITECTURE.md

# GlitchLab Analysis v2 — Architektura (warstwa `analysis/`)

> **Status:** scalony dokument. Zawartość poprzedniego `analysis/analysis.md` została **wchłonięta** i uporządkowana tutaj. Ten plik jest źródłem prawdy dla API warstwy analitycznej i jej integracji z `core`/`gui`.

---

## 1) Cel, zakres i rola warstwy

Warstwa **`analysis/`** dostarcza miary, wizualizacje i eksporty **diagnostyczne** dla GlitchLab. Ujmujemy artefakt jako **sygnał informacyjny**: sterując miejscem (`maski/ROI`) i siłą (`amplitude`) wywoływania efektów, wnioskujemy o **strukturze obrazu** i **relacjach między transformacjami**.

**Założenia:**

* Obrazy 2D (RGB `uint8` wej./wyj.), wewnątrz liczmy w `float32 [0,1]`.
* Brak SciPy/OpenCV — **NumPy** (+ ewentualnie Pillow do pomocniczych konwersji).
* Determinizm (brak ukrytej losowości).
* Wydajność: metryki ≤ \~50 ms @ max side 1K (guideline).
* Telemetria ląduje w `ctx.cache` pod **stabilnymi kluczami**, które czyta HUD w GUI.

---

## 2) Inwentarz katalogu i odpowiedzialności

```
analysis/
  __init__.py
  metrics.py     # global/kafelkowe metryki, entropia/gradient/kontrast/tiles
  diff.py        # różnice wizualne i statystyczne (Δ mapy, L1/L2/SSIM proxy)
  spectral.py    # FFT 2D, maski ring/sector, widma i wizualizacje
  formats.py     # forensyka formatów (JPEG/PNG): grid 8×8, notatki
  exporters.py   # bundling DTO dla HUD/GUI (bez obrazów)
```

> **Uwaga:** Warstwa `core/metrics/` zawiera szybkie, niskopoziomowe funkcje (np. `compute_entropy`, `edge_density`). `analysis/metrics.py` może je **opakowywać** i rozszerzać (np. warianty kafelkowe), zachowując kompatybilne sygnatury. `core` może być „cienkim wrapperem” na `analysis`.

---

## 3) Kontrakt integracyjny z `core` i HUD

### 3.1 Kluczowe kanały w `ctx.cache`

* `stage/{i}/metrics_in` / `stage/{i}/metrics_out` — słownik metryk skalarów.
* `stage/{i}/diff` / `stage/{i}/diff_stats` — obraz różnicy i statystyki.
* `stage/{i}/fft_mag` — log-magnitude FFT (wizualizacja).
* `stage/{i}/hist` — histogram jasności lub kanałów.
* `stage/{i}/mosaic` / `stage/{i}/mosaic_meta` — nakładka mozaiki (square/hex).
* `diag/<filter>/...` — dowolne mapy diagnostyczne specyficzne dla filtra.
* `ast/json` — DAG/AST procesu (do widoków grafowych i mozaiki kodu).
* `format/jpg_grid`, `format/notes` — forensyka JPEG/PNG.
* `run/id`, `run/snapshot`, `cfg/*` — meta.

### 3.2 Thin-DTO dla GUI (eksporter)

```python
# analysis/exporters.py
def export_hud_bundle(ctx_like: Mapping[str, Any]) -> dict:
    """
    Zwraca lekki DTO bez obrazów. Obrazy GUI pobiera po kluczach z ctx.cache.
    Struktura:
    {
      "run":   {"id": "...", "seed": 7, "source": {...}, "versions": {...}},
      "ast":   {... echo cache["ast/json"] ...},
      "stages":[
        {"i":0,"name":"...","t_ms":..., "metrics_in":{...}, "metrics_out":{...},
         "diff_stats":{...},
         "keys":{"in":"stage/0/in","out":"stage/0/out","diff":"stage/0/diff","mosaic":"stage/0/mosaic"}}
      ],
      "format":{"notes":[...], "has_grid": true}
    }
    """
```

---

## 4) API modułów (kontrakty)

### 4.1 `analysis.metrics`

Globalne i kafelkowe metryki, gotowe do użycia przez `core.pipeline`.

```python
# Typ pomocniczy
from typing import Dict, Tuple, Mapping, Iterable
import numpy as np

def to_gray_f32(img_u8: np.ndarray) -> np.ndarray
def downsample_max_side(img_f32: np.ndarray, max_side: int = 1024) -> np.ndarray

# Global
def entropy(img_u8: np.ndarray, bins: int = 256, *, max_side: int = 1024) -> float
def edge_density(img_u8: np.ndarray, *, max_side: int = 1024) -> float
def contrast_rms(img_u8: np.ndarray, *, max_side: int = 1024) -> float

# Histogramy
def hist_luma(img_u8: np.ndarray, bins: int = 256) -> np.ndarray  # (bins,)
def hist_rgb(img_u8: np.ndarray, bins: int = 256) -> np.ndarray   # (3,bins)

# Bloki (kafle)
def block_stats(
    img_u8: np.ndarray,
    block: int = 16,
    *,
    max_side: int = 1024,
    bins: int = 64
) -> Dict[Tuple[int,int], Dict[str, float]]
"""
Zwraca mapę (bx,by) -> {"entropy":..., "edge":..., "contrast":..., "mean":...}
Zgodne z projekcją mozaiki (core.mosaic).
"""
```

**Zakresy i normy:**

* `entropy` \~ 0..\~8 (dla 8-bit), `edge_density` ∈ \[0,1], `contrast_rms` ≥ 0.
* Wszystko liczone po downsamplu do `max_side` (spójność HUD i szybkość).

---

### 4.2 `analysis.diff`

Porównania obrazów: różnice wizualne i statystyki.

```python
def l1(a_u8: np.ndarray, b_u8: np.ndarray) -> float
def l2(a_u8: np.ndarray, b_u8: np.ndarray) -> float
def psnr(a_u8: np.ndarray, b_u8: np.ndarray) -> float         # wrapper na core.metrics.compare.psnr
def ssim_box(a_u8: np.ndarray, b_u8: np.ndarray, win: int=7) -> float  # wrapper na core.metrics.compare.ssim_box

def visual_diff(a_u8: np.ndarray, b_u8: np.ndarray) -> np.ndarray
"""
Zwraca obraz Δ (uint8 RGB) do HUD (np. |a-b| wzmocnione + fałszywe kolory).
"""

def stats(a_u8: np.ndarray, b_u8: np.ndarray) -> Dict[str, float]
"""
{"l1":..., "l2":..., "psnr":..., "ssim":...}
"""
```

---

### 4.3 `analysis.spectral`

FFT 2D, maski selektywne i wizualizacje częstotliwości.

```python
def fft_mag(img_u8: np.ndarray, *, channel: str = "luma", max_side: int = 1024) -> np.ndarray
"""
Zwraca log-magnitude (float32 [0,1]) do HUD: "stage/{i}/fft_mag".
"""

def ring_mask(shape: Tuple[int,int], r0: float, r1: float) -> np.ndarray
def sector_mask(shape: Tuple[int,int], angle0: float, angle1: float, *, radians: bool = False) -> np.ndarray

def spectral_envelope(
    img_u8: np.ndarray,
    *,
    ring: Tuple[float,float] | None = None,
    sector: Tuple[float,float] | None = None,
    channel: str = "luma",
    max_side: int = 1024
) -> Dict[str, float]
"""
Zwraca {"energy_total":..., "energy_sel":..., "ratio":...} dla HUD/logów.
"""

def shaped_ifft(
    img_u8: np.ndarray,
    shaper: np.ndarray,
    *,
    channel: str = "per_channel"
) -> np.ndarray
"""
Użyteczne do wizualizacji efektu maski pasma (NIE modyfikuje oryginału w pipeline).
"""
```

---

### 4.4 `analysis.formats`

Prosta forensyka formatów i „rezonanse” siatki.

```python
def jpeg_grid_probe(img_u8: np.ndarray) -> Dict[str, object]
"""
Heurystyka dla 8×8:
- {"has_grid": bool, "period": 8, "score": float, "heatmap": np.ndarray (H,W) f32 [0,1], "notes":[...]}
Wpisywane do: cache["format/jpg_grid"], cache["format/notes"].
"""

def png_alpha_stats(img_u8: np.ndarray) -> Dict[str, float]
"""
{"alpha_ratio":..., "opaque_ratio":..., "has_premultiplied": bool?}
"""
```

---

### 4.5 `analysis.exporters`

Patrz §3.2 — agregacja metadanych i kluczy w **thin-DTO** dla GUI. Funkcja **nie** pakuje obrazów (GUI odczyta je z `ctx.cache` po kluczach).

---

## 5) Metapłaszczyzna „mozaiki” (ontologia projekcji)

**Mozaika** to wspólna siatka (square/hex), na którą rzutujemy **metryki blokowe** obrazów **i** graf AST (kod/pipeline). Warstwa `analysis` współpracuje z `core.mosaic`:

1. `analysis.metrics.block_stats(...)` → mapy (bx,by) → **projekcja**:

   * `core.mosaic.mosaic_project_blocks(block_stats, mosaic)` → `overlay_u8`.
2. `core.astmap.ast_to_graph(...)` → graf → projekcja na mozaikę (ta sama geometria).

**Zalecenie nazw metryk blokowych:**

* `"entropy"`, `"edge"`, `"contrast"`, `"mean"`, ewentualnie `"resonance_size"` (dla BMG).

HUD może wówczas pokazać **porównywalne** nakładki dla różnych etapów i „soczewek” (obraz vs AST), bez interpretacji po stronie GUI.

---

## 6) Przepływ E2E z udziałem `analysis`

```
load image
→ core.pipeline.normalize_preset(cfg)
→ core.pipeline.build_ctx(img, seed, cfg)
→ core.pipeline.apply_pipeline(img, ctx, steps)
    ↳ per step:
        - metrics_in/out via analysis.metrics.*
        - diff + stats via analysis.diff.*
        - fft/hist via analysis.spectral/metrics
        - block_stats via analysis.metrics → mosaic overlay (core.mosaic)
    ↳ wszystko do ctx.cache (klucze HUD)
→ core.graph.build_and_export_graph(...) → ctx.cache["ast/json"]
→ analysis.exporters.export_hud_bundle(ctx) → DTO dla GUI
→ GUI renderuje sloty HUD po kluczach z cache
```

---

## 7) Zestaw metod badawczych (zintegrowane z dawnym `analysis.md`)

> To „manual” wnioskowania — spójny z API i HUD. Parametryzacja przez presety v2 lub panel pojedynczego filtra.

### 7.1 Test **komutacji** (A/B)

```
A: I → F1 → F2 → I_A
B: I → F2 → F1 → I_B
Δ = 1 − SSIM(I_A, I_B)
```

**Wskazówka:** F1=`anisotropic_contour_warp`, F2=`block_mosh_grid`.
**Wniosek:** małe Δ → *prawie komutują*; duże Δ → *kolejność istotna*.

### 7.2 **ROI-scan** (lokalność/przenikalność)

* Amplitude: `kind=mask` → `mask_key=ROI_A` → zastosuj filtr F.
* Zmień na `ROI_B` i powtórz.
* Porównaj |Δ| w ROI vs poza ROI (metryki z `analysis.diff.stats`).

**Wniosek:** artefakt w ROI ⇒ filtr lokalny; „wyciek” ⇒ zależności nielokalne.

### 7.3 **Sweep parametrów** (progi/przejścia)

* 1D: `strength ∈ [0.2…2.0]`.
* 2D: siatka `(size, p)` w BMG.

Szukamy „kolan” — nagłe zmiany entropii/SSIM.

### 7.4 **Seed sweep** (losowość kontrolowana)

* Stały preset, różne `seed`.
* Licz wariancję metryk i zbieżność map HUD.

Mała wariancja ⇒ efekt wynika ze struktury obrazu.

---

## 8) Metryki (definicje operacyjne)

**Podstawowe (global):**

* **SSIM** ∈ \[0..1], **PSNR** \[dB], **L1/L2** (średnie błędy).
* **Entropy** \[bit], **edge\_density** ∈ \[0..1], **contrast\_rms** ≥ 0.

**Strukturalne (spektrum):**

* Energia w **pierścieniu**/ **sektorze** (`spectral_envelope.ratio`).

**Specyficzne filtrów (przykłady):**

* BMG: histogram/przeciętne `bmg_dx/dy`, rezonans na `size`.
* ACW: rozkład `acw_mag` vs iteracje (`iters`).

**ROI:**

* Średni |artefakt| w ROI vs poza (ratio), opcjonalnie „MI-proxy” (różnica gęstości).

---

## 9) Procedury w GUI (krok po kroku)

**Anizotropia (ACW):**

1. Open → Filter `anisotropic_contour_warp` → `strength≈1.2`, `iters=2`, `edge_bias=0.5`, `smooth≈0.7`.
2. Apply → sprawdź `acw_mag` + czytelność obiektów.
3. Zwiększ `iters`; zanotuj próg pogorszenia.

**Blokowość (BMG):**

1. `size=16/24/32`, `p≈0.5`, `max_shift≈28`, `mix≈0.9`.
2. Apply → obserwuj `bmg_dx/dy`.
3. Szukaj rezonansu przy którym „wzór skacze w oczy”.

**ROI-scan:**

1. Load mask → `Amplitude.kind=mask`.
2. Użyj filtra F z `mix≈0.7` → artefakt powinien pozostać w ROI.
3. Zmień maskę, porównaj intensywność.

**Komutacja A/B:**

1. ACW→BMG, zapisz.
2. BMG→ACW, porównaj Δ (subiektywnie / SSIM offline).

---

## 10) Wydajność i inwarianty

* **Downsample** do `max_side=1024` w metrykach/FFT/hist — spójność i szybkość.
* Praca w `float32`, konwersja do/ze `uint8` tylko na granicach.
* **Zero** pętli per-piksel; używaj `np.roll`, `reshape`, broadcast.
* Brak losowości (wszystko deterministyczne w `analysis/`).

---

## 11) Testy i walidacja (smoke)

Minimalny test dymny dla całej warstwy:

```python
import numpy as np
from analysis import metrics, diff, spectral, formats

img = np.zeros((96,128,3), np.uint8) + 40
img2 = np.copy(img); img2[16:32, 16:96] = 120

assert isinstance(metrics.entropy(img), float)
bs = metrics.block_stats(img, block=16); assert isinstance(bs, dict)

dvis = diff.visual_diff(img, img2); assert dvis.dtype == np.uint8
dst = diff.stats(img, img2); assert {"l1","l2","psnr","ssim"} <= dst.keys()

fft = spectral.fft_mag(img); assert fft.ndim == 2
env = spectral.spectral_envelope(img, ring=(0.2,0.4)); assert "ratio" in env

jf = formats.jpeg_grid_probe(img); assert "has_grid" in jf and "notes" in jf
```

---

## 12) Eksport i raportowanie (replikowalność)

**Checklista raportu:**

* obraz wejściowy (nazwa/hash),
* preset/filtr + parametry (YAML/GUI),
* seed RNG,
* zrzuty HUD (amplitude/mask + dwie diagnostyki),
* procedura (A/B, sweep, ROI),
* metryki (SSIM/PSNR/entropia/ratio spektrum),
* wnioski (progi, relacje, komutacja).

---

## 13) Ograniczenia i pułapki

* Duże obrazy ⇒ licz metryki na downsamplu; „final pass” na pełnej rozdzielczości.
* Niedopasowane maski ⇒ artefakty na krawędziach (przeskaluj precyzyjnie).
* Zbyt małe `mix/p/strength` lub `amplitude≈0` ⇒ „brak efektu” (fałszywa diagnoza).
* HUD to diagnostyka **wizualna** — do liczb używaj funkcji `analysis.*` lub eksportów.

---

## 14) Przykładowe receptury (YAML v2)

**Anizotropia + lekki blokowy probe**

```yaml
edge_mask: { thresh: 60, dilate: 6, ksize: 3 }
amplitude: { kind: perlin, scale: 96, octaves: 4, strength: 1.0 }
steps:
  - { name: anisotropic_contour_warp, params: { strength: 1.2, iters: 2, edge_bias: 0.5, smooth: 0.7 } }
  - { name: block_mosh_grid,        params: { size: 24, p: 0.35, max_shift: 24, mix: 0.75 } }
```

**ROI-scan (celowane zaburzanie)**

```yaml
amplitude: { kind: mask, strength: 1.0, mask_key: text_roi }
steps:
  - { name: anisotropic_contour_warp, params: { strength: 1.0, iters: 2, use_amp: true } }
```

**Komutacja A/B**

```yaml
# Wariant A
steps:
  - { name: anisotropic_contour_warp, params: { strength: 1.1, iters: 2 } }
  - { name: block_mosh_grid,         params: { size: 24, p: 0.4, max_shift: 28, mix: 0.85 } }
# Wariant B = zamiana kolejności
```

---

## 15) Tabela decyzji (ściąga)

| pytanie                     | użyj                        | obserwacja/metryka       |   |                   |
| --------------------------- | --------------------------- | ------------------------ | - | ----------------- |
| anizotropia treści?         | `anisotropic_contour_warp`  | czytelność, `acw_mag`    |   |                   |
| ślad kompresji / blokowość? | `block_mosh_grid`           | rezonans `size`, `dx/dy` |   |                   |
| lokalność efektu?           | Amplitude `kind=mask` + ROI |                          | Δ | w ROI vs poza ROI |
| istotność kolejności?       | test A/B                    | Δ = 1−SSIM(A,B) / L1     |   |                   |
| dominujące pasma/kierunki?  | `spectral_envelope` / `fft` | `ratio` / widmo          |   |                   |

---

## 16) Zgodność i wersjonowanie

* Stabilne nazwy kluczy HUD i pól w DTO (współdzielone z `core`/`gui`).
* Zmiany w API → opis w `CHANGELOG.md` i aktualizacja dokumentacji.
* Brak zależności zewnętrznych poza NumPy/Pillow (opcjonalnie YAML u góry GUI, nie tutaj).

---

## 17) Słownik

* **anizotropia** — własność zależna od kierunku (stabilność „wzdłuż” konturów).
* **blokowość** — regularność siatki (kompresja, skalowanie).
* **komutacja** — zamiana kolejności filtrów nie zmienia wyniku.
* **próg/przejście** — punkt parametru z jakościową zmianą artefaktu.
* **mozaika** — wspólna siatka projekcyjna metryk (obraz/AST) do nakładek HUD.

---

## 18) Załącznik: szkice implementacyjne

**FFT log-magnitude (luma):**

```python
def fft_mag(img_u8, channel="luma", max_side=1024):
    import numpy as np
    x = img_u8.astype(np.float32) / 255.0
    if channel == "luma":
        x = (0.2126*x[...,0] + 0.7152*x[...,1] + 0.0722*x[...,2]).astype(np.float32)
    H,W = x.shape[:2]
    s = max(H,W)
    if s > max_side:
        k = max_side / s
        H2, W2 = int(H*k), int(W*k)
        x = np.stack([np.interp(np.linspace(0,H-1,H2), np.arange(H), x[:,j]) for j in range(W)], axis=1)
        x = np.stack([np.interp(np.linspace(0,W-1,W2), np.arange(W), x[i]) for i in range(H2)], axis=0)
    f = np.fft.fftshift(np.fft.fft2(x))
    m = np.log1p(np.abs(f))
    m /= (m.max() + 1e-6)
    return m.astype(np.float32)
```

**Maska pierścienia:**

```python
def ring_mask(shape, r0, r1):
    import numpy as np
    H,W = shape
    yy,xx = np.mgrid[:H,:W]
    cy, cx = (H-1)/2.0, (W-1)/2.0
    rr = np.sqrt((yy-cy)**2 + (xx-cx)**2) / np.sqrt(cy**2 + cx**2)
    return ((rr >= r0) & (rr <= r1)).astype(np.float32)
```

> Kod poglądowy — właściwe implementacje w repo trzymają się konwencji dtype/kształtów używanych w `core`.

---

