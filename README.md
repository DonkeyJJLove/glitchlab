# glitchlab — controlled glitch for analysis

![Interfejs](screen.png)

glitchlab to narzędzie do **kontrolowanej generacji artefaktów** w obrazach (2D), projektowane pod **analizę** i **wnioskowanie**. Błąd traktujemy jako **sygnał diagnostyczny**: poprzez maski (ROI), pola amplitudy (siła lokalna) i deterministyczne filtry można **wywołać** i **ukształtować** artefakty tak, by ujawniały informacje o strukturze danych i relacjach między transformacjami.

---

## Najważniejsze
- **Błąd jako sonda:** artefakt nie jest „szumem” — sterowany lokalnie staje się **narzędziem pomiarowym** (np. test anizotropii, detekcja blokowości).
- **Kontrola lokalna:** maski (ROI) + amplitude (pole [0..1]) → decydujesz **gdzie** i **jak mocno** filtr działa.
- **Replikowalność:** seed RNG, YAML presety, diagnostyka w HUD (mapy w `ctx.cache`).
- **Praktyka > estetyka:** GUI do szybkiego strojenia; filtry i presety projektowane pod **stabilne, mierzalne** efekty.

---

## Instalacja
Wymagania: Python 3.9+, NumPy, Pillow, Tkinter (w standardowej dystrybucji na Windows/Linux/macOS).

```bash
git clone https://github.com/you/glitchlab
cd glitchlab
pip install -r requirements.txt
````

---

## Uruchomienie (GUI)

```bash
python -m glitchlab.gui.main
```

**Workflow (skrót):**

1. **Open image…** (PNG/JPG/WEBP)
2. Ustaw **Amplitude & Edge** (panel po prawej)
3. (Opcjonalnie) **Load mask…** (ROI) — maska od razu jest widoczna w HUD
4. Wybierz **Preset** lub **Filter** → **Apply**
5. Odczytaj mapy w **Filter Diagnostics** (dół, środek) i iteruj parametry
6. **Save result…**

---

## Po co to (analitycznie)?

Artefakty są wytwarzane **intencjonalnie** i **lokalnie**. Gdy zachowujesz kontrolę:

* możesz **potwierdzić hipotezę** o strukturze obrazu (np. czy semantyka „leży” wzdłuż konturów),
* możesz **wykryć rezonanse** z siatką bloków (ślady kompresji/skalowania),
* możesz **zbadać relacje** między filtrami (czy kolejność ma znaczenie — test komutacji A/B).

> Szczegółowa metodologia (protokół A/B, sweepy, metryki) jest w osobnym pliku: **ANALYSIS.md**.

---

## Interfejs (HUD)

* **Masks & Amplitude** — po lewej na dole: aktualna maska (priorytet: `Amplitude.kind=mask → mask_key`, w przeciwnym razie `edge` lub pierwsza dostępna) oraz pole amplitudy.
* **Filter Diagnostics** — dwie mapy z `ctx.cache` wystawiane przez filtr (np. gradient, wektory przesunięć).
* **Logs** — kroki i parametry pipeline’u.

---

## Filtry referencyjne (skrót merytoryczny)

### `anisotropic_contour_warp`

* **Cel:** przemieszcza piksele **wzdłuż konturów** (tangent do ∇I).
* **Użycie analityczne:** test **anizotropii** — jeśli po kilku iteracjach semantyka (np. tekst) pozostaje czytelna, struktura jest stabilna tangencjalnie.
* **Parametry:** `strength`, `iters`, `ksize`, `smooth`, `edge_bias`, `mask_key`, `use_amp`.
* **Diagnostyka:** `acw_mag` (|∇I|), `acw_tx/ty` (tangenty).

### `block_mosh_grid`

* **Cel:** przesuwa/rotuje bloki w siatce; tryby `shift`/`swap`.
* **Użycie analityczne:** ujawnia **skalę i blokowość** (rezonans na rozmiarze bloku = ślad kompresji/skalowania).
* **Parametry:** `size`, `p`, `max_shift`, `mode`, `wrap`, `mask_key`, `amp_influence`, `channel_jitter`, `posterize_bits`, `mix`.
* **Diagnostyka:** `bmg_select` (wybór bloków), `bmg_dx/dy` (przesunięcia).

---

## Presety (YAML)

Presety łączą kroki filtrów z konfiguracją `edge_mask` i `amplitude`.

```yaml
# glitchlab/presets/depth_parallax.yaml (przykład)
edge_mask: { thresh: 60, dilate: 6, ksize: 3 }
amplitude: { kind: perlin, scale: 120, octaves: 4, strength: 1.2 }

steps:
  - { name: anisotropic_contour_warp, params: { strength: 1.2, iters: 2, edge_bias: 0.5 } }
  - { name: block_mosh_grid,        params: { size: 24, p: 0.45, max_shift: 32, mix: 0.9 } }
```

**Wskazówki:**

* `p` i `mix` domykają „siłę wizualną”; do analizy nie przesadzaj (łatwo zamaskować wzorce).
* `amplitude.kind=mask` + `mask_key` → **celowane** zaburzanie (ROI-scan).
* Dla testów porównawczych trzymaj **seed** stały.

---

## Maski i amplitude (praktyka)

* **Maski**: grayscale (0..255) → \[0..1]; wczytasz przez **Load mask…**; klucz maski pojawi się w dropdownie amplitude (`kind=mask`).
* **Amplitude**: `linear_x/y`, `radial`, `perlin`, `mask`; w filtrach często mnoży siłę (`use_amp`, `amp_influence`).
* Bazę amplitude trzymamy **>0** (np. `0.25 + 0.75*A`), by uniknąć całkowitego „wyłączenia” efektu i artefaktów krawędziowych.

---

## Replikowalność i raportowanie

* Utrwal **wejście** (hash pliku), **preset/filtr + parametry**, **seed**.
* Dołącz zrzuty **HUD** (mapy diagnostyczne) — to część wyniku, nie tylko obraz końcowy.
* Test relacji między filtrami rób jako **A/B** (zamiana kolejności) i notuj różnice.

---

## Rozszerzanie

* **Nowy filtr:** plik w `glitchlab/filters/` z `@register("nazwa")` i **jawny import** w `glitchlab/filters/__init__.py`.
* **Mini-UI do filtra:** panel w `glitchlab/gui/panels/` + rejestr w `panel_loader`.
* **Preset:** nowy YAML w `glitchlab/presets/`.

Szkielet filtra:

```python
from glitchlab.core.registry import register
import numpy as np

@register("my_filter")
def my_filter(img, ctx, strength: float = 1.0, mask_key: str|None = None):
    # użyj ctx.masks / ctx.amplitude; diagnostyka → ctx.cache["..."] = mapa 2D/3D
    return img
```

---

## Rozwiązywanie problemów (konkretnie)

* **„Unknown filter '…'”** — dopisz jawny import modułu w `glitchlab/filters/__init__.py`; sprawdź alias nazwy.
* **„Maska nie w HUD”** — po wczytaniu maski GUI tworzy `ctx` i rejestruje maskę; jeśli nie widzisz, zweryfikuj rozmiar maski (= obraz).
* **„Brak efektu”** — zbyt niskie `p/mix/strength` albo amplitude bliskie 0; podnieś wartości i sprawdź mapy diagnostyczne.

---

## Licencja i autorzy

Open Source — D2J3 aka Cha0s (for test and fun)

---

## Dalsza lektura

* **ANALYSIS.md** — metody badań, testy A/B, sweepy parametrów, miary (SSIM/PSNR/entropia), interpretacja wzorców.

---

