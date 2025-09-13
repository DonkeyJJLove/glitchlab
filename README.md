# GlitchLab — controlled glitch for analysis

![UI](screen.png)

GlitchLab to narzędzie badawczo-analityczne do **kontrolowanej generacji artefaktów** w obrazach 2D. Błąd traktujemy jako **sygnał diagnostyczny**: sterując **gdzie** (maski ROI) i **jak mocno** (pole amplitudy) występuje, wyciągamy wnioski o **strukturze danych** i **sprzężeniach między transformacjami**. Całość jest deterministyczna (seed RNG), lekka (NumPy + Pillow), a diagnostyka jest produktem pierwszej klasy — wszystkie kroki odkładają telemetrię do `ctx.cache` (HUD).

> **Teza praktyczna:** artefakt = obserwowalny sygnał. Sterowanie jego lokalizacją i energią umożliwia wnioskowanie o strukturach i relacjach.

---

## Spis treści

* [Cele i założenia](#cele-i-założenia)
* [Szybki start](#szybki-start)
* [Mapa warstw (E2E)](#mapa-warstw-e2e)
* [Meta-struktura mozaikowa i „metametryka”](#meta-struktura-mozaikowa-i-metametryka)
* [Przepływy badań (protokóły)](#przepływy-badań-protokóły)
* [Artefakty a struktury (przykłady)](#artefakty-a-struktury-przykłady)
* [Struktura repo i dokumentacja](#struktura-repo-i-dokumentacja)
* [Licencja i autorzy](#licencja-i-autorzy)

---

## Cele i założenia

* **Deterministyczność:** jeden seed (`ctx.rng`), brak ukrytej losowości.
* **Jedno API filtrów:** `fn(img_u8, ctx, **params) -> np.ndarray` (wew. `float32 [0,1]`, I/O `uint8 RGB`).
* **Diagnostyka jako produkt:** każdy krok emituje kanały HUD (`stage/{i}/…`, `diag/<filter>/…`, `ast/json`, itp.).
* **Lekkość zależności:** Python 3.9+, **NumPy**, **Pillow**, **Tkinter** (GUI). **Bez** SciPy/OpenCV.
* **Spójne nazwy i presety v2:** jedno schema `version/name/amplitude/edge_mask/steps`.

---

## Szybki start

Wymagania: Python 3.9+, NumPy, Pillow, Tkinter.

```bash
pip install -r requirements.txt
python -m glitchlab.gui.app
```

Workflow:

1. **Open image…**
2. Ustaw **Amplitude/Edge/Mask** (prawy panel).
3. Wybierz **Preset** lub **Filter** → **Apply**.
4. Odczytaj diagnostykę w HUD, iteruj parametry.
5. **Save result…**

Do szybkich testów użyj planszy kalibracyjnej:

![Test chart](glitchlab_testchart_v1.png)

---

## Mapa warstw (E2E)

* **Core (`glitchlab/core`)** – rejestr filtrów, pipeline (metryki/diff), DAG procesu, mozaika, AST→graf, narzędzia.
* **Analysis (`glitchlab/analysis`)** – metryki (globalne/kafelkowe), diff, FFT/hist, forensyka formatu, eksport DTO.
* **Filters (`glitchlab/filters`)** – moduły z `@register`, spójne parametry (`mask_key|use_amp|clamp`), diagnostyki do HUD.
* **GUI (`glitchlab/gui`)** – stały layout: duży viewer, prawa kolumna „Parameters”, trzy sloty HUD, panele pływające.

Szczegóły:

* [core/ARCHITECTURE.md](core/ARCHITECTURE.md)
* [analysis/ARCHITECTURE.md](analysis/ARCHITECTURE.md)
* [filters/ARCHITECTURE.md](filters/ARCHITECTURE.md)
* [gui/ARCHITECTURE.md](gui/ARCHITECTURE.md)

---

## Meta-struktura mozaikowa i „metametryka”

**Mozaika** jest wspólną soczewką: raster komórek (square/hex) łączący obraz, statystyki blokowe i **strukturę kodu** (AST) w jednej przestrzeni wizualnej.

* `core/mosaic.py` – tworzy mapę etykiet komórek i projektuje metryki blokowe na overlay RGB.
* `core/astmap.py` – parsuje Python AST → lekki graf (węzły: funkcje/klasy; krawędzie: `contains`, `calls`) z metrykami:
  `weight` (rozmiar poddrzewa), `branching` (liczba {If/For/While/Try}), `fan_in/out`. Graf może być rzutowany na mozaikę.

**Metametryka** (warstwa spajająca):

* **Obraz → bloki:** `analysis.metrics.block_stats` → entropia, krawędzie, kontrast per blok.
* **Kod → węzły:** `astmap.ast_to_graph` → `weight/branching/fan_out`.
* **Mapowanie do RGB:** np. `R←branching`, `G←weight`, `B←fan_out` (normalizacja po grafie).
* **HUD:** `stage/{i}/mosaic` + `ast/json` — te same sloty służą do oglądania obrazu i struktury kodu.

> Dzięki temu ta sama „siatka uwagi” służy obserwacji **geometrii obrazu** i **geometrii programu**.

---

## Przepływy badań (protokóły)

* **Test komutacji (A/B):** sprawdza, czy kolejność filtrów ma znaczenie (`Δ = 1 − SSIM(A,B)`).
* **ROI-scan:** zmieniaj maskę ROI i porównuj |artefakt| w/poza ROI → lokalność/nielokalność.
* **Sweep parametrów:** szukaj progów (nagłe zmiany metryk).
* **Seed sweep:** deterministycznie, ale z kontrolą komponentu losowego.

Zobacz: [analysis/ARCHITECTURE.md](analysis/ARCHITECTURE.md) (sekcje: protokoły, metryki, receptury YAML).

---

## Artefakty a struktury (przykłady)

* **Anizotropia (ACW):** *anisotropic\_contour\_warp* przesuwa piksele **wzdłuż** konturów. Stabilna czytelność po kilku iteracjach ⇒ treść leży tangencjalnie do ∇I.
* **Blokowość/kompresja (BMG):** *block\_mosh\_grid* ujawnia rezonanse skali (np. 8/16). Histogram `bmg_dx/dy` i „siatka ducha” na test charcie naprowadzają na rozmiary bloków/skalowanie.
* **Pasma i aliasy:** widoczne w FFT/hist; kierunki odpowiadają dominującej geometrii tekstur.

---

## Struktura repo i dokumentacja

```
glitchlab/
  core/        # registry, pipeline, graph, mosaic, astmap, metrics, utils, roi, symbols
  analysis/    # metrics, diff, spectral, formats, exporters
  filters/     # filtry z @register
  gui/         # aplikacja i panele (HUD, graf, mozaika, parametry)
  presets/     # YAML v2
  screen.png
  glitchlab_testchart_v1.png
```

Każdy dział posiada **ARCHITECTURE.md** (interfejsy/kontrakty) i ten **README** (użycie, praktyki, metapoziom).

---

## Licencja i autorzy

Open Source — D2J3 aka Cha0s (for test and fun).
Wkład: struktury v2, HUD, mozaika/AST, metametryka.

---

