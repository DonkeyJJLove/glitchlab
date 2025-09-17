# GlitchLab Core v4.5 – ARCHITECTURE.md

**Autorzy i licencja:** Open Source — D2J3 aka Cha0s (for test and fun)

---

## Spis treści

1. [Overview](#1-overview)  
2. [Główne odpowiedzialności](#2-główne-odpowiedzialności)  
3. [Struktura modułów](#3-struktura-modułów)  
   - [3.1 registry.py](#31-registrypy)  
   - [3.2 pipeline.py](#32-pipelinepy)  
   - [3.3 graph.py](#33-graphpy)  
   - [3.4 mosaic.py](#34-mosaicpy)  
   - [3.5 astmap.py](#35-astmappy)  
   - [3.6 metrics/](#36-metrics)  
   - [3.7 utils.py / roi.py / symbols.py](#37-utilspy--roipy--symbolspy)  
4. [Kontrakty API](#4-kontrakty-api)  
5. [Przepływ danych](#5-przepływ-danych)  
6. [Integracja z Analysis i GUI](#6-integracja-z-analysis-i-gui)  
7. [Status wdrożenia](#7-status-wdrożenia)  

---

## 1. Overview

Core to **serce GlitchLab**: odpowiada za wykonywanie pipeline filtrów, obliczanie metryk, budowę grafów, zarządzanie presetami i zapisywanie wszystkich wyników w postaci kluczy w `ctx.cache`.  
Core **nie renderuje UI** – jego jedynym wyjściem są dane (obrazy, miniatury, metryki, JSON), które następnie wykorzystuje GUI/HUD.

---

## 2. Główne odpowiedzialności

- Rejestracja i zarządzanie filtrami (`registry.py`).  
- Normalizacja i uruchamianie presetów (`pipeline.py`).  
- Budowa grafów procesów (DAG) i ich eksport (`graph.py`).  
- Tworzenie map mozaikowych (wizualizacje metryk) (`mosaic.py`).  
- Mapowanie AST kodu → graf semantyczny (`astmap.py`).  
- Liczenie metryk jakościowych i porównawczych (`metrics/`).  
- Zarządzanie maskami, symbolami i konwersjami typów (`utils.py`, `roi.py`, `symbols.py`).  

---

## 3. Struktura modułów

### 3.1 `registry.py`
- Funkcje:  
  - `register(name, defaults, doc)` – rejestracja filtra.  
  - `get(name)` – pobranie funkcji filtra.  
  - `available()` – lista dostępnych filtrów.  
  - `canonical(name)` – rozwiązywanie aliasów.  
  - `alias(src, dst)` – tworzenie aliasów.  
  - `meta(name)` – metadane (nazwa, domyślne parametry, dokumentacja).  
- Kluczowe: zapewnia **jedną sygnaturę API** dla wszystkich filtrów.

### 3.2 `pipeline.py`
- Funkcje:  
  - `normalize_preset(cfg)` – normalizacja presetów do schematu v2.  
  - `build_ctx(img, seed, cfg)` – konstrukcja kontekstu `Ctx`.  
  - `apply_pipeline(img, ctx, steps)` – wykonanie kroków pipeline.  
- Odpowiedzialności:  
  - Każdy krok dodaje wyniki do `ctx.cache`.  
  - Obsługa czasu wykonania (`t_ms`) i różnic między krokami.  
  - Deterministyczne RNG (kontrolowane seedem).  

### 3.3 `graph.py`
- Budowa lekkiego DAG procesu (nodes/edges/delta).  
- Eksport grafu jako JSON (`ctx.cache["ast/json"]`).  
- Wsparcie dla wizualizacji w GUI (`GraphView`).  

### 3.4 `mosaic.py`
- Projekcja metryk blokowych (square/hex) na mapy.  
- Funkcja `mosaic_project_blocks(...)`.  
- Wynik zapisywany w `ctx.cache["stage/{i}/mosaic"]`.  

### 3.5 `astmap.py`
- Konwersja AST Pythona do grafu semantycznego.  
- Normalizacja nazw i powiązań.  
- Eksport spójny z `graph.py` dla HUD.  

### 3.6 `metrics/`
- `basic.py` – podstawowe metryki: entropia, krawędzie, kontrast, bloki.  
- `compare.py` – metryki porównawcze: PSNR, SSIM.  
- Używane w pipeline i zapisane jako `ctx.cache["stage/{i}/metrics_*"]`.  

### 3.7 `utils.py / roi.py / symbols.py`
- **utils.py** – konwersje typów, operacje pomocnicze.  
- **roi.py** – zarządzanie regionami zainteresowania (ROI).  
- **symbols.py** – tłumaczenie symboli na maski i odwrotnie.  

---

## 4. Kontrakty API

### 4.1 Filter API v2
```python
def my_filter(img: np.ndarray, ctx: Ctx, **params) -> np.ndarray
````

* Wejście: `np.uint8 RGB`.
* Wyjście: `np.uint8 RGB`.
* RNG: tylko `ctx.rng`.
* Diagnostyki: `ctx.cache["diag/<name>/..."]`.

### 4.2 Kontekst wykonania (`Ctx`)

```python
@dataclass
class Ctx:
    rng: np.random.Generator
    amplitude: np.ndarray
    masks: Dict[str, np.ndarray]
    cache: Dict[str, Any]
    meta: Dict[str, Any]
```

---

## 5. Przepływ danych

```
Preset YAML → normalize_preset(cfg)
→ build_ctx(img, seed, cfg)
→ apply_pipeline(img, ctx, steps)
    ↳ filtr → zapis wyników w ctx.cache
→ build_and_export_graph(...)
→ ctx.cache (HUD keys)
```

---

## 6. Integracja z Analysis i GUI

* **Analysis**: Core może opakowywać wybrane metryki i eksportery, zachowując interfejsy.
* **GUI**: działa wyłącznie na danych z `ctx.cache`.
* GUI nigdy nie woła filtrów bezpośrednio → zawsze przez `pipeline.apply_pipeline`.

---

## 7. Status wdrożenia

* Wszystkie opisane komponenty działają zgodnie z architekturą.
* Elementy oznaczone jako **on-run**:

  * Migracja presetów legacy → v2 — **on-run**.
  * Walidator presetów i dry-run smoke tests — **on-run**.
  * Rozszerzone metryki porównawcze (dla A/B testing) — **on-run**.

---