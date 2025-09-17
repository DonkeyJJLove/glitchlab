# GlitchLab Filters v4.5 – ARCHITECTURE.md

**Autorzy i licencja:** Open Source — D2J3 aka Cha0s (for test and fun)

---

## Spis treści
1. [Overview](#1-overview)  
2. [Rola filtrów](#2-rola-filtrów)  
3. [Struktura katalogu](#3-struktura-katalogu)  
4. [Kontrakty API](#4-kontrakty-api)  
5. [Wspólne parametry](#5-wspólne-parametry)  
6. [Diagnostyka i cache](#6-diagnostyka-i-cache)  
7. [Dodawanie nowego filtra](#7-dodawanie-nowego-filtra)  
8. [Status wdrożenia](#8-status-wdrożenia)  

---

## 1. Overview
Filtry to podstawowe jednostki transformacji obrazu w GlitchLab.  
Każdy filtr jest **funkcją zarejestrowaną w registry**, wywoływaną w ramach pipeline.  
Filtry odpowiadają za tworzenie efektów, ale także zapisują mapy diagnostyczne do HUD.

---

## 2. Rola filtrów
- Transformują obraz wejściowy w obraz wyjściowy.  
- Mogą korzystać z masek (`ctx.masks`).  
- Mogą uwzględniać mapę amplitude (`ctx.amplitude`).  
- Zapisują diagnostykę do `ctx.cache["diag/<filter>/..."]`.  

---

## 3. Struktura katalogu
```

glitchlab/filters/
**init**.py        # rejestracja filtrów
filter\_a.py        # przykład filtra A
filter\_b.py        # przykład filtra B
...

````

---

## 4. Kontrakty API
```python
@register(name="my_filter", defaults={"strength":1.0}, doc="Opis filtra")
def my_filter(img: np.ndarray, ctx: Ctx, **params) -> np.ndarray:
    # img: np.uint8 RGB
    # ctx: kontekst wykonania
    # params: specyficzne parametry filtra
    return out_img
````

---

## 5. Wspólne parametry

* `mask_key: str|None` — wybór maski z `ctx.masks`.
* `use_amp: bool|float` — skalowanie efektu mapą amplitude.
* `clamp: bool` — czy obcinać wartości wyjściowe do `[0..255]`.

---

## 6. Diagnostyka i cache

Każdy filtr zapisuje przynajmniej 1–2 mapy diagnostyczne:

* `ctx.cache["diag/<name>/mask"]`
* `ctx.cache["diag/<name>/strength"]`
* `ctx.cache["diag/<name>/debug"]`

---

## 7. Dodawanie nowego filtra

1. Utwórz moduł `filters/my_filter.py`.
2. Dodaj funkcję z dekoratorem `@register`.
3. Obsłuż wspólne parametry (`mask_key`, `use_amp`, `clamp`).
4. Dodaj mapy diagnostyczne do `ctx.cache`.
5. (Opcjonalnie) przygotuj panel GUI w `gui/panels/panel_my_filter.py`.

---

## 8. Status wdrożenia

* Wszystkie filtry v4 kompatybilne z API v2.
* Migracja filtrów legacy do nowego API — **on-run**.
* Dokumentacja filtrów (opis parametrów, przykłady) — **on-run**.

````

---

# `glitchlab/presets/ARCHITECTURE.md`

```markdown
# GlitchLab Presets v4.5 – ARCHITECTURE.md

**Autorzy i licencja:** Open Source — D2J3 aka Cha0s (for test and fun)

---

## Spis treści
1. [Overview](#1-overview)  
2. [Rola presetów](#2-rola-presetów)  
3. [Struktura katalogu](#3-struktura-katalogu)  
4. [Schema v2](#4-schema-v2)  
5. [Migracja i walidacja](#5-migracja-i-walidacja)  
6. [Ładowanie i zapisywanie](#6-ładowanie-i-zapisywanie)  
7. [Dodawanie nowego presetu](#7-dodawanie-nowego-presetu)  
8. [Status wdrożenia](#8-status-wdrożenia)  

---

## 1. Overview
Presety to zapisane konfiguracje pipeline.  
Są przechowywane jako pliki YAML i stanowią **punkt wejścia** do Core.  

---

## 2. Rola presetów
- Określają sekwencję filtrów i ich parametry.  
- Umożliwiają powtarzalność i testy A/B.  
- Są podstawą do integracji z GUI (ładowanie, zapisywanie, edycja).  

---

## 3. Struktura katalogu
````

glitchlab/presets/
**init**.py
example1.yaml
example2.yaml
...

````

---

## 4. Schema v2
```yaml
version: 2
name: "<NAME>"
seed: 7
amplitude: { kind: none|linear_x|linear_y|radial|perlin|mask, strength: 1.0 }
edge_mask: { thresh: 60, dilate: 0, ksize: 3 }
steps:
  - name: <canonical_filter_name>
    params: { ... parametry filtra ... }
````

---

## 5. Migracja i walidacja

* Presety legacy → v2 (narzędzie migracyjne).
* Walidator schema v2 + dry-run (P3) — **on-run**.

---

## 6. Ładowanie i zapisywanie

* `PresetService` w GUI odpowiada za open/save.
* Core przyjmuje presety wyłącznie w schema v2.

---

## 7. Dodawanie nowego presetu

1. Utwórz plik YAML zgodny ze schema v2.
2. Upewnij się, że wszystkie filtry istnieją w rejestrze.
3. Zweryfikuj preset walidatorem.
4. Przetestuj smoke-run w Core.

---

## 8. Status wdrożenia

* Schema v2 stabilna.
* Migracja presetów legacy → v2 — **on-run**.
* Walidator i dry-run smoke testy — **on-run**.

```

