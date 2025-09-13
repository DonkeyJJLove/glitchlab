# GlitchLab Filters v2 — Architektura (warstwa `filters/`)

Ten dokument opisuje **warstwę filtrów** GlitchLab v2: kontrakty API, konwencje, katalog dostępnych rodzin filtrów, standard diagnostyki (HUD), wzorce implementacyjne oraz testy smoke. Jest spójny z dokumentami:
**System Architecture**, **core/ARCHITECTURE.md**, **gui/ARCHITECTURE.md**.
Refaktoryzacje i zmiany projektowe opisujemy **w osobnym pliku**.

---

## 1) Cel i zakres

Warstwa `filters/` dostarcza **deterministyczne** transformaty obrazu 2D, używane przez `core.pipeline` w krokach presetów v2. Każdy filtr:

* implementuje **jedną sygnaturę** (`Filter API v2`),
* publikuje **defaults/doc** przez rejestr (`core.registry`),
* odkłada **mapy diagnostyczne** do `ctx.cache["diag/<name>/..."]`,
* szanuje globalne **maski** i **amplitude** (obsługiwane wrapperem w `pipeline`),
* działa **bez** SciPy/OpenCV: tylko **NumPy** + **Pillow** (opcjonalnie `noise`).

---

## 2) Kontrakt: Filter API v2

```python
def my_filter(img_u8: np.ndarray, ctx: Ctx, **params) -> np.ndarray
# Wejście:  uint8 RGB (H, W, 3)
# Wnętrze:  rekomendowane float32 [0,1]
# Wyjście:  uint8 RGB (H, W, 3)

# Parametry wspólne (zdejmuje je wrapper w pipeline):
#   mask_key: str|None   # blend wejścia/efektu z ctx.masks[mask_key] ∈ [0,1]
#   use_amp:  float|bool # modulacja siły efektem ctx.amplitude ∈ [0,1]
#   clamp:    bool       # końcowy clip do [0,1] po blendzie
#
# Losowość tylko przez ctx.rng (deterministyczny seed)
# Diagnostyka do ctx.cache["diag/<filter_name>/..."]
```

> **Ważne:** wrapper w `core.pipeline.apply_pipeline` wycina `mask_key|use_amp|clamp` z `params` i nakłada je **po** zwróceniu obrazu przez filtr. Filtr może je respektować wewnętrznie, ale zalecamy **pozostawić to wrapperowi**, aby GUI/HUD były spójne.

---

## 3) Rejestracja i metadane

Każdy moduł w `filters/` rejestruje funkcję dekoratorem:

```python
from glitchlab.core.registry import register

DEFAULTS = {"strength": 1.0, "mask_key": None, "use_amp": 1.0, "clamp": True}
DOC = "Krótki opis techniczny i przeznaczenie; 1–3 zdania."

@register("my_filter", defaults=DEFAULTS, doc=DOC)
def my_filter(img_u8: np.ndarray, ctx, **p) -> np.ndarray:
    ...
    ctx.cache["diag/my_filter/example"] = some_map_or_scalar
    return out_u8
```

**Zasady:**

* Nazwy **kanoniczne** są **małymi literami** z `_` (np. `anisotropic_contour_warp`).
* Aliasowanie tylko przez `core.registry.alias(...)` (bez prywatnych mapowań).
* `defaults` obejmuje także pola wspólne (`mask_key|use_amp|clamp`), aby GUI je wyświetlało (mimo, że pipeline je zdejmuje).

---

## 4) Kanon nazw i aliasy (v2)

**Alias → kanon (wycinek używany w presetach i GUI):**

* `conture_flow | anisotropic_contour_flow` → `anisotropic_contour_warp`
* `block_mosh` → `block_mosh_grid`
* `spectral_shaper_lab | spectral_ring` → `spectral_shaper`
* `perlin_grid | nosh_perlin_grid` → `noise_perlin_grid`

> Jeśli w repo istnieją starsze nazwy, zapewnij alias w rejestrze *po* zarejestrowaniu kanonu.

---

## 5) Inwentarz rodzin filtrów (stan wg paneli i dokumentacji)

> Lista odzwierciedla **rzeczywiste** moduły/panele wymienione w GUI i dokumentach. Nazwy kanoniczne ustala rejestr (`registry.available()` jest źródłem prawdy).

**Geometria / deformacje**

* `anisotropic_contour_warp` — przesuw wzdłuż konturów (tangent do ∇I).
* `depth_displace` — przemieszczenie wg mapy głębi.
* `depth_parallax` — paralaksa na bazie głębi/parallaksowania (panel w GUI).

**Blokowość / siatki**

* `block_mosh_grid` — przesunięcia/rotacje bloków na siatce (rezonans rozmiaru).
* `tile_tess_probe` — testy/sondy mozaik/tilingu (panel w GUI).

**Spektralne / częstotliwości**

* `spectral_shaper` — kształtowanie pasm (np. ring/sector w FFT).
* `phase_glitch` — manipulacje fazą (FFT → ifft).

**Sortowanie / porządkowanie**

* `pixel_sort_adaptive` — adaptacyjny pixel sort (po kierunku/regionach).

**Kolor / kanały**

* `rgb_offset` — przesunięcia kanałów RGB (chromatyczny drift).
* `rgb_glow` — poświata RGB (rozlane halo).
* `gamma_gain` — krzywa gamma / gain (panel w GUI).
* `default_identity` — no-op (do testów formularzy, sanity w GUI).

**Szum / wzorce**

* `noise_perlin_grid` — wzorce perlin/value-noise na siatce (używane też jako efekt).

> Powyższe nazwy mogą być rozszerzone przez inne filtry obecne w repo; jeśli panel istnieje (w `gui/panels/`), oczekuj odpowiadającego modułu w `filters/` (lub aliasu w rejestrze).

---

## 6) Wzorce implementacyjne (bez SciPy/OpenCV)

### 6.1 Geometria (warp/offset/flow)

* **Wejście**: `x ∈ float32 [0,1]` (z `img_u8/255`), kształt `(H,W,3)`.
* **Pole przesunięcia**: buduj wektorowe mapy `dx, dy ∈ float32`, najlepiej wektorowo (bez pętli per piksel).
* **Próbkowanie**: używaj `np.roll` dla przesunięć całkowitych; dla subpikselowych — bilinear w NumPy (2×2 sąsiedzi; waga).
  Uwaga: Pillow może pomóc w globalnych transformacjach, ale dla pola wektorowego implementuj własny sampler.
* **Przykłady diagnostyk**: `diag/<name>/dx`, `diag/<name>/dy`, `diag/<name>/mag`.

```python
def _bilinear_sample(img, x, y):
    # img: (H,W,3) f32 [0,1], x/y: float grids (H,W)
    H, W = img.shape[:2]
    x0 = np.floor(x).astype(np.int32).clip(0, W-1)
    x1 = (x0 + 1).clip(0, W-1)
    y0 = np.floor(y).astype(np.int32).clip(0, H-1)
    y1 = (y0 + 1).clip(0, H-1)
    wx = (x - x0).astype(np.float32)
    wy = (y - y0).astype(np.float32)

    Ia = img[y0, x0]; Ib = img[y0, x1]
    Ic = img[y1, x0]; Id = img[y1, x1]
    Iab = Ia * (1-wx[...,None]) + Ib * (wx[...,None])
    Icd = Ic * (1-wx[...,None]) + Id * (wx[...,None])
    return Iab * (1-wy[...,None]) + Icd * (wy[...,None])
```

### 6.2 Siatka bloków

* Podział `(H,W)` na bloki `B×B`: użyj `reshape` lub `np.add.reduceat`.
* Losowy wybór bloków: **tylko** `ctx.rng` (np. `rng.random(size)`).
* Ruch/rotacja bloków: indeksowanie na płaskim wektorze bloków, a potem re-reshape.
* Diagnostyki: `diag/<name>/select` (maska wybranych bloków), `diag/<name>/dx/dy`.

```python
def _block_view(x, B):
    H, W, C = x.shape
    H2, W2 = H//B, W//B
    x = x[:H2*B, :W2*B]
    return x.reshape(H2, B, W2, B, C).swapaxes(1,2)  # (H2, W2, B, B, C)
```

### 6.3 FFT / spektrum

* Używaj `np.fft.rfft2` / `np.fft.irfft2` (dla kanałów osobno lub na luminancji).
* Kształtuj **maski częstotliwości** (pierścienie/sektory) wektorowo; unikaj pętli.
* Zwróć uwagę na normalizację amplitud (po ifft clip do \[0,1]).
* Diagnostyki: `diag/<name>/fft_mag` (log-magnitude), `diag/<name>/mask`.

### 6.4 Sortowanie pikseli

* Kierunkowość: użyj gradientu (proste różnice `np.roll`), by określić rampę sortowania.
* Operuj na wierszach/kolumnach/segmentach przez **maski** wektorowe.
* Diagnostyki: `diag/<name>/sort_mask`, `diag/<name>/order`.

### 6.5 Kolor / kanały

* `rgb_offset`: `np.roll` niezależnie dla kanałów, offsety mogą być modulowane amplitude.
* `gamma_gain`: `y = x**gamma * gain` (zabezpieczyć \[0,1]).
* Diagnostyki: `diag/<name>/ch_offset` (np. `[dx_r, dx_g, dx_b]`), `diag/<name>/lut`.

### 6.6 Szum / Perlin

* Preferuj `noise.pnoise2` (jeśli lib dostępna) lub fallback „value-noise” (sumy skali + blur Box).
* Losowość i oktawy/skalowanie — zdeterminowane przez `ctx.rng`.
* Diagnostyki: `diag/<name>/field` (pole szumu), `diag/<name>/octaves`.

---

## 7) Diagnostyka i HUD — konwencje kluczy

Filtry **muszą** odkładać przynajmniej 1–2 artefakty diagnostyczne. Zalecane nazewnictwo:

* Maska/wybór: `diag/<name>/mask`, `diag/<name>/select`
* Pole wektorowe: `diag/<name>/dx`, `diag/<name>/dy`, `diag/<name>/mag`
* Spektrum: `diag/<name>/fft_mag`, `diag/<name>/ring_mask`, `diag/<name>/sector_mask`
* Histogram/rozklad: `diag/<name>/hist`
* Parametry runtime: `diag/<name>/strength`, `diag/<name>/iters`, ...

**Formaty:**

* Obrazy: `np.ndarray` (`uint8 RGB` lub `float32 [0,1]`), kształt kompatybilny z viewerem GUI.
* Skalary/listy: dozwolone (GUI pokaże etykietę/tekst).

> Wszystkie nazwy są stabilne; unikaj zmian bez aktualizacji dokumentacji.

---

## 8) Walidacja parametrów i typy

**Zasady ogólne:**

* Każdy parametr z `params` rzutuj na **oczekiwany typ** (`int/float/bool/str/enum`).
* Zakresy:
  `strength ∈ [0, +]`, `p ∈ [0,1]`, `size ∈ ℕ+`, `ksize ∈ {3,5,7,...}`,
  `mode ∈ {"wrap","clamp","mirror",...}` (jeśli dotyczy).
* Ignoruj **nadmiarowe** parametry (zaloguje to `pipeline`), nie podnoś wyjątków.
* Ustal **DEFAULTS**: sensowne i bezpieczne wartości (no-op przy `strength=0` gdzie to możliwe).

---

## 9) Deterministyczność i RNG

* **Wyłącznie** `ctx.rng` (np. `rng = ctx.rng; u = rng.random(shape, dtype=np.float32)`).
* Zero globalnego `np.random.*`.
* Jeżeli filtr implementuje złożoną stochastykę (np. wybór bloków), odłóż **seed/param** do `ctx.cache["diag/<name>/seed"]` dla reprodukcji.

---

## 10) Wydajność

* Unikaj pętli per-piksel; stosuj `np.roll`, broadcast, wektoryzację.
* Dla bloków: `reshape/swapaxes` zamiast iteracji.
* FFT: preferuj luminancję lub pojedynczy kanał, jeśli pełne RGB nie jest potrzebne.
* Próbkowanie bilinear: licz wektorowo; unikaj `for`.
* Konwersje dtype minimalizuj do wejścia/wyjścia; w środku trzymaj `float32`.

---

## 11) Integracja z maską i amplitude

* **Wrapper** w `pipeline` stosuje `mask_key|use_amp`.
  Rekomendacja: filtr **nie** musi sam nakładać maski/amplitude — dzięki temu diagnostyki są spójne, a GUI może porównywać „czysty” efekt filtra do wejścia.

* Jeżeli jednak filtr **musi** wiedzieć o amplitude wewnętrznie (np. nieliniowa modulacja parametru), odczytaj:

  ```python
  A = ctx.amplitude  # (H,W) f32 [0,1] lub None
  ```

  i **zapisz** tę decyzję w diagnostyce (`diag/<name>/amp_used`), aby było to widoczne w HUD.

---

## 12) Przykładowe szkielety

### 12.1 Minimalny filtr (tonalny)

```python
from glitchlab.core.registry import register
import numpy as np

DEFAULTS = {"gain": 1.0, "gamma": 1.0, "mask_key": None, "use_amp": 1.0, "clamp": True}
DOC = "Prosty filtr tonalny: y = x**gamma * gain."

@register("gamma_gain", defaults=DEFAULTS, doc=DOC)
def gamma_gain(img_u8, ctx, **p):
    x = img_u8.astype(np.float32) / 255.0
    gain  = float(p.get("gain", 1.0))
    gamma = float(p.get("gamma", 1.0))
    y = np.power(np.clip(x, 1e-6, 1.0), gamma) * gain
    y = np.clip(y, 0.0, 1.0)
    ctx.cache["diag/gamma_gain/lut"] = (np.linspace(0,1,256, dtype=np.float32) ** gamma) * gain
    return (y * 255.0 + 0.5).astype(np.uint8)
```

### 12.2 Warp anizotropowy (rdzeń)

```python
@register("anisotropic_contour_warp", defaults={
    "strength": 1.0, "iters": 1, "ksize": 3, "smooth": 0.0, "edge_bias": 0.5,
    "mask_key": None, "use_amp": 1.0, "clamp": True
}, doc="Przesuw po tangencie konturu (∇I⊥).")
def anisotropic_contour_warp(img_u8, ctx, **p):
    x = img_u8.astype(np.float32) / 255.0
    ksize   = int(p.get("ksize", 3))
    iters   = int(p.get("iters", 1))
    strength= float(p.get("strength", 1.0))
    # gradient (różnice centralne)
    gx = np.roll(x, -1, axis=1) - np.roll(x, 1, axis=1)
    gy = np.roll(x, -1, axis=0) - np.roll(x, 1, axis=0)
    g  = np.mean(np.abs(gx)+np.abs(gy), axis=2).astype(np.float32)
    # tangent ~ (ty, -tx) po uśrednieniu
    tx = gy.mean(axis=2); ty = -gx.mean(axis=2)
    n  = np.maximum(np.sqrt(tx*tx + ty*ty), 1e-6)
    tx /= n; ty /= n
    # kroki Eulera po tangencie
    H,W = x.shape[:2]
    yy, xx = np.meshgrid(np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing='ij')
    for _ in range(iters):
        xx = np.clip(xx + strength * tx, 0, W-1)
        yy = np.clip(yy + strength * ty, 0, H-1)
    y = _bilinear_sample(x, xx, yy)
    y = np.clip(y, 0.0, 1.0)
    ctx.cache["diag/anisotropic_contour_warp/acw_mag"] = g
    ctx.cache["diag/anisotropic_contour_warp/acw_tx"]  = tx
    ctx.cache["diag/anisotropic_contour_warp/acw_ty"]  = ty
    return (y * 255.0 + 0.5).astype(np.uint8)
```

### 12.3 Blokowy mosh

```python
@register("block_mosh_grid", defaults={
    "size": 24, "p": 0.5, "max_shift": 16, "mode": "wrap", "mix": 1.0,
    "mask_key": None, "use_amp": 1.0, "clamp": True
}, doc="Losowe przesunięcia/rotacje bloków na siatce.")
def block_mosh_grid(img_u8, ctx, **p):
    x = img_u8.astype(np.float32) / 255.0
    B = int(p.get("size", 24))
    H,W,_ = x.shape
    H2,W2 = H//B, W//B
    xb = _block_view(x, B)                     # (H2, W2, B, B, C)
    rng = ctx.rng
    select = (rng.random((H2, W2)) < float(p.get("p", 0.5)))
    dx = (rng.integers(-1, 2, size=(H2,W2)) * int(p.get("max_shift", 16))).astype(np.int32)
    dy = (rng.integers(-1, 2, size=(H2,W2)) * int(p.get("max_shift", 16))).astype(np.int32)
    xb2 = np.copy(xb)
    for i in range(H2):
        for j in range(W2):
            if not select[i,j]: continue
            ii = np.clip(i + dy[i,j]//max(1,B), 0, H2-1)
            jj = np.clip(j + dx[i,j]//max(1,B), 0, W2-1)
            xb2[i,j] = xb[ii,jj]
    y = xb2.swapaxes(1,2).reshape(H2*B, W2*B, 3)
    # reszta brzegowa
    out = np.copy(x)
    out[:H2*B, :W2*B] = y
    mix = float(p.get("mix", 1.0))
    out = x*(1-mix) + out*mix
    out = np.clip(out, 0, 1)
    ctx.cache["diag/block_mosh_grid/bmg_select"] = select.astype(np.float32)
    ctx.cache["diag/block_mosh_grid/bmg_dx"] = dx
    ctx.cache["diag/block_mosh_grid/bmg_dy"] = dy
    return (out * 255.0 + 0.5).astype(np.uint8)
```

*(Pętle po blokach są dopuszczalne — ich liczba jest o rząd wielkości mniejsza niż liczba pikseli. Można je zredukować przez wektorowe permutacje indeksów.)*

---

## 13) Przykłady kroków w presetach (YAML v2)

```yaml
steps:
  - name: anisotropic_contour_warp
    params: { strength: 1.2, iters: 2, ksize: 3, edge_bias: 0.5, use_amp: 1.0, clamp: true }
  - name: block_mosh_grid
    params: { size: 24, p: 0.45, max_shift: 32, mode: wrap, mix: 0.9 }
  - name: spectral_shaper
    params: { ring_center: 0.35, ring_width: 0.1, gain: 1.25 }
  - name: pixel_sort_adaptive
    params: { axis: auto, window: 33, threshold: 0.15 }
  - name: rgb_offset
    params: { dx_r: 2, dy_r: 0, dx_g: -1, dy_g: 1, dx_b: 0, dy_b: -2, mix: 0.8 }
```

---

## 14) Testy smoke (dla każdego filtra)

Minimalny **test dymny** uruchamiany lokalnie:

```python
import numpy as np
from glitchlab.core.pipeline import build_ctx
from glitchlab.core.registry import get, meta

def smoke_filter(name: str):
    img = np.zeros((48,64,3), np.uint8) + 40
    ctx = build_ctx(img, seed=7, cfg=None)
    fn  = get(name)
    dfl = meta(name)["defaults"]
    # przypadki: bez amp; z amp=0; z maską edge
    out1 = fn(img, ctx, **{k:v for k,v in dfl.items() if k not in ("mask_key","use_amp","clamp")})
    assert out1.shape == img.shape and out1.dtype == np.uint8
    ctx.masks["edge"] = np.where(np.indices(img.shape[:2])[0] % 2 == 0, 1.0, 0.0).astype(np.float32)
    out2 = fn(img, ctx, **{**dfl, "mask_key":"edge"})
    out3 = fn(img, ctx, **{**dfl, "use_amp":0.0})
    assert out2.shape == img.shape and out3.shape == img.shape
```

Wymagania testu:

* **Brak wyjątków**, poprawny kształt/dtype.
* W `ctx.cache` pojawiają się co najmniej 1–2 klucze `diag/<name>/...`.
* Parametry spoza `defaults` są **ignorowane**, nie powodują błędów.

---

## 15) Spójność z „mozaiką” (meta-struktura)

Filtry są **soczewek generacyjnymi**: produkują pola (maski, wektory, mapy częstotliwości), które `core`/`analysis` potrafią rzucić na **wspólną mozaikę** (blokową lub hex).
Zalecenia:

* jeżeli filtr naturalnie wytwarza **mapę skalarową** (np. selekcję bloków, gęstość krawędzi, siłę efektu), zapisz ją w `diag/<name>/map` — ułatwi to projekcję przez `core.mosaic`.
* jeżeli filtr wytwarza **pole wektorowe** (dx/dy), zapisz oba kanały; HUD może pokazać je jako parę map lub pseudo-kolor.

Taka „mozaikowa” ontologia umożliwia porównywanie wyników między filtrami i etapami pipeline’u w jednym, wspólnym układzie odniesienia.

---

## 16) Zgodność i wersjonowanie

* Filtry v2 utrzymują stabilne **sygnatury** i **defaults**; zmiany wymagają aktualizacji rejestru i dokumentacji.
* Nowe aliasy rejestrować jawnie (bez ukrytych mapowań).
* Zmiana zachowania domyślnego → opis w `CHANGELOG.md` (część core).

---

## 17) Checklist dla nowego filtra

* [ ] Sygnatura zgodna z **Filter API v2**.
* [ ] Rejestracja przez `@register(...)` z sensownymi `defaults/doc`.
* [ ] Brak SciPy/OpenCV; tylko NumPy + Pillow (opcjonalnie `noise`).
* [ ] Diagnostyki w `ctx.cache["diag/<name>/..."]` (min. 1–2 mapy/skalary).
* [ ] Deterministyczne RNG (`ctx.rng`) — zero `np.random.*`.
* [ ] Wektoryzacja — bez pętli per-piksel (wyjątki: niskoliczbowe pętle po blokach).
* [ ] Parametry nadmiarowe **ignorowane**; typy rzutowane wewnętrznie.
* [ ] No-op przy siłach 0 (jeśli to naturalne dla efektu).
* [ ] Smoke-test przechodzi lokalnie.

---

## 18) FAQ (skrót)

**Czy mogę modyfikować `ctx.cache` poza `diag/...`?**
Tak, ale trzymaj się stabilnych kluczy (zob. słownik HUD w dokumentach core). `diag/...` jest przestrzenią nazw filtra.

**Czy filtr może zwrócić `float32 [0,1]`?**
Zalecamy **zwracać `uint8`**. Pipeline spróbuje rzutować, ale kontrola po stronie filtra jest bezpieczniejsza.

**Jak obsłużyć brak `noise`?**
Zapewnij fallback na „value-noise” (sumy skal + blur Box) inicjowany `ctx.rng`.

**Czy muszę sam robić clamp/mask/amplitude?**
Nie — wrapper w pipeline to zrobi. Wewnętrzna modulacja parametru przez amplitude jest dozwolona (odłóż `diag/<name>/amp_used`).

---

## 19) Licencja i autorzy

Open Source — D2J3 aka Cha0s (for test and fun)

---

