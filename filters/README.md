# GlitchLab Filters — projektowanie i diagnostyka

> **Filtry** są atomami transformacji. Każdy filtr ma **jedno API**, wspólne parametry i obowiązek zapisu diagnostyk do HUD. Filtry nie „wiedzą” o GUI — to core orkiestruje wykonanie i metryki.

---

## API i kontrakty

```python
@register("filter_name", defaults=DEFAULTS, doc=DOC)
def my_filter(img_u8: np.ndarray, ctx: Ctx, **params) -> np.ndarray:
    ...
```

* **Wejście/wyjście:** `uint8 RGB`. Wnętrze: `float32 [0,1]`.
* **Wspólne parametry:**
  `mask_key: str|None` – blend w ROI,
  `use_amp: float|bool` – modulacja siły przez `ctx.amplitude`,
  `clamp: bool` – końcowy `clip` i u8.
* **RNG:** tylko `ctx.rng`.
* **Diagnostyki:** wpisy do `ctx.cache["diag/<filter>/…"]` (min. 1–2 mapy).

---

## Nazewnictwo i aliasy

Rejestr (**core/registry.py**) trzyma kanony i aliasy. Przykłady mapowania:

* `conture_flow|anisotropic_contour_flow → anisotropic_contour_warp`
* `block_mosh → block_mosh_grid`
* `spectral_shaper_lab|spectral_ring → spectral_shaper`
* `perlin_grid|nosh_perlin_grid → noise_perlin_grid`

---

## Przykładowe filtry (skrót)

* **anisotropic\_contour\_warp (ACW):** przesuw wzdłuż konturów; diagnostyki: `acw_mag`, `acw_tx/ty`.
* **block\_mosh\_grid (BMG):** przestawianie/rotacja bloków; diagnostyki: `bmg_select`, `bmg_dx/dy`.

---

## Wzorce implementacyjne

* **Zero pętli per-piksel** (wektoryzacja, operacje blokowe).
* **Parametry poza `defaults`** – ignoruj i loguj (pipeline zapisze ostrzeżenie).
* **Maski/ROI** – dopasuj wymiar (H,W); w razie potrzeby bezpiecznie przeskaluj.

---

## Testy „smoke”

* Wejście: `zeros(48×64)+40` (u8).
* Przypadki: `use_amp=1.0`, `use_amp=0.0`, `mask_key="edge"`.
* Asercje: kształt, dtype, zakres \[0..255], brak wyjątków.

---

## Dokumentacja

* Interfejsy i kanały HUD: [../core/ARCHITECTURE.md](../core/ARCHITECTURE.md)
* Zasady i preset v2: [../ARCHITECTURE.md](../ARCHITECTURE.md)

---