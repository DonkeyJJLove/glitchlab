# GlitchLab Analysis — metryki, różnice, pasma

> Warstwa **analysis** dostarcza **miary ilościowe** i wizualizacje, które wykorzystują core i GUI: metryki globalne/kafelkowe, porównania (diff/PSNR/SSIM), widma, forensykę formatu oraz eksport lekkich DTO dla HUD.

---

## Cele i zakres

* Pracujemy na obrazach 2D (RGB/Gray).
* Regulatory: maski ROI (`ctx.masks`) i pole amplitudy (`ctx.amplitude`).
* Obserwowalne: wynik kroku + mapy diagnostyczne (`ctx.cache`).

**Kluczowe pytania badawcze**:

* Czy treść jest **anizotropowa**?
* Czy istnieje **blokowość** (ślad kompresji/skalowania)?
* Czy kolejność filtrów ma znaczenie (**komutacja A/B**)?
* Gdzie dokładnie działa efekt (**lokalność**, ROI)?

---

## Moduły

* `metrics.py` – entropia, gęstość krawędzi, kontrast RMS; warianty blokowe.
* `diff.py` – wizualny diff i statystyki (mean/p95/max, opcjonalnie PSNR).
  Interfejs (skrótem):

  ```python
  from glitchlab.analysis.diff import compute_diff
  out = compute_diff(img_a, img_b, max_side=1024, add_psnr=True)
  # out: {"abs": f32[0,1], "per_channel": (dR,dG,dB), "heatmap", "stats": {...}}
  ```
* `spectral.py` – FFT, ring/sector, histogram.
* `formats.py` – noty forensyczne (np. grid 8×8).
* `exporters.py` – pakowanie telemetrii (bez obrazów) do HUD.

Szczegóły: [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Metametryka (mapy → liczby)

* **Globalne:** SSIM, PSNR, L1/L2, entropia, kontrast RMS.
* **Kafelkowe (mozaika):** statystyki per komórka; projekcja na overlay RGB (`core/mosaic.py`).
* **Specyficzne dla filtrów:** histogram `bmg_dx/dy`, mapa `acw_mag`.

---

## Protokóły badawcze

* **Komutacja A/B**

  ```
  I → F1 → F2 → A
  I → F2 → F1 → B
  Δ = 1−SSIM(A,B)
  ```
* **ROI-scan:** amplitude.kind=mask → |artefakt| w ROI vs. poza.
* **Sweep:** skany 1D/2D parametrów, szukanie progów.
* **Seed sweep:** kontrolowana losowość, wariancja metryk vs. seed.

---

## Praktyka z planszą testową

Użyj `glitchlab_testchart_v1.png`:

* pola jednolite → test nielokalności/„wycieku” efektu,
* szachownice i siatki → rezonanse BMG (dobór `size`),
* wachlarze i pasma → kierunki w FFT.

---

## Replikowalność (checklista)

* obraz wejściowy (nazwa/hasz),
* preset/filtr + parametry (YAML/GUI),
* seed RNG,
* screen HUD (amplitude/maska + 2 mapy diag),
* opis procedury (A/B, ROI, sweep),
* metryki (SSIM/PSNR/entropia/…) i wnioski (progi, relacje).

---

