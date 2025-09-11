# analysis — controlled glitch for inference

> **Teza praktyczna:** artefakt to sygnał diagnostyczny. Jeśli potrafisz sterować *gdzie* (maski) i *jak mocno* (amplitude) występuje, możesz wnioskować o strukturze danych oraz o relacjach między transformacjami.

---

## 0. zakres i założenia

- Pracujemy na obrazach 2D (RGB/RGBA).  
- **Regulatory**: maski ROI (`ctx.masks`) i pole amplitudy (`ctx.amplitude`).  
- **Obserwable**: wynik końcowy + mapy diagnostyczne z HUD (`ctx.cache`), np. `acw_mag`, `bmg_dx/dy`.  
- **Cel**: projektować eksperymenty, które dają **mierzalne** wnioski o:
  1) strukturze obrazu (anizotropia, blokowość, pasma),  
  2) relacjach między filtrami (komutacja A/B, wrażliwość na kolejność),  
  3) stabilności i progach („przejścia fazowe” parametrów).

---

## 1. model i notacja

- Obraz: \( I:\Omega\subset\mathbb{Z}^2\to\mathbb{R}^3 \).  
- Maska: \( M:\Omega\to[0,1] \).  
- Amplitude: \( A:\Omega\to[0,1] \).  
- Krok: \( I' = F_\theta(I; M, A) \).  
- Pipeline: \( I^{(k+1)} = F^{(k)}_{\theta_k}(I^{(k)}; M_k, A_k) \).  
- Artefakt (lokalna zmiana): \( E^{(k)} = I^{(k+1)} - I^{(k)} \).

**pozytyw** — cecha stabilna względem klasy deformacji (artefakt mały w ROI).  
**negatyw** — cecha rezonuje z deformacją (artefakt duży / strukturalny w ROI).

---

## 2. projekt eksperymentu (szkielet)

1. **hipoteza**: np. „obraz ma anizotropię konturową” / „widać ślad blokowości”.  
2. **konfiguracja**: filtr/preset, ROI (maski), amplitude.  
3. **eksperyment**: sweep parametrów, warianty ROI, seedy.  
4. **pomiary**: SSIM/PSNR, entropia gradientu, histogram `dx/dy`, różnice A/B.  
5. **wniosek**: stabilność, progi, relacje (komutacja).

---

## 3. protokoły badawcze

### 3.1. test komutacji (A/B)
Sprawdza, czy kolejność filtrów ma znaczenie.

```

A: I → F1 → F2 → I\_A
B: I → F2 → F1 → I\_B
Δ = 1 − SSIM(I\_A, I\_B)   # ewentualnie: ||I\_A − I\_B||₁ / N

````

**Interpretacja**  
- Δ małe → F1 i F2 prawie komutują (sprzężenie słabe).  
- Δ duże → silne sprzężenie transformacji (kolejność istotna).

**Praktyka**  
- F1: `anisotropic_contour_warp` (ACW), F2: `block_mosh_grid` (BMG).  
- Zmieniaj `ACW.strength/iters` i `BMG.size/max_shift`; notuj Δ.

---

### 3.2. roi-scan (przenikalność i lokalność)
- Ustaw `Amplitude.kind=mask` → `mask_key=ROI_A` i zastosuj F.  
- Zmień na `ROI_B` i powtórz.  
- Porównaj uśredniony |artefakt| w ROI vs. poza ROI.

**Wniosek**  
- Artefakt ograniczony do ROI → filtr lokalny i zgodny z maską.  
- „Wyciek” poza ROI → filtr nielokalny / zależy od sąsiadów.

---

### 3.3. sweep parametrów (progi i przejścia)
- Jednowymiarowy: `strength ∈ [0.2…2.0]` przy stałych pozostałych.
- Dwuwymiarowy: siatka `(size, p)` w BMG.

**Szukaj** „kolan” krzywych (nagły wzrost entropii / spadek SSIM): to **progi** istotnych zmian.

---

### 3.4. seed sweep (losowość kontrolowana)
- Ten sam preset, różne `seed`.  
- Licz wariancję metryk i zbieżność map HUD.

**Wniosek**  
- Mała wariancja → efekt wynika ze struktury obrazu, nie z RNG.  
- Duża → mocny komponent losowy (zredukuj `p`, podnieś wpływ amplitude/mask).

---

## 4. metryki i wskaźniki

> HUD daje mapy **wizualne**; do ilościowych porównań eksportuj wyniki i licz offline.

**Podstawowe**
- **SSIM** (0..1; 1 = identyczne) — strukturalna podobność.  
- **PSNR** (dB) — siła zniekształcenia względem szumu białego.  
- **L1 / L2** — średnia bezwzględna / kwadratowa różnica (proste, globalne).

**Strukturalne**
- **Entropia gradientu** — wzrost = większa złożoność lokalnych krawędzi.  
- **Energia kierunkowa** (FFT lub filtry Gabor) — pasma/kierunki pobudzone przez filtr.

**Specyficzne dla filtrów**
- BMG: histogram `bmg_dx/dy` (rozkład przesunięć), rezonans na `size`.  
- ACW: rozkład `acw_mag` vs. czytelność semantyki po `iters↑`.

**ROI / zależność**
- Średni |artefakt| w ROI vs. poza ROI (+ różnica/ratio).  
- MI-proxy: różnica gęstości artefaktu między ROI a tłem (zamiast pełnej informacji wzajemnej).

---

## 5. wnioski z geometrii artefaktów

### 5.1. anizotropia (acw)
- Jeśli po 2–3 iteracjach ACW obiekty (np. litery) są czytelne → cecha **stabilna tangencjalnie** (treść „leży” wzdłuż konturów).  
- `acw_mag` wskazuje, gdzie gradient podtrzymuje strukturę; koreluj z ROI.

### 5.2. blokowość / ślad kompresji (bmg)
- Rezonans przy konkretnym `size` (np. 8/16) → możliwy ślad JPEG/skalowania.  
- Rozkład `bmg_dx/dy` skupiony wokół 0, ale z ogonami → „miękkie” regiony podatne na przesunięcia.

### 5.3. pasma i aliasy
- Po filtrach zobacz pasma (kierunki) w wysokich częstotliwościach: dominujący kierunek = dominująca geometria tekstur.

---

## 6. procedury „krok po kroku” (gui)

### 6.1. test anizotropii (acw)
1) Open image…  
2) Filter: `anisotropic_contour_warp` → `strength=1.2`, `iters=2`, `edge_bias=0.5`, `smooth=0.7`  
3) Apply single → sprawdź `acw_mag` i czytelność obiektów.  
4) Zwiększ `iters` do 3–4; gdy czytelność spada — zanotuj próg.

### 6.2. test blokowości (bmg)
1) Filter: `block_mosh_grid` → `size=16/24/32`, `p=0.5`, `max_shift=28`, `mix=0.9`  
2) Apply single → obserwuj `bmg_dx/dy`.  
3) Zmieniaj `size`; jeśli przy którymś wzór „skacze w oczy”, notuj rezonans.

### 6.3. roi-scan
1) Load mask… (np. tylko tekst) → `Amplitude.kind=mask`, `mask_key=<nazwa>`  
2) Filter (dowolny) z `mix≈0.7` → artefakt powinien pozostać w ROI.  
3) Zmień maskę na tło i powtórz; porównuj intensywność.

### 6.4. komutacja A/B
1) Zastosuj `ACW → BMG` (zapisz wynik).  
2) Otwórz ponownie wejście i wykonaj `BMG → ACW`.  
3) Oceń Δ (subiektywnie / offline SSIM). Duża Δ ⇒ silne sprzężenie.

---

## 7. raport i replikowalność (checklista)

- [ ] nazwa/hasz obrazu wejściowego  
- [ ] preset/filtr + **parametry** (YAML/GUI)  
- [ ] **seed** RNG  
- [ ] screen HUD: amplitude, mask, diagnostyki (2 mapy)  
- [ ] opis procedury (A/B, sweepy, ROI)  
- [ ] metryki (SSIM/PSNR/entropia itd.) i wnioski (progi, relacje)

---

## 8. ograniczenia i pułapki

- duże obrazy = czas (NumPy/CPU). Strojenie rób na downsamplu, „final pass” na pełnej.  
- maski nietrafione rozmiarem → artefakty na krawędziach. Dopasuj precyzyjnie.  
- zbyt małe `mix/p/strength` lub amplitude ≈ 0 → „brak efektu” (fałszywa diagnoza).  
- HUD to wizualna diagnostyka; do liczb eksportuj obrazy i licz offline.

---

## 9. przykładowe receptury (yaml)

### 9.1. anizotropia + lekki blokowy probe
```yaml
edge_mask: { thresh: 60, dilate: 6, ksize: 3 }
amplitude: { kind: perlin, scale: 96, octaves: 4, strength: 1.0 }
steps:
  - { name: anisotropic_contour_warp, params: { strength: 1.2, iters: 2, edge_bias: 0.5, smooth: 0.7 } }
  - { name: block_mosh_grid,        params: { size: 24, p: 0.35, max_shift: 24, mix: 0.75 } }
````

### 9.2. roi-scan (celowane zaburzanie)

```yaml
amplitude: { kind: mask, strength: 1.0, mask_key: text_roi }
steps:
  - { name: anisotropic_contour_warp, params: { strength: 1.0, iters: 2, use_amp: true } }
```

### 9.3. komutacja A/B

```yaml
# F1
steps:
  - { name: anisotropic_contour_warp, params: { strength: 1.1, iters: 2 } }
  - { name: block_mosh_grid,         params: { size: 24, p: 0.4, max_shift: 28, mix: 0.85 } }
# F2: zamień kolejność kroków i porównaj
```

---

## 10. notatki implementacyjne (dla pomiarów offline)

* SSIM/PSNR (np. `scikit-image`) — licz na wersjach 8-bit lub znormalizowanych.
* Dla ROI licz metryki osobno w masce i poza maską (np. średni |Δ|, entropia gradientu).
* Histogram `dx/dy` (z BMG) i gęstość wyborów (mapa `bmg_select`) traktuj jako dane ilościowe.

---

## 11. skrócona „matryca decyzji”

| pytanie                      | użyj                        | co mierzyć / obserwować           |
| ---------------------------- | --------------------------- | --------------------------------- |
| czy treść jest anizotropowa? | `anisotropic_contour_warp`  | czytelność po `iters↑`, `acw_mag` |
| ślad kompresji / blokowość?  | `block_mosh_grid`           | rezonans na `size`, `bmg_dx/dy`   |
| gdzie działa efekt?          | `Amplitude.kind=mask` + ROI | artefakt w ROI vs. poza ROI       |
| czy kolejność ma znaczenie?  | test A/B                    | Δ = 1−SSIM(A,B) / L1              |

---

## 12. słownik (operacyjny)

* **anizotropia** — własność zależna od kierunku (stabilność „wzdłuż” konturów).
* **blokowość** — regularność wynikająca z siatki (kompresja, skalowanie).
* **komutacja** — zamiana kolejności filtrów nie zmienia wyniku (rzadkie).
* **próg/przejście** — wartość parametru, po której pojawia się jakościowo nowa struktura artefaktu.

