# GlitchLab — Spec: Inwarianty I1–I4 i bramki decyzyjne · vNext

> Ten dokument **normatywnie** definiuje inwarianty jakości GlitchLab (I1–I4), sposób ich **pomiaru**, **kalibracji** (α/β/ζ), **agregacji** do decyzji oraz **artefakty** i **zdarzenia** BUS. Jest częścią SSOT razem z `docs/11_spec_glossary.md` i plikami w `/spec/`.

---

## 0) Zakres i konwencje

* Słowa **MUST / MUST NOT / SHOULD / MAY** są normatywne (RFC 2119).
* Wszystkie progi i parametry są wersjonowane w `/spec/invariants.yaml` (żywy dokument).
* Czas w ISO-8601 (UTC, `Z`). Skale metryk w `0..1`, chyba że określono inaczej.
* Kotwica ścieżek: **git root**. Artefakty odkładamy do `.glx/`.

---

## 1) Streszczenie (TL;DR)

| Kod | Nazwa                   | Co mierzy?                                  | Główne wejścia                  | Decyzja / próg domyślny |
| --- | ----------------------- | ------------------------------------------- | ------------------------------- | ----------------------- |
| I1  | Energia zmiany **ΔZ**   | *Ile* i *jakiej jakości* zmiany wprowadzono | Δ-tokeny, cechy S/H, wagi `w`   | α/β/ζ (kwantyle)        |
| I2  | Kontrakty API/portów    | Stabilność publicznych interfejsów          | AST (sygnatury), mapy portów    | HARD gate               |
| I3  | Komutacja **Φ** z **Δ** | Spójność AST-diff ↔ Mozaika (Δ_MOZ)         | Φ(Δ_AST), Δ_MOZ, norma/IoU/SSIM | HARD gate (τ)           |
| I4  | Stabilność obserwowalna | Testy, wydajność, jakość (SSIM/PSNR)        | CI/testy/telemetria/metyki GUI  | α/β (soft → hard)       |

**Kolejność oceny (lexicographic guard):** najpierw I2/I3 (HARD), potem I4, na końcu I1.

---

## 2) I1 — Energia zmiany ΔZ

### 2.1 Definicja

Niech `z(Δ) ∈ ℝ^k` będzie wektorem cech z Δ-tokenów i kontekstu (por. słownik). Energia:

```
ΔZ_raw = wᵀ · z(Δ)              # w - wagi (per moduł/warstwa), z - znormalizowane cechy
ΔZ     = clip(norm01(ΔZ_raw))   # normalizacja do [0,1] wg bieżącego stanu spec
```

**Źródła cech (przykładowe):**

* histogram tokenów (`ADD_FN`, `MODIFY_SIG`, `ΔIMPORT`, `ΔCC`, …),
* miary z `analysis.*` (gęstości/FFT/histogramy) jeśli dotyczy artefaktów wizualnych,
* zmiany testów/pokrycia, fan-in/out, wielkość diffa (ΔLOC) z karami za hot-spoty.

### 2.2 Progi i decyzje

Progi **per moduł** (np. `core`, `analysis`, `gui`) kalibrowane strumieniowo (§5):

* `α(module)` – q90, **ACCEPT** jeśli `ΔZ ≤ α`.
* `β(module)` – q95, **REVIEW** jeśli `α < ΔZ ≤ β`.
* `ζ(module)` – q99, **BLOCK** jeśli `ΔZ > β` (a szczególnie **HARD BLOCK** jeśli `ΔZ ≫ ζ`).

> **SHOULD:** dodatkowa kara (additive) za zdarzenia bezpieczeństwa (np. CRITICAL z SAST-Bridge).

### 2.3 Artefakty

* `.glx/delta.report.v1` → `score: { z, w, ΔZ, α, β, ζ, decision }`
* `.glx/spec_state.v1` → aktualne kwantyle i EWMA/MAD.

---

## 3) I2 — Kontrakty API / portów

### 3.1 Definicja

**Kontrakt publiczny** to zbiór **portów** (symbole eksportowane przez moduły/warstwy oznaczone jako publiczne).

**Naruszeniem** I2 jest zmiana, która bezpośrednio łamie kompatybilność:

* `MODIFY_SIG`: zmiana arity/kolejności/typów **bez** adaptera,
* `RENAME_FN/CLASS` w przestrzeni publicznej bez aliasu/deprecation,
* `MOVE_FILE/MOVE_SYMBOL` łamiące ścieżki importu bez warstwy zgodności,
* **usunięcie** portu publicznego bez polityki deprecacji.

Dopuszczalne:

* dodanie parametrów opcjonalnych z wartością domyślną,
* zmiana typów zawężająca niejednoznaczność (covariant return, contravariant args) z testem kompatybilności,
* **adapter** (forwarder) + `@deprecated` na starym symbolu w oknie `T_depr`.

### 3.2 Reguła decyzji

* **HARD FAIL** jeśli znaleziono naruszenie bez aktywnej polityki deprecacji/adaptera.
* **REVIEW** jeśli zmiana jest kompatybilna warunkowo (nowy parametr opcjonalny) — wymaga potwierdzenia.
* **PASS** w przeciwnym razie.

### 3.3 Artefakty i BUS

* `.glx/delta.report.v1.contracts` — lista portów dotkniętych + werdykty.
* Event: `invariants.violation { kind:"I2", items:[…] }`.

---

## 4) I3 — Komutacja Φ z Δ

### 4.1 Definicja

Wymagamy, by odwzorowanie zmian AST przez **Φ** było spójne z obserwowaną różnicą w Mozaice:

```
E = Dist( Φ(Δ_AST), Δ_MOZ )
I3 holds  ⟺  E ≤ τ(module)
```

Gdzie `Dist` to **norma** między polami intensywności na spójnej siatce:

* IoU (intersection-over-union) binarnej maski obszarów Δ (1-IoU jako odległość), **lub**
* SSIM/L2 na znormalizowanych heatmapach (większość przypadków), **lub**
* EMD (earth mover’s distance) dla przesuniętych bloków (opcjonalnie cięższa).

Domyślne `τ(module)` kalibruje się jak q95 rozkładu `E` dla „dobrych” commitów.

### 4.2 Procedura (pseudo)

```python
# Wejścia: tokens (Δ_AST), mosaic_prev.png, mosaic_curr.png
mask_ast = phi_delta_mask(tokens, grid=G)       # Φ(Δ_AST) → maska na siatce
mask_moz = mosaic_diff_mask(prev, curr, grid=G) # Δ_MOZ    → maska/heatmapa
E = 1 - IoU(binarize(mask_ast), binarize(mask_moz))  # lub SSIM/L2
verdict = (E <= tau(module))
```

### 4.3 Decyzja

* **HARD FAIL** jeśli `E > τ` (niespójność percepcji systemu),
* **REVIEW** blisko progu (`τ < E ≤ τ + ε`),
* **PASS** w przeciwnym razie.

> Uwaga: przy zmianach czysto dokumentacyjnych (brak Δ_AST istotnego) **I3 jest N/A**.

---

## 5) I4 — Stabilność obserwowalna (testy, wydajność, jakość)

### 5.1 Definicja

I4 agreguje sygnały **runtime/CI** istotne dla użytkownika:

* **Testy:** pokrycie (globalne i per moduł), liczba testów, wynik `pytest` (flaky/xfail).
* **Wydajność:** czas krytycznych kroków (P95/P99), zużycie pamięci dla profili.
* **Jakość wizualna:** SSIM/PSNR dla referencyjnych presetów GUI/HUD.

### 5.2 Progi

* `coverage_drop ≤ δ_cov(module)` – dopuszczalny spadek (np. 1–2 p.p.).
* `perf_regress ≤ δ_perf(module)` – dopuszczalny wzrost P95 (np. +5%).
* `SSIM ≥ ssim_min(preset)` / `PSNR ≥ psnr_min(preset)` dla próbek demo.

**Kalibracja** identyczna jak w I1 (EWMA/MAD + kwantyle), ale per-metryka (§6).

### 5.3 Decyzja

* **REVIEW** jeśli jedna z metryk przekracza `α/β`,
* **HARD FAIL** jeśli przekracza `ζ` albo testy **nie przechodzą**.

---

## 6) Kalibracja progów (α/β/ζ) i drift

### 6.1 Silnik (strumieniowo, bez ciężkiej ML)

Dla każdej metryki `x_t` w module `m` utrzymujemy:

```
EWMA:   μ_t = λ·x_t + (1-λ)·μ_{t-1}
MAD:    σ̂_t ≈ 1.4826 · median(|x_t - median_window|)
Quant:  q90, q95, q99 via t-digest (albo P²) na ostatnim oknie N
```

* **Domyślne λ**: 0.1–0.2; **okno N**: 90 commitów lub 30 dni (co pierwsze).
* **Freeze on drift**: wykrywanie dryfu (Page-Hinkley/ADWIN). W stanie `frozen:true` progi nie zmieniają się przez `N_freeze` commitów; raport zawiera przyczyny.

### 6.2 Artefakty

* `.glx/spec_state.v1`:

  ```json
  {
    "modules": {
      "core": { "alpha": 0.31, "beta": 0.44, "zeta": 0.61, "drift": { "frozen": false } },
      "analysis": { ... }
    },
    "metrics": {
      "coverage": { "alpha": 0.78, "beta": 0.76, "zeta": 0.72, "lambda": 0.15, "N": 90 }
    }
  }
  ```

* Event: `spec.thresholds.update`.

---

## 7) Agregacja decyzji (gate policy)

### 7.1 Reguła łączna (deterministyczna)

1. **I2** → jeśli **FAIL** → **BLOCK (hard)**.
2. **I3** → jeśli **FAIL** → **BLOCK (hard)**.
3. **I4** → jeśli **ζ** przekroczone lub testy padły → **BLOCK**, jeśli tylko `β` → **REVIEW**.
4. **I1** → mapuj `ΔZ` na {ACCEPT/REVIEW/BLOCK} wg progów modułu.
5. **Wzmocnienie SAST** (opcjonalnie): jeśli w kolejce PQ są **CRITICAL** bez fixu → podnieś decyzję o 1 poziom (ACCEPT→REVIEW→BLOCK).

### 7.2 Próg wielomodułowy

Przy PR dotykającym wielu modułów:

* decyduje **max(severity)** z modułów,
* **SHOULD** zastosować wagę rozmiaru zmian per moduł (udział `ΔLOC` lub energii w module).

---

## 8) Konfiguracja: `/spec/invariants.yaml` (schemat)

```yaml
schema: invariants.v1
modules:
  core:
    i1:
      weights: { ADD_FN: 0.08, MODIFY_SIG: 0.22, DEL_FN: 0.35, ΔIMPORT: 0.05, ΔCC: 0.15 }
      penalties:
        security_critical: 0.20
        hot_spot: 0.10
    i3:
      dist: ssim          # ssim | iou | l2 | emd
      tau: auto           # auto == q95(spec_state), fallback: 0.15
    i4:
      coverage_drop_pp:
        alpha: 0.01
        beta:  0.02
        zeta:  0.05
      perf_regress_p95:
        alpha: 0.03
        beta:  0.05
        zeta:  0.10
      visual:
        ssim_min: 0.96
        psnr_min_db: 35.0
  analysis: { ... }       # analogicznie
globals:
  lambda: 0.15            # EWMA
  window_commits: 90
  drift:
    detector: page_hinkley
    freeze_commits: 20
public_api:
  deprecation_window_commits: 60
  allow_optional_params: true
overrides:
  - when: { path: "^tests/.*" }
    i1: { weights: { ΔCC: 0.0 }, penalties: { security_critical: 0.0 } }
```

---

## 9) Zdarzenia BUS i logika HUD

* `invariants.violation` — `{ sha, module, kind: "I2|I3|I4|I1", details, severity }`
* `delta.tokens` — lista tokenów do panelu „Delta Inspector”
* `spec.thresholds.update` — aktualizacja HUD „Spec Monitor”
* `security.alert` — sygnał z SAST-Bridge (powiązany z I1/I4)

**GUI/HUD MUST** pokazywać: **„dlaczego”** (tokenujące źródła, progi, bieżące wartości) i linki do artefaktów `.glx/*`.

---

## 10) Testy kontraktowe i scenariusze

### 10.1 Jednostkowe (fast)

* **I1:** mapowanie tokenów → ΔZ (snapshot histogramu i ΔZ).
* **I2:** zestaw sygnatur (before/after) → PASS/REVIEW/BLOCK.
* **I3:** syntetyczne maski AST/MOZ → E i werdykt.
* **I4:** symulacja regresji pokrycia / P95 → decyzja.

### 10.2 Integracyjne (CI)

* PR z `MODIFY_SIG` bez adaptera → **BLOCK (I2)**.
* PR o `ΔZ` w q92 → **REVIEW (I1)**, HUD wyjaśnia wkład tokenów.
* PR zmieniający wizualia; SSIM=0.93<0.96 → **REVIEW/BLOCK (I4)**.
* PR z niezgodną Mozaiką (E>τ) → **BLOCK (I3)**.

---

## 11) Edge-cases i wyjątki (kontrolowane)

* **Generated/vendor**: ścieżki w `overrides` mogą obniżać wagi/wyłączać I3/I4.
* **Docs-only commits**: I3=N/A; I1 liczone wyłącznie z tokenów dokumentacyjnych (wagi bliskie 0).
* **Hotfix emergency**: label `override:I*:<ticket>` → **MAY** obniżyć poziom decyzji **o 1** (ACCEPT nigdy). Wymagane `reason` i ticket.

---

## 12) Interfejsy artefaktów (skrót schematów)

### 12.1 `.glx/delta.report.v1` (fragment)

```json
{
  "sha": "…",
  "modules": ["core"],
  "tokens": [{ "kind": "MODIFY_SIG", "file": "core/pipeline.py", "line": 210 }],
  "fingerprint": "fp:sha256:…",
  "scores": {
    "I1": { "ΔZ": 0.41, "α": 0.32, "β": 0.45, "ζ": 0.61, "decision": "REVIEW" },
    "I2": { "status": "PASS", "details": [] },
    "I3": { "E": 0.08, "τ": 0.12, "status": "PASS" },
    "I4": { "coverage_drop_pp": 0.0, "perf_regress_p95": 0.01, "status": "PASS" }
  },
  "decision": "REVIEW",
  "explain": ["I1 above α due to MODIFY_SIG (+0.22), ΔCC (+0.06)"]
}
```

### 12.2 `.glx/spec_state.v1` (fragment)

```json
{
  "modules": {
    "core": { "alpha": 0.32, "beta": 0.45, "zeta": 0.61, "drift": { "frozen": false } }
  },
  "metrics": { "coverage": { "alpha": 0.78, "beta": 0.76, "zeta": 0.72 } }
}
```

---

## 13) Zgodność, wersjonowanie i migracje

* Zmiana semantyki inwariantu → **inkrementuj** `schema` (`invariants.vN`), dodaj migrator stanów.
* GUI/HUD **SHOULD** wspierać co najmniej `vN` i `vN-1`.
* **Reproducibility:** przy ocenie PR zapisz snapshot `invariants.yaml` do artefaktu raportu.

---

## 14) Checklist wdrożeniowy (must-have)

* [ ] `/spec/invariants.yaml` z wartościami bazowymi i `overrides`.
* [ ] Implementacja ekstrakcji `z(Δ)` i wag `w` per moduł.
* [ ] Tokenizer Δ i mapowanie Φ(Δ_AST) → maska (I3).
* [ ] Integracja testów/telemetrii do I4 (coverage, P95, SSIM/PSNR).
* [ ] Akcje CI publikujące `.glx/delta.report.v1` i `.glx/spec_state.v1`.
* [ ] Panele HUD: „Delta Inspector” + „Spec Monitor”.

---

## 15) Załączniki i zależności

* **Współzależne dokumenty:**
  `docs/11_spec_glossary.md`, `docs/10_architecture.md`, `docs/00_overview.md`.

* **Pliki konfiguracyjne i schematy:**
  `/spec/invariants.yaml`, `/spec/schemas/*.json`.

---

**Status pliku:** ✅ **Final (Spec – Inwarianty)**
Zmiana któregokolwiek z inwariantów, progu lub algorytmu kalibracji **MUST** zostać odzwierciedlona w tym pliku i w `/spec/invariants.yaml`.
