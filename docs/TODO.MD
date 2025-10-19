# I. Audyt paczek i decyzje

## 1) `glx_catalog_v5.zip` (katalog referencyjny)

**Zawartość kluczowa (wyciąg):**

* `docs/bus.md`, `docs/egdb.md`, `docs/ci_ops.md`, `docs/gui.md`, `docs/observability.md`, `docs/pipelines.md`,
* `docs/GLX_Mosaika_Matematyka_v5.md`, `docs/12_mosaic_metrics.md`,
* `schemas/*.json` (m.in. `git.delta.ready.schema.json`, `publish.schema.json`, `fail.closed.schema.json`).

**Decyzja:**

* ✅ **Włączyć jako źródła kanoniczne/załączniki**:
  *`GLX_Mosaika_Matematyka_v5.md`*, *`12_mosaic_metrics.md`* → do **appendix** sekcji *Mosaic/Mathematics*.
  *`schemas/*.json`* → do `/spec/schemas` (SSOT).
* 🧩 **Scalić z obecnymi**: *`bus.md`, `egdb.md`, `observability.md`, `ci_ops.md`, `gui.md`, `pipelines.md`* — masz już świeże wersje w `docs/20_bus.md`, `docs/21_egdb.md`, `docs/70_observability.md`, `docs/50_ci_ops.md`, `docs/40_gui_app.md`, `docs/41_pipelines.md`. Treści z katalogu potraktujmy jako **materiał porównawczy**: fragmenty, które rozszerzają definicje lub mają ryciny, włączymy jako przypisy/ramki w bieżących dokumentach.
* ⚙️ **Do archiwum**: starsze wersje opisów repo/release (*`repo.md`*) — zachować w `/glx_catalog_v5/backup/` dla historii.

## 2) `core_docs.zip` (nowsze wersje Twoich nowych plików `docs/..`)

**Zawartość:**

* `docs/10_architecture.md`, `docs/11_spec_glossary.md`, `docs/12_invariants.md`, `docs/13_delta_algebra.md`, `docs/20_bus.md`, `docs/21_egdb.md`, `docs/41_pipelines.md`.

**Decyzja:**

* ✅ **Zostają**: nasze świeżo wygenerowane pliki są spójne z układem repo i już odwołują się do BUS/EGDB/Mozaiki/CI. Ten ZIP traktujemy jako **snapshot kontrolny** (w razie rollbacku).
* 🧩 **Drobne uzupełnienia**: do `docs/41_pipelines.md` dodamy odnośniki do `core/registry.py` i `core/steps/*` (z Twojej struktury repo), oraz do `gui/services/pipeline_runner.py` (ścieżka wywołania).

## 3) `analysis.zip`

**Zawartość (skrót):**

* `analysis/*.py` (metrics, diff, spectral, formats, exporters, ast_*),
* `analysis/analiza_progi.md`,
* artefakty `analysis/last/*` (pack/summary) i dzienniki commitów.

**Decyzja:**

* ✅ **Włączyć do dokumentacji**: *`analysis/analiza_progi.md`* — to bezpośrednio zasila rozdział o kalibracji progów (link do `docs/12_invariants.md` i nowego `docs/31_analytics.md`).
* 🧩 **Wkomponować jako “Examples”**: *`analysis/last/summary.md`*, *`analysis/last/autonomy/pack.md`* — dodamy jako **Appendix: Example Artifacts** w `docs/31_analytics.md`.
* ⚙️ **Do archiwum**: logi `analysis/logs/*.json` (warto zachować w repo pod `.glx/`/`backup/`, ale nie jako dokumentacja).

## 4) Pliki luzem w /mnt/data

* `GlitchLab_Arch_v5.md` — ✅ **Appendix** do `docs/10_architecture.md` (diagramy/terminologia spaść do przypisów).
* `GLXENCJE — MANIFEST v3.6 (FINAL).MD` — 🧩 **Wyciąg definicji** (nomenklatura GLXENCJE) do `docs/11_spec_glossary.md` (sekcja Słownik → blok *GLXENCJE* jako *namespace*).
* `DIFF_SUMMARY.md` — 🧩 **Zacytować w** `docs/13_delta_algebra.md` (część *Fingerprint & Histogram*).
* `README.MD` — ✅ pozostaje w repo; **sekcję Quickstart** zsynchronizujemy po scaleniu TOC.

---

# II. Porządek numeracji i ToC (z renamami)

Aby **wstawić nową gałąź “Analytics”** bez ruszania istniejących identyfikatorów, proponuję:

* **NOWY:** `docs/31_analytics.md` (między 30 a 40).
* **Renamy kosmetyczne** (dla przejrzystości, *zachowujemy stare pliki z krótkim bannerem “przeniesiono do …”*):

  * `docs/80_release.md` → `docs/82_release_and_channels.md`
  * `docs/90_playbooks.md` → `docs/92_playbooks.md`

> Dzięki temu **nie psujemy** istniejących linków (stare pliki zostają jako “stub” z przekierowaniem), a ToC układa się czytelnie:

```
00_overview.md
10_architecture.md
11_spec_glossary.md
12_invariants.md
13_delta_algebra.md
14_mosaic.md
20_bus.md
21_egdb.md
30_sast_bridge.md
31_analytics.md   <-- NOWE
40_gui_app.md
41_pipelines.md
50_ci_ops.md
60_security.md
70_observability.md
82_release_and_channels.md   <-- rename z 80
92_playbooks.md              <-- rename z 90
```

---

# III. (NOWE) `docs/31_analytics.md` — **pełna treść**

> **Cel:** scalić *warstwę analityczną* (analysis/), *kalibrację progów* oraz *powiązanie z AgentemAI* w jedną, operacyjną specyfikację. Dokument jest SSOT dla dynamiki metryk Δ, EWMA/MAD, t-digest, drift i HUD.

```markdown
# GlitchLab — Analytics Layer (Δ-Metrics, Calibration, Drift, AgentAI)

**Id:** docs/31_analytics.md  
**Zakres:** strumień metryk Δ, kalibracja progów (α/β/Z), detekcja driftu, integracja z AgentemAI, artefakty `.glx/*`, zapytania EGDB/EGQL, HUD (Spec Monitor).

---

## 1. Rola warstwy Analytics

Warstwa Analytics tworzy **żywy model zachowania repo** na osi czasu:
- destyluje **cechy Δ** (kod/obrazy/AST) z bieżących commitów/PR,
- utrzymuje **kalibrowane progi** α/β/Z (per moduł/warstwa),
- wykrywa **drift** i zarządza stanem „freeze/unfreeze” progów,
- zasila **HUD** (panele: *Spec Monitor*, *Delta Inspector*) oraz **AgentemAI** (RAG + reguły *self-healing*).

> **SSOT:** definicje metryk + reguł żyją w `/spec/invariants.yaml` oraz `/spec/schemas/*.json`  
> **Źródła danych:** moduły `analysis/*`, wyniki CI/Hooków, EGDB (event log).

---

## 2. Strumień metryk Δ (features)

### 2.1 Źródła

- `analysis/metrics.py` — entropia, gęstość krawędzi, RMS, itp. *(wejścia obrazu)*  
- `analysis/diff.py` — SSIM/PSNR/delta-heatmapy *(zmiany obrazów)*  
- `analysis/spectral.py` — FFT + pasma *(częstotliwości)*  
- `analysis/formats.py` — heurystyki JPEG/PNG *(artefakty kompresji)*  
- `analysis/ast_*` (`ast_index.py`, `ast_delta.py`) — indeks AST, tokeny Δ (we współpracy z `delta/tokens.py`)  
- `core/metrics/*` — fallback i metryki rdzenia  
- `bench/*` — pomiary A/B (opcjonalnie zasilają EGDB)

### 2.2 Feature Vector (przykładowy)

Każdy commit/PR → wiersz `Δ_features` (klucze zależne od repo):
```

{
"sha": "...",
"module": "analysis|core|gui|mosaic|filters|...",
"ΔLOC": 18,
"ΔImports": +3,
"ΔCC": +2.0,
"ΔTests": -1,
"SSIM_drop": 0.07,
"PSNR_drop": 1.9,
"ΔFanIn": +1,
"ΔFanOut": 0,
"Token:MODIFY_SIG": 1,
"Token:ΔTYPE_HINTS": 3,
...
}

```

### 2.3 Artefakty

- `.glx/metrics.parquet` — przyrostowy log cech (commit→features)  
- `.glx/delta_report.json` — tokeny Δ + fingerprint + score + naruszenia I*  
- `.glx/spec_state.json` — snapshot α/β/Z + EWMA/MAD + stan freeze  
- **Appendix (opcjonalnie)**: `mosaic_*.png` — wizualizacja Δ na mozaice (jeśli dotyczy)

---

## 3. Kalibracja progów (α/β/Z)

**Założenia:** progi są **regułami**, nie stałymi liczbami. Są liczone **per moduł** z historii Δ (okno kroczące).

### 3.1 Estymatory

- **EWMA**: `μ_t = λ·x_t + (1-λ)·μ_{t-1}` (domyślnie `λ∈[0.1,0.3]`)  
- **MAD (robust σ)**: `σ ≈ 1.4826·MAD`  
- **Kwantyle t-digest / P²**: `α=Q(0.90)`, `β=Q(0.95)`, `Z=Q(0.99)` (per cecha i/lub score)

### 3.2 Score i bramki

`score = wᵀ·z(Δ_features)` (standaryzowane z użyciem EWMA/MAD).  
**Gates (CI/BUS):**
- `score ≤ α` → *auto-merge okno techniczne* (jeśli polityka na to pozwala),
- `α < score ≤ β` → *wymagany review*,
- `β < score ≤ Z` → *twardy block*,
- `score > Z` → *NO-GO* + alert.

> Wagi `w` na starcie ręczne (w `/spec/invariants.yaml`), później **uczone** z historii (logistic/XGBoost).

---

## 4. Drift i stan „Freeze”

**Sygnał detekcji:** Page-Hinkley / ADWIN na `score`/wybranych cechach.  
**Reakcja:**  
- publikacja `spec.freeze.start` (BUS),  
- zamrożenie aktualnych progów na N commitów,  
- zbieranie etykiet (czy PR bezproblemowy),  
- `spec.freeze.end` po stabilizacji i **re-kalibracja** z nowymi priors.

HUD (Spec Monitor) pokazuje **historię progów**, okresy freeze i „dlaczego”.

---

## 5. Integracja z AgentemAI (RAG + Healer)

**Kontekst RAG** (dla propozycji napraw/diagnoz):  
- Top-K zdarzeń EGDB (ostatnie N minut/godz.) dla `sha`/gałęzi,  
- `Δ_features` + tokeny `delta/tokens.py`,  
- mapy `Φ(Δ_AST) ↔ Δ_MOZ` (komutacja dla I3),  
- polityki z `security/sast_bridge.py` i `spec/invariants.yaml`.

**Przepływ (skrótem):**
1. `test.fail | build.fail | invariants.violation` → BUS  
2. AgentAI pobiera **kontekst** (EGDB + Δ + spec_state)  
3. Tworzy `heal.proposal` (patch/test/config) + **dowody**  
4. `guard` sprawdza I1–I4 + polityki ryzyka  
5. sandbox/CI/mutation/canary → `heal.verify.*`  
6. wynik → `heal.applied | heal.rejected | rollback`

---

## 6. EGDB / EGQL — minimalny zestaw widoków

**Widoki przykładowe (`views.sql`):**
- `vw_delta_over_time(module)` — trendy score/cech w oknie  
- `vw_gate_outcomes()` — outcome vs próg (ROC-like)  
- `vw_drifts()` — rejestr freeze i przyczyn

**Zapytania EGQL (przykłady):**
```

-- ostatnie zdarzenia dla SHA, z naruszeniami I*
events.where(sha=$sha)
.join('invariants.violation')
.last(200)

```

---

## 7. HUD (Spec Monitor + Delta Inspector)

- **Spec Monitor:** α/β/Z per moduł, stan freeze, drift, „explain why blocked”.  
- **Delta Inspector:** histogram tokenów, energia ΔZ, fingerprint PR, linki do linii.

---

## 8. API i kontrakty

- **BUS topics (subset):**  
  `metrics.delta.upsert`, `invariants.thresholds.update`,  
  `spec.freeze.start|end`, `invariants.violation`,  
  `heal.proposal`, `heal.verify.start|done`, `heal.applied|rollback`.

- **Schematy (`/spec/schemas`):**  
  `git.delta.ready.schema.json`, `publish.schema.json`, `fail.closed.schema.json` (dziedziczone z katalogu v5).

---

## 9. Operacja w CI/Hookach

- **pre-commit**: szybkie cechy Δ z dotkniętych plików + mini-score,  
- **post-commit**: pełny wektor cech → `.glx/metrics.parquet` + update `spec_state.json`,  
- **PR Action**: tokeny Δ + `delta_report.json` → komentarz do PR + gate wg progów.

---

## 10. Akceptacja i metryki sukcesu

- **Coverage Analytics:** ≥ 90% commitów ma poprawnie zapisane `Δ_features`.  
- **Stabilność progów:** drift ≤ 1/tydzień per moduł po 2 tyg. kalibracji.  
- **MTTR-Sec/Quality:** spadek mediany o ≥ 20% po włączeniu gatingu.  
- **FP-rate (gates):** ≤ 10% po pierwszym miesiącu (z zamrożeniami).

---

## Appendix A — Mapowanie modułów

- `analysis/metrics.py`, `diff.py`, `spectral.py`, `formats.py`, `exporters.py`  
- `delta/tokens.py`, `delta/fingerprint.py`, `delta/features.py`  
- `spec/calibrate.py`, `spec/loader.py`  
- `io/artifacts.py` (zapis `.glx/*`)  
- `security/sast_bridge.py` (wpływa na score/priorytety)

## Appendix B — Przykładowe artefakty

Dołączone z `analysis/last/*`: `summary.md`, `autonomy/pack.md` (przykłady raportów).
```

---

# IV. Podsumowanie działań

**Co dołączamy teraz:**

* `docs/31_analytics.md` (powyżej — kompletna treść).
* Appendixy: włączymy *wyciągi* z `GLX_Mosaika_Matematyka_v5.md`, `12_mosaic_metrics.md` do `docs/14_mosaic.md`;
  fragmenty `DIFF_SUMMARY.md` do `docs/13_delta_algebra.md`.

**Co regenerujemy w następnych pętlach:**

* Drobne uzupełnienia `docs/41_pipelines.md` (ścieżki wywołań/registry).
* Dodatkowe sekcje ilustracyjne do `docs/40_gui_app.md` (zrzuty HUD i link do `gui/services/pipeline_runner.py`).

**Co do archiwum (nie publikujemy jako docs):**

* Dzienniki `analysis/logs/*.json`, snapshoty packów w `analysis/last/*` (zostają jako artefakty `.glx/backup`).

**Rename (zostawiamy stuby-redirecty):**

* `docs/80_release.md` → `docs/82_release_and_channels.md`
* `docs/90_playbooks.md` → `docs/92_playbooks.md`


