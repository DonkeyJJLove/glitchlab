# GlitchLab — architektura platformy (vNext)

> Ten dokument opisuje architekturę **GlitchLab** jako spójnej platformy **IDE/APP + analityka + generacja + self-healing**. Jest punktem odniesienia dla implementacji, testów, CI/CD oraz dokumentacji niższego poziomu.

---

## 1. Zakres i zasady projektowe

**Cele:**

* Jedno narzędzie do codziennej pracy zespołu Human-AI: edycja, inspekcja, generacja i leczenie kodu.
* **Delta-first**: wszystko kręci się wokół zmian (Δ), nie „migawki stanu”.
* **Spójność matematyczna**: sprzężenie **Φ : AST → Mozaika** i **Ψ : Mozaika → AST**, egzekwowane przez inwarianty **I1–I4**.
* **SSOT**: `/spec` jako żywe źródło prawdy (glossary, invariants.yaml, schemas).
* **Jedna pętla → jeden artefakt**: minimalny krok, przejrzyste artefakty `.glx/*`, łatwe cofnięcie.

**Prymitywy:**

* **Soczewka GLX (Lens)** — metaopis generacji/inspekcji pliku.
* **EventBus** — proste, synchroniczne PUB/SUB w procesie (z opcją IPC).
* **EGDB** — relacyjna baza zdarzeń/metryk (SQLite lub Postgres).
* **Inwarianty I1–I4** — bramki jakości/ryzyka w CI i w GUI.

---

## 2. Topologia (wysoki poziom)

```
┌─────────────────┐     events      ┌───────────────┐
│   GUI / APP     │◀───────────────▶│    EventBus   │
│  (HUD + IDE)    │                 └──────┬────────┘
│ DeltaInspector  │                        │
│ SpecMonitor     │                        │
└───────▲─────────┘                        │
        │                                  │
        │ queries/artifacts                │
        │                                  ▼
  ┌─────┴────────┐    Δ / Φ/Ψ / I1–I4   ┌──────────────┐
  │  Core        │  ┌──────────────────▶ │   Analysis   │
  │  pipeline    │  │                    │  (metrics)   │
  │  invariants  │  │ ◀──────────────────└──────────────┘
  │  mosaic/AST  │  │    maps/heat       ┌──────────────┐
  └─────┬────────┘  │                    │ Delta        │
        │           │ tokens,fingerprint │ (tokens/FP)  │
        │           │                    └──────────────┘
        │           │                    ┌──────────────┐
        │           └──────────────────▶ │ Security     │
        │                findings/FC     │ (SAST-Bridge)│
        │                                └──────────────┘
        │   artifacts (.glx/*)
        ▼
┌──────────────────┐      SQL/Views      ┌───────────────┐
│  IO / Artifacts  │  ◀────────────────▶ │    EGDB       │
│  (.glx, export)  │                    │ (Relational)  │
└──────────────────┘                    └───────────────┘
```

---

## 3. Pakiety i odpowiedzialności

### 3.1 `gui/` (APP + HUD)

* **Aplikacja** desktop (Win/macOS/Linux).
* **Panele HUD**:

  * **Delta Inspector** — tokeny Δ, histogram energii ΔZ, heatmapa Mozaiki, linki do plików/zakresów.
  * **Spec Monitor** — progi α/β/Z, drift, naruszenia I*, uzasadnienia „explain why”.
* **Interakcje**: subskrybuje `EventBus`, renderuje artefakty `.glx/*`.

**Publiczne API (przykład):**

```python
def open_project(path: Path) -> None: ...
def show_delta(delta_report_path: Path) -> None: ...
def show_spec_state(spec_state_path: Path) -> None: ...
```

---

### 3.2 `core/`

* **Pipeline** kroków (przetwarzanie, walidacje, publikacja zdarzeń).
* **Invariants**: I1–I4 (patrz §6).
* **Mosaic & ASTMap**: budowa map/overlays, odwzorowania Φ/Ψ.
* **Registry** kroków i integracji.
* **Services.bus**: in-proc EventBus (z opcją backendu IPC).

**Interfejs kroków:**

```python
class Step(Protocol):
    name: str
    def prepare(self, ctx: RunCtx) -> None: ...
    def run(self, ctx: RunCtx) -> None: ...
    def validate(self, ctx: RunCtx) -> list["Violation"]: ...
```

---

### 3.3 `analysis/`

* **Metrics**: entropia, krawędzie, kontrast, bloki, histogramy.
* **Diff**: |Δ| per kanał, PSNR, SSIM (przybliżony), polityka resize.
* **Spectral**: FFT log-mag, pasma pierścieniowe/kątowe.
* **Formats**: heurystyki JPEG/PNG.
* **Exporters**: bundling do HUD.

---

### 3.4 `delta/`

* **Tokens**: `extract(prev_code, curr_code) -> list[Token]` (libcst + import graph + ΔCC).
* **Fingerprint**: histogram + bigramy + hash; deterministyczny podpis PR.
* **Features**: wektor cech Δ do kalibracji/spec.

**Token (zarys):**

```json
{ "kind": "MODIFY_SIG", "file": "m.py", "line": 42, "detail": {"Δarity": +1} }
```

---

### 3.5 `spec/`

* **Loader**: spina `/spec/glossary.md`, `/spec/invariants.yaml`, `/spec/schemas/*.json`.
* **Calibrate**: EWMA, MAD, t-digest/P² (kwantyle), drift (Page-Hinkley).
* **Schemas**: JSON Schema dla zdarzeń/artefaktów.

---

### 3.6 `security/` (SAST-Bridge)

* **Ingest**: SARIF/JSON (Bandit, Semgrep, Gitleaks, pip-audit/OSV, lokalny AST-lint).
* **NF (Normalized Findings)** → **PQ (Prioritized Queue)** → **FC (FixCandidate)**.
* **Zdarzenia**: `SAST.*` (patrz §8.2).

---

### 3.7 `io/`

* **Artifacts**: zapis/odczyt `.glx/*` (raporty, mozaiki, spec_state, metrics.parquet).
* **Konwencje nazewnicze** i wersjonowanie artefaktów.

---

### 3.8 `/spec` (SSOT w repo)

* `glossary.md` — definicje S/H/Z, Δ, Φ, Ψ, I1–I4.
* `invariants.yaml` — reguły, progi, wagi (żywe, aktualizowane kalibracją).
* `schemas/*.json` — kontrakty danych (events, tokens, delta_report, spec_state, findings).

---

## 4. Model danych i pliki kontraktowe

### 4.1 Δ-Tokens

```json
{
  "version": "delta.tokens.v1",
  "items": [
    {"kind":"ADD_FN","file":"a.py","line":10,"name":"foo"},
    {"kind":"ΔIMPORT","file":"a.py","added":["x"],"removed":["y"]},
    {"kind":"MODIFY_SIG","file":"a.py","line":42,"detail":{"Δarity":+1}}
  ]
}
```

### 4.2 Fingerprint

```json
{
  "version":"delta.fp.v1",
  "histogram":{"ADD_FN":3,"ΔIMPORT":1,"MODIFY_SIG":1},
  "bigrams":{"ADD_FN→MODIFY_SIG":1},
  "hash":"fp:sha256:…"
}
```

### 4.3 `delta_report.json`

```json
{
  "version":"delta.report.v1",
  "tokens": [...],
  "fingerprint": {...},
  "violations": [{"invariant":"I3","score":0.12,"explain":"Φ(Δ_AST) ≠ Δ_MOZ in core/mosaic.py"}],
  "score":{"alpha":0.9,"beta":0.95,"zeta":0.99,"value":0.93,"decision":"REVIEW"}
}
```

### 4.4 `spec_state.json`

```json
{
  "version":"spec.state.v1",
  "updated":"2025-10-09T12:34:56Z",
  "thresholds":{
    "global":{"alpha":0.90,"beta":0.95,"zeta":0.99},
    "modules":{"core":{...},"analysis":{...},"gui":{...}}
  },
  "stats":{"ewma":{...},"mad":{...},"quantiles":{...}},
  "drift":{"state":"stable","frozen_until":null}
}
```

### 4.5 SAST — NF (Normalized Finding)

```json
{
  "id":"NF-…","tool":"bandit","rule_id":"B602","cwe":"CWE-78",
  "severity":"HIGH","confidence":0.85,"category":"RCE",
  "location":{"file":"x.py","line":123,"col":8},
  "ast_ref":"AST::<uid>","mosaic_cell":"MOZ::<cell>",
  "delta_bind":"modified","evidence":{"snippet":"..."},
  "suppression":{"state":"none","reason":""},
  "fingerprint":"sha256:…"
}
```

---

## 5. Przepływy sterowania (end-to-end)

### 5.1 Pętla developerska (local)

1. Dev zmienia plik → **pre-commit**: lint/type/AST-tokens/quick I-check.
2. **commit-msg**: dopisuje **Δ-Fingerprint** (komentarz).
3. **post-commit**: generuje `.glx/delta_report.json`, `.glx/mosaic_*.png`, aktualizuje `/spec_state.json`.
4. GUI/HUD podświetla zmiany; **Spec Monitor** decyduje (α/β/Z).

### 5.2 CI / PR

1. Actions: `lint → typecheck → tests → delta-tokens → invariants-gate → build`.
2. Publikacja artefaktów `.glx/*`, komentarz do PR (fingerprint + decyzja).
3. **Block / Require review / Auto-merge** zgodnie z α/β/Z.

### 5.3 SAST-Bridge

1. `SAST.ScanRequested` → adaptery czytają SARIF/JSON → **NF**.
2. NF wiązane do **AST/Mozaiki** (ast_ref, mosaic_cell) + `delta_bind`.
3. **PQ** (kolejka priorytetów) z `risk_score`.
4. Na żądanie: **FixCandidate** (patch/test/config) → walidacje I* → weryfikacja → decyzja.

---

## 6. Φ/Ψ i inwarianty I1–I4

**Φ : AST → Mozaika** – odwzorowanie strukturalne (funkcje/klasy/importy/fan-in/out) w siatkę komórek.
**Ψ : Mozaika → AST** – inferencja „co w AST” z geometryki (gęstości/gradienty/zmiany).

**Inwarianty (skrót):**

* **I1 — Energia ΔZ** w granicach: `ΔZ ≤ α(module)` (lub soft-violations z karą).
* **I2 — Kontrakty**: brak niedozwolonej zmiany API/publicznych sygnatur, zachowana semantyka portów.
* **I3 — Komutacja z Δ**: `|| Φ(Δ_AST) – Δ_MOZ || ≤ τ`.
* **I4 — Stabilność obserwowalna**: brak spadku test-cov/SSIM/PSNR poniżej progów.

**Decyzje:**
`score ≤ α → ACCEPT` • `α < score ≤ β → REVIEW` • `score > β → BLOCK` • `≫ Z → HARD BLOCK`.

---

## 7. Kalibracja progów (żywy Spec)

* **EWMA** dla trendu, **MAD** dla odporności, **t-digest/P²** dla kwantyli.
* Progi **per moduł** (core/analysis/gui/security).
* Detekcja **driftu** (Page-Hinkley): zamrożenie progów i log przyczyn.

**Aktualizacja:**

* Po każdym `post-commit`: dopisz wektor cech do `metrics.parquet`, przelicz `spec_state.json`.
* W CI: uśredniaj w oknie N commitów (odporność na szum).

---

## 8. Komunikacja i kontrakty zdarzeń

### 8.1 Envelope (EventBus)

```json
{
  "topic":"core.step.finished",
  "ts":"2025-10-09T12:34:56Z",
  "sha":"…",
  "payload":{ "...": "..." },
  "schema":"spec/schemas/events/core.step.finished.json"
}
```

### 8.2 Tematy (wybór)

* `core.step.prepare|run|finished`
* `invariants.violation` (I1–I4, score, explain)
* `delta.tokens.ready`, `delta.fingerprint.ready`
* `spec.thresholds.update`, `spec.drift.alert`
* `SAST.ScanRequested|FindingsReady|Prioritized|FixProposed|FixValidated`
* `gui.hud.request|render`

---

## 9. Artefakty `.glx/*` (konwencje)

* `delta_report.json` — raport pętli Δ (wersjonowany `delta.report.v1`).
* `mosaic_*.png` — wizualizacje (rozmiary/warstwy w nazwie).
* `spec_state.json` — stan kalibracji (wersja, stemple czasowe).
* `metrics.parquet` — dane strumieniowe do kalibracji.
* `security/merged.sarif` — po unifikacji skanerów.
* Wszystko **w repo** (anchor: git root), deterministyczne timestampy w ZIPach.

---

## 10. Rozszerzalność

* **Steps/Plugins** — rejestr przez dekorator `steps.register("name")`.
* **GUI Panels** — `gui/panels/*.py` ładowane dynamicznie (białe listy).
* **SAST Adapters** — `security/adapters/{bandit,semgrep,...}.py` → NF.
* **GLX Lens** — `*.lens.yaml` (schema), używane przez generator i hooki.

---

## 11. Bezpieczeństwo

* **Fail-closed** na bramkach I* i Guard (polityki repo).
* **Sandbox** dla pluginów/filtrów (oddzielny proces, biała lista modułów).
* **Sekrety**: Gitleaks + polityka rotacji; brak zapisu sekretów w artefaktach.
* **.env reżim**: czytany **wyłącznie** z katalogu projektu; ścieżki wyjściowe muszą być **wewnątrz repo**.

---

## 12. Obserwowalność (EGDB)

**Tabele (zarys):**

* `events(topic, ts, sha, payload_json)`
* `delta_commits(sha, fp_hash, score, decision)`
* `invariant_hits(sha, invariant, score, details_json)`
* `sast_findings(id, sha, severity, file, line, fingerprint, state)`
* `heal_candidates(id, sha, kind, scope_json, confidence, deltas_json)`
* `heal_verification(candidate_id, sandbox_ok, ci_ok, mutation_score, notes)`

**Widoki:**

* `vw_pr_fingerprint`, `vw_invariants_heat`, `vw_mttr_sec`.

---

## 13. Wydajność i niezawodność

* **Wektoryzacja** (NumPy) + unikanie zbędnych kopii (uint8↔f32).
* **Cache** okien FFT (Hann) per shape.
* **Tokenizacja AST** tylko dla dotkniętych plików; cache AST/graph.
* **ThreadPool** dla zadań I/O; CPU-bound rozdzielone per krok.
* **Deterministyczne** ścieżki (seed, kolejność, ZIP timestamps).

---

## 14. Wersjonowanie i zgodność

* **SemVer** dla pakietu.
* **Schematy** posiadają wersje (`*.v1`, `*.v2`...), migratory łatają stare artefakty.
* **Spec** — wersjonowany w repo; zmiany schematów = minor/major.

---

## 15. Interfejsy publiczne (skrót)

**CLI (docelowo):**

```
glx gen --lens <id>                   # generacja wg soczewki
glx delta report --base <ref>         # raport Δ i artefakty .glx/*
glx spec calibrate --input .glx/…     # aktualizacja progów α/β/Z
glx sast scan --scope changed         # NF→PQ→artefakty bezpieczeństwa
```

**Python (importy):**

```python
from glitchlab.core import run_pipeline, check_invariants
from glitchlab.delta import extract_tokens, fingerprint
from glitchlab.spec import load_spec, calibrate
from glitchlab.security import scan_sast, make_fix_candidates
```

---

## 16. Deplo/Ops

* **Local dev**: venv + `pip install -e .` + `python -m glitchlab.gui.app`.
* **CI**: GitHub Actions (matrix py3.10–3.12), artefakty `.glx/*` do podglądu.
* **Opcjonalnie**: kontenery dla `sast_bridge` i zewn. BUS/DB (NATS/Postgres).

---

## 17. Kryteria akceptacji (etapy vNext)

* **Etap 0**: `pyproject.toml`, `/spec` (glossary/invariants/schemas), minimalne CI, pre-commit.
  *Zielone: lint, typecheck, tests; GUI startuje; zapisuje `.glx/spec_state.json`.*
* **Etap 1**: `delta.tokens` + `fingerprint` + `delta_report.json`.
  *Stabilny hash; zgodność tokenów z diffem w 3 próbkach.*
* **Etap 2**: I1–I4 + kalibracja + bramki α/β/Z w CI; panele HUD.
  *PR łamiący I3 blokowany; HUD wyjaśnia decyzję.*
* **Etap 3**: SAST-Bridge (NF→PQ→FC) z walidacją I*.
  *CRITICAL → blokada + propozycja patcha z uzasadnieniem.*

---

## 18. Załączniki (skrót schematów)

**`/spec/invariants.yaml` (zarys):**

```yaml
version: invariants.v1
thresholds:
  global: { alpha: q90, beta: q95, zeta: q99 }
  by_module:
    core:   { alpha: q90, beta: q95, zeta: q99 }
    gui:    { alpha: q90, beta: q95, zeta: q99 }
    analysis:{ alpha: q90, beta: q95, zeta: q99 }
features:
  - ΔLOC
  - ΔCC
  - ΔImports
  - SSIM_drop
  - ΔFanIn
  - ΔFanOut
weights:
  ΔLOC: 0.1
  ΔCC:  0.2
  ΔImports: 0.15
  SSIM_drop: 0.3
  ΔFanIn: 0.125
  ΔFanOut:0.125
rules:
  I1: { expr: "ΔZ <= alpha(module)", penalty: soft }
  I2: { expr: "no_public_api_break", penalty: hard }
  I3: { expr: "norm(Phi(Δ_AST)-Δ_MOZ)<=τ", penalty: hard }
  I4: { expr: "no_observable_regression", penalty: soft }
```

**`/spec/schemas/events/*.json` (envelope skeleton):**

```json
{
  "$id":"events.core.step.finished.v1",
  "type":"object",
  "properties":{
    "topic":{"const":"core.step.finished"},
    "ts":{"type":"string","format":"date-time"},
    "sha":{"type":"string"},
    "payload":{"type":"object"}
  },
  "required":["topic","ts","payload"]
}
```

---

## 19. Uwagi implementacyjne (binding do istniejącego kodu)

* **Łączenie z repo**: zachowaj istniejący układ (`analysis/`, `core/`, `gui/`, `filters/`, `mosaic/`, `tests/`, `.githooks/`, `.glx/`).
* **Hooki**: przestrzegaj reżimu `.env` (wyłącznie katalog projektu), **GLX_ROOT == project_dir**, wyjścia **wewnątrz** repo.
* **Hot-spoty**: zrefaktoruj Perlin fallback (wektorowo), cache FFT, polityka resize w `analysis.diff`.

---

### TL;DR dla zespołu

* GUI to twarz systemu (Delta Inspector, Spec Monitor).
* Core+Analysis+Delta+Security to mięśnie (Φ/Ψ, I*, Δ-tokens, SAST).
* `.glx/*` to krew obiegu (artefakty i fakty).
* `/spec` to pamięć i kompas (definicje, progi, schematy).

Z tym podziałem budujemy produkt, który **rozumie swoje zmiany**, **sam się pilnuje** i **potrafi proponować leczenie** — bez utraty prostoty codziennej pracy.
