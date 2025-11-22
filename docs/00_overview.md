# GlitchLab — platforma Human-AI do generatywnego rozwoju oprogramowania

**GlitchLab** to multiplatformowa (Win/macOS/Linux) platforma łącząca **IDE/GUI**, **silniki analityczne i generatywne**, oraz **kontrolę jakości i bezpieczeństwa**. Umożliwia projektowanie, uruchamianie i *samoleczenie* kodu w cyklu „**jedna pętla → jeden artefakt**”, wykorzystując algebraiczne postrzeganie zmian (Δ) i sprzężenie struktur **AST ↔ Mozaika** (**Φ/Ψ**).

---

## Najkrócej

* **IDE/APP**: jedna aplikacja do pisania, inspekcji, uruchamiania i wizualnej diagnozy kodu (HUD).
* **Delta-first**: każdą zmianę opisujemy tokenami **Δ-AST** i podpisem **Fingerprint**, zamiast „stanu”.
* **Matematyczna spójność**: inwarianty **I1–I4** egzekwują zgodność pomiędzy AST a Mozaiką.
* **SSOT / Spec żywy**: definicje i progi (**α/β/Z**) żyją w `/spec`, kalibrują się na historii zmian.
* **BUS + EGDB**: zdarzenia i telemetryka toczy się przez prosty Bus i relacyjną bazę (widoki/raporty).
* **SAST-Bridge**: unifikacja wyników skanerów bezpieczeństwa → kolejka priorytetów → kandydaci poprawek.
* **Self-healing**: propozycje łatek i testów pod kontrolą Guard (fail-closed) i inwariantów jakości.

---

## Filary projektowe

1. **Prostota × Moc** – minimalne prymitywy, kompozycja i *samoopisujące się* struktury.
2. **Delta-first** – generacja, dokumentacja i gating działają na różnicach, nie na migawkach.
3. **Soczewka GLX (Lens)** – jeden metaopis określa, co generować (py/bash/md), skąd brać kontekst i które reguły egzekwować.
4. **Event-driven** – wszystko jest zdarzeniem: analiza, walidacja, naprawa.
5. **Fail-closed** – bezpieczeństwo i jakość zawsze wygrywają: inwarianty I1–I4 i polityki Guard zatrzymują niepożądane zmiany.

---

## Kluczowe pojęcia (rdzeń matematyczny)

* **S/H/Z** – składowe opisu stanu i energii zmian (strukturalne, geometryczne, złożonościowe).
* **Δ (delta)** – algebra zmian rozkładana na **tokeny Δ-AST** (`ADD_FN`, `MODIFY_SIG`, `ΔIMPORT`, …).
* **Φ : AST → Mozaika** i **Ψ : Mozaika → AST** – sprzężone odwzorowania; ich zgodność mierzą inwarianty.
* **I1–I4 (inwarianty)** – zestaw reguł jakości:

  * **I1**: monotoniczność/energia Δ w dopuszczalnych granicach,
  * **I2**: brak niedozwolonych zmian kontraktów (API, semantyka),
  * **I3**: *komutacja z Δ* (Φ(Δ_AST) ≈ Δ_MOZ),
  * **I4**: stabilność obserwowalna (testy, pokrycie, metryki).
* **Progi α/β/Z** – trzy poziomy decyzji (akceptuj / review / blokuj), kalibrowane z historii (kwantyle, EWMA, MAD).

---

## Architektura (wysoki poziom)

```
+--------------------+     +----------------------+     +----------------------+
|  GUI/APP (HUD/IDE) | <-- |         BUS          | --> |         EGDB         |
|  - Delta Inspector |     |  core.*, delta.*,    |     |  widoki/raporty      |
|  - Spec Monitor    |     |  invariants.*, ...   |     |  telemetry historii  |
+---------^----------+     +-----------^----------+     +-----------^----------+
          |                                |                           |
          |        +-----------------------+------------------+       |
          |        |   Silnik: Core/Analysis/Delta/Spec/Sec   |       |
          |        |  - pipeline, AST↔Mozaika (Φ/Ψ), I1–I4    |       |
          |        |  - Δ-tokens, Fingerprint, kalibracja     |       |
          |        |  - SAST-Bridge (NF → PQ → FC)            |       |
          |        +-----------------------^------------------+       |
          |                                |                           |
          +--------------------------------+---------------------------+
                   (artefakty .glx/*: delta_report, spec_state, mozaiki)
```

---

## GUI/APP (IDE + produkt)

* **Delta Inspector** — lista tokenów Δ, histogram energii ΔZ, heatmapa Mozaiki, linki do plików/linii.
* **Spec Monitor** — progi α/β/Z, stan kalibracji i driftu, naruszenia inwariantów, *explain why*.
* **Tryby pracy**:

  * **Builder** – generacja plików wg Soczewki GLX,
  * **Inspector** – przegląd i audyt zmian,
  * **Healer** – propozycje poprawek (patch/test/config) z dowodami i walidacją.

---

## Soczewka GLX (Lens)

Deklaratywny plik (YAML) opisujący **co** generować i **jak** to weryfikować:

```yaml
lens: v1
id: glx.core.metrics.basic
kind: python_module | bash_script | doc_page
inputs:
  - src: core/metrics/basic.py
  - ctx: .glx/spec_state.json
outputs:
  - path: core/metrics/basic.py
  - path: docs/core_metrics_basic.md
invariants: [I1_monotonicity_deltaZ, I3_commutes_with_delta]
security:
  checks: [no_exec_eval, disallow_subprocess_shell]
generation:
  strategy: [template_first, diff_guided_prompting, retrieve_repo_context]
```

> To **jedno źródło prawdy** (SSOT) dla generatora, hooków i bramek CI.

---

## SAST-Bridge (bezpieczeństwo jako przepływ)

* **NF (Normalized Findings)** — zunifikowane znaleziska (Bandit/Semgrep/Gitleaks/pip-audit/OSV/AST-lint).
* **PQ (Prioritized Queue)** — kolejka z **risk_score = f(severity, confidence, reachability, Δ, public_api)**.
* **FC (FixCandidate)** — kontrakt proponowanej poprawki (patch/test/constraints), sprawdzany przez I1–I4 i Guard.
* **Tematy BUS**: `SAST.FindingsReady`, `SAST.Prioritized`, `SAST.FixProposed`, `SAST.FixValidated`.

---

## Artefakty `.glx/*`

* `delta_report.json` — tokeny Δ, Fingerprint, score, naruszenia I*,
* `mosaic_*.png` — wizualizacje Δ/heatmapy,
* `spec_state.json` — aktualne progi/kwantyle/EWMA/MAD,
* `metrics.parquet` — cechy Δ per commit/PR.

---

## CI/CD i hooki

* **pre-commit**: lint (ruff), typy (mypy), szybkie I-check, AST-tokens.
* **commit-msg**: podpis **Δ-Fingerprint** + link do artefaktów `.glx/*`.
* **post-commit**: generacja artefaktów delta/spec; aktualizacja kalibracji.
* **GitHub Actions**: `lint → typecheck → tests → invariants-gate → build → artifacts`.
* **Gating (α/β/Z)**: auto-merge / wymuszenie review / blokada.

---

## Layout projektu (skrót)

```
glitchlab/
  gui/ (app + panele HUD)
  core/ (pipeline, invariants, mosaic, astmap, steps)
  analysis/ (metrics, diff, spectral, formats, exporters)
  delta/ (tokens, fingerprint, features)
  spec/ (loader, calibrate, schemas)
  security/ (sast_bridge)
  io/ (artifacts)
spec/ (glossary, invariants.yaml, schemas)
.githooks/  .github/workflows/  .glx/  pyproject.toml  README.md
```

---

## Tryb pracy: „jedna pętla → jeden artefakt”

Każda iteracja generacji dotyczy **jednego** pliku (kod / dokument / skrypt).
Soczewka GLX określa wejścia, wyjścia i reguły. **Mały model** działa na kontekście **Δ-only**, a wynik natychmiast przechodzi walidacje (I1–I4, SAST, testy szybkie). Dzięki temu **tempo** i **miarodajność** rosną, a koszty maleją.

---

## Bezpieczeństwo i zgodność

* **Fail-closed**: żadne działanie nie omija Guard i I1–I4.
* **Sekrety**: skan Gitleaks + polityka rotacji i migracji do sekret store/KMS.
* **Sandbox**: pluginy/filtry uruchamiane w odseparowanym środowisku z białą listą.

---

## Quickstart

> Wymagania: Python 3.10–3.12, systemowe biblioteki dla Pillow.

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e .                                    # instalacja edytowalna z pyproject.toml
python -m glitchlab.gui.app                         # start GUI/APP (HUD/IDE)
```

Przykładowy obieg:

1. wprowadzisz zmianę w module → 2) **post-commit** zbuduje artefakty `.glx/*` → 3) w **Delta Inspector** obejrzysz tokeny i mozaikę → 4) **Spec Monitor** pokaże decyzję α/β/Z → 5) w razie potrzeby **Healer** zaproponuje patch/test.

---

## Gdzie czytać dalej

* **Architektura:** `docs/10_architecture.md`
* **Słownik/Spec:** `docs/11_spec_glossary.md`
* **Inwarianty i kalibracja:** `docs/12_invariants.md`
* **Δ-Algebra i Fingerprint:** `docs/13_delta_algebra.md`
* **BUS/EGDB:** `docs/20_bus.md`, `docs/21_egdb.md`
* **SAST-Bridge:** `docs/30_sast_bridge.md`
* **GUI/APP:** `docs/40_gui_app.md`
* **CI/Ops:** `docs/50_ci_ops.md`
* **Observability:** `docs/70_observability.md`

---

## Licencja i wkład

* Licencja i zgodność SPDX są deklarowane w root (patrz `LICENSE` i nagłówki plików).
* Kontrybucje są mile widziane: każdy PR przechodzi te same bramki α/β/Z.

---

**GlitchLab** porządkuje rzeczywistość kodu w jeden, spójny język zmian – od AST, przez mozaikę i HUD, po bezpieczeństwo i CI – tak, by zespół Human-AI mógł **szybciej budować**, **pewniej wdrażać** i **mądrzej się leczyć**.
