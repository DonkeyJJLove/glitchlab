# CI/Ops — hooki Git + zalecany workflow GitHub Actions

**Plik:** `docs/50_ci_ops.md`
**Status:** produkcyjny (SSOT dla `.githooks/*` i `.github/workflows/ci.yml`)
**Powiązania:** `docs/12_invariants.md`, `docs/13_delta_algebra.md`, `docs/14_mosaic.md`, `docs/41_pipelines.md`, `docs/20_bus.md`, `docs/21_egdb.md`, `docs/60_security.md`, `docs/30_sast_bridge.md`.

---

## 0) Filozofia CI/Ops w GlitchLab

* **Δ-first:** wszystkie kontrole i raporty są liczone z **różnic** (AST-DIFF, tokeny Δ, mozaika Φ(Δ)).
* **One loop → one artifact:** każdy przebieg generuje **pojedynczy artefakt** kanoniczny w `.glx/*`.
* **Fail-closed:** bramki **I1–I4** (patrz `docs/12_invariants.md`) decydują o blokadzie/akceptacji.
* **SSOT:** hooki i Actions czytają te same **progi α/β/Z** z `.glx/spec_state.json` (aktualizowane przez kalibrację).

---

## 1) Hooki Git — kontrakt i instalacja

### 1.1 Instalacja (w repo)

```bash
git config core.hooksPath .githooks
chmod +x .githooks/*         # dla bezpieczeństwa
```

> Hooki korzystają ze **środowiska projektu** (`.env/.env.local` w katalogu projektu).
> Wymagane klucze: `GLX_RUN, GLX_PKG, GLX_MOSAIC, GLX_OUT, GLX_AUTONOMY_OUT` (+ parametry Mozaiki).
> **Domyślna kotwica ścieżek:** `GLX_PATH_ANCHOR=git` (wyjścia muszą leżeć w repo).

### 1.2 Zmienne środowiskowe (minimum)

W pliku `.env.local` (przykład):

```ini
GLX_PKG=glitchlab
GLX_MOSAIC=hybrid
GLX_RUN=A+M+Z                 # A:gateway, M:mosaic, Z:audit zip
GLX_OUT=.glx/last
GLX_AUTONOMY_OUT=analysis/last/autonomy
GLX_ROWS=32
GLX_COLS=32
GLX_EDGE_THR=0.25
GLX_DELTA=0.25
GLX_KAPPA=0.5
GLX_PHI=policy                # lub: 'raw'
GLX_POLICY=analysis/policy.json
```

### 1.3 Przepisy hooków (SSOT)

| Hook                 | Cel (Δ-first)                                                                                              | Blokujące                          | Co publikuje                                             |
| -------------------- | ---------------------------------------------------------------------------------------------------------- | ---------------------------------- | -------------------------------------------------------- |
| `pre-commit`         | Szybkie kontrole: **Ruff + MyPy (fast)**, tokeny Δ (dotknięte pliki), szybki SAST-lite                     | Tak (lint, typy, SAST-CRITICAL)    | `.glx/commit_snippet.txt` (krótki podpis Δ), log lokalny |
| `prepare-commit-msg` | Dokleja **Δ-fingerprint** do treści commita (sekcja `#Δ:`)                                                 | Nie                                | –                                                        |
| `post-commit.py`     | **A/M/Z** wg `GLX_RUN`: buduje paczkę autonomii (**A**), mozaikę/hitmapy (**M**), tworzy audit ZIP (**Z**) | Nie                                | `.glx/last/*`, `backup/AUDIT_*.zip`                      |
| `pre-push`           | Pełne **I1–I4** + gate 3-progowy **α/β/Z**; sprawdza dryf; sygnalizuje BUS                                 | Tak (`score>β` albo naruszenie I*) | `.glx/delta_report.json`, `.glx/spec_state.json`         |

> Implementacyjne szczegóły `post-commit.py`:
>
> * Ładuje `.env` **wyłącznie** z katalogu projektu.
> * Wymusza wyjścia **wewnątrz repo**.
> * Obsługuje tryby `A|M|E|Z` i porządek wykonania.
> * Dla **M** uruchamia `mosaic.hybrid_ast_mosaic from-git-dump --base … --head HEAD`.

---

## 2) Zalecany workflow GitHub Actions (`.github/workflows/ci.yml`)

### 2.1 Założenia

* **Matrix**: Python `3.10–3.12`.
* **Concurrency**: pojedynczy workflow na gałąź.
* **Permissions**: minimalne; komentarze do PR tylko, gdy potrzebne.
* **Cache**: `pip` + artefakty `.glx/*` jako **upload-artifact**.
* **Gating**: osobny krok `invariants-check` (zwraca exit≠0 przy blokadzie).

### 2.2 Minimalny, produkcyjny szablon

```yaml
name: CI

on:
  push:
    branches: [ "**" ]
  pull_request:
    branches: [ "**" ]
  workflow_dispatch:

concurrency:
  group: ci-${{ github.ref }}
  cancel-in-progress: true

permissions:
  contents: read
  pull-requests: write

jobs:
  lint_type:
    runs-on: ubuntu-latest
    strategy:
      matrix: { python: [ "3.10", "3.11", "3.12" ] }
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: ${{ matrix.python }} }
      - name: Install deps
        run: |
          python -m pip install -U pip wheel
          pip install -e .[dev]            # ruff, mypy, pytest w extras
      - name: Ruff (lint)
        run: ruff check .
      - name: MyPy (fast)
        run: mypy --install-types --non-interactive --pretty --no-error-summary .

  test:
    needs: [lint_type]
    runs-on: ubuntu-latest
    strategy:
      matrix: { python: [ "3.11" ] }       # 1 wersja na szybkie testy
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: ${{ matrix.python }} }
      - name: Install deps
        run: |
          python -m pip install -U pip wheel
          pip install -e .[dev]
      - name: PyTest + coverage
        run: |
          pytest -q --maxfail=1 --disable-warnings \
                 --cov=glitchlab --cov-report=xml
      - name: Upload coverage
        uses: actions/upload-artifact@v4
        with: { name: coverage-xml, path: coverage.xml, if-no-files-found: ignore }

  delta_build:
    needs: [test]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with: { fetch-depth: 0 }           # potrzebny pełny git dla from-git-dump
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - name: Install deps
        run: |
          python -m pip install -U pip wheel
          pip install -e .
      - name: Prepare env (.env CI)
        run: |
          cat > .env <<'EOF'
          GLX_PKG=glitchlab
          GLX_MOSAIC=hybrid
          GLX_RUN=A+M
          GLX_OUT=.glx/last
          GLX_AUTONOMY_OUT=analysis/last/autonomy
          GLX_ROWS=32
          GLX_COLS=32
          GLX_EDGE_THR=0.25
          GLX_DELTA=0.25
          GLX_KAPPA=0.5
          GLX_PHI=policy
          GLX_POLICY=analysis/policy.json
          EOF
      - name: Build Δ artifacts (Mozaika + Delta)
        run: |
          # krok analogiczny do post-commit (A/M), ale jawnie w CI:
          python - <<'PY'
          import os, subprocess, sys, pathlib
          root = pathlib.Path(__file__).resolve().parents[2]
          env = dict(os.environ)
          env["PYTHONUTF8"]="1"
          # 1) Gateway (A)
          subprocess.check_call([sys.executable, "-m", "analysis.autonomy.gateway", "build", "--out", "analysis/last/autonomy"], cwd=root, env=env)
          # 2) Mozaika (M)
          subprocess.check_call([sys.executable, "-m", "mosaic.hybrid_ast_mosaic",
                                 "--mosaic", "hybrid", "--rows", "32", "--cols", "32",
                                 "--edge-thr", "0.25", "--kappa-ab", "0.5",
                                 "--phi", "policy", "--policy-file", "analysis/policy.json",
                                 "--repo-root", ".", "from-git-dump", "--base", "origin/${{ github.base_ref || 'master' }}", "--head", "HEAD",
                                 "--delta", "0.25", "--out", ".glx/last", "--strict-artifacts"
                                 ], cwd=root, env=env)
          PY
      - name: Upload Δ artifacts
        uses: actions/upload-artifact@v4
        with:
          name: glx-delta
          path: |
            .glx/last/**
            .glx/spec_state.json
            .glx/delta_report.json
          if-no-files-found: warn

  invariants_gate:
    needs: [delta_build]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/download-artifact@v4
        with: { name: glx-delta, path: .glx/collect }
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - name: Install package
        run: pip install -e .
      - name: Enforce I1–I4 (gate α/β/Z)
        run: |
          python -m glitchlab.core.invariants --ci-gate \
                 --spec .glx/collect/spec_state.json \
                 --delta .glx/collect/delta_report.json
      - name: Comment PR (summary)  # uruchomi się tylko dla pull_request
        if: ${{ github.event_name == 'pull_request' }}
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          python - <<'PY'
          import os, json, pathlib, textwrap, requests
          p = pathlib.Path(".glx/collect/delta_report.json")
          spec = pathlib.Path(".glx/collect/spec_state.json")
          if p.exists():
            delta = json.loads(p.read_text())
            title = "Δ Gate Result"
            score = delta.get("score","?")
            tokens = delta.get("tokens_summary",{})
            msg = f"**Δ score:** `{score}`\n\n**Tokens:** `{tokens}`\n\nArtifacts: _glx-delta_"
            pr_api = f"https://api.github.com/repos/{os.environ['GITHUB_REPOSITORY']}/issues/{os.environ['PR_NUMBER']}/comments"
          PY

  sast_bridge:
    needs: [test]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with: { fetch-depth: 0 }
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - name: Install scanners + package
        run: |
          python -m pip install -U pip wheel
          pip install -e .
          pip install bandit semgrep gitleaks-cli pip-audit
      - name: Run SAST Bridge (merge SARIF → NF/PQ)
        run: |
          python -m glitchlab.security.sast_bridge scan --out .glx/security
      - name: Upload SAST artifacts
        uses: actions/upload-artifact@v4
        with: { name: glx-sast, path: .glx/security/**, if-no-files-found: warn }
```

> **Uwaga:** krok komentarza PR (REST) jest szkicem; w pakiecie dostarczamy dedykowany skrypt CLI do publikacji komentarza z czytelną tabelą tokenów i naruszeń I*.

### 2.3 Gate 3-progowy (α/β/Z)

* **`score ≤ α`** → ✅ zielone (auto-merge polityką repo).
* **`α < score ≤ β`** → 🧩 żółte (wymagany review).
* **`score > β`** lub **jakiekolwiek I*** → ❌ czerwone (blokada).

**Źródło progów:** `.glx/spec_state.json` (aktualizowane przez `spec/calibrate.py` w cyklu dataflow/CI).

---

## 3) Artefakty CI (do pobrania)

* `.glx/last/**` – Mozaika, mapy, raporty Δ (JSON/PNG/MD).
* `.glx/delta_report.json` – skondensowany wynik tokenizacji i scoringu.
* `.glx/spec_state.json` – progi i stan kalibracji (α/β/Z, MAD, drift).
* `.glx/security/**` – **SAST Bridge**: `merged.sarif`, `queue.json`, `fix_candidates.json` (jeśli włączone).
* `coverage.xml` – raport pokrycia.

---

## 4) Rekomendowane zasady blokowania

1. **CRITICAL (SAST)** lub **secrets** → natychmiastowa blokada (pre-commit i CI).
2. **I1–I4 naruszone** → blokada w `pre-push` i w `invariants_gate`.
3. **Drift progów** (Page-Hinkley) → **freeze thresholds** i żądanie anotacji PR (dlaczego zmiana).
4. **ΔTESTS < 0** → wymuś dodanie testów (auto-stub może być wygenerowany).

---

## 5) Release i kanały (beta/stable)

### 5.1 Tagowanie i build

* **Tag SemVer**: `vX.Y.Z[-beta.N]` → uruchamia workflow `release.yml` (oddzielny).
* **Build**: `sdist + wheel`, podpis (opcjonalnie), upload do rejestru (TestPyPI/PyPI) lub GH Releases.
* **Artefakty**: paczka binarna GUI (opcjonalnie), checksumy, `CHANGELOG.md` (generowany z Conventional Commits).

### 5.2 Polityka wersjonowania

* **Publiczne API** (importowalne moduły) → SemVer.
* **Panele/GUI** → minor na nowe panele, patch na poprawki interfejsu; major gdy Panel API się zmienia.

---

## 6) Odwzorowanie lokalne (reprodukcja CI)

Do szybkiej reprodukcji pipeline’u CI lokalnie:

```bash
# 1) szybkie sanity
ruff check . && mypy .

# 2) testy
pytest -q --maxfail=1

# 3) artefakty Δ (jak w CI)
export $(cat .env.local | xargs)
python -m analysis.autonomy.gateway build --out analysis/last/autonomy
python -m mosaic.hybrid_ast_mosaic --mosaic hybrid --rows 32 --cols 32 \
       --edge-thr 0.25 --kappa-ab 0.5 --phi policy --policy-file analysis/policy.json \
       --repo-root . from-git-dump --base origin/master --head HEAD \
       --delta 0.25 --out .glx/last --strict-artifacts

# 4) gate I1–I4 (wyjście !=0 gdy blokada)
python -m glitchlab.core.invariants --ci-gate \
       --spec .glx/spec_state.json \
       --delta .glx/delta_report.json
```

---

## 7) Dobre praktyki operacyjne

* **Szybkie hooki:** `pre-commit` ≤ 5–8 s (lint/typy/tokeny Δ tylko dla dotkniętych plików).
* **Pełne bramki** w `pre-push`/CI, nie w `pre-commit`.
* **Artefakty jako dowody:** każdy PR ma link do `glx-delta` (mosaic + delta report).
* **Tryb headless** GUI w CI (smoke).
* **Quarantine pluginów** w testach (sandbox off-by-default).
* **Freeze thresholds** dokumentowany w komentarzu CI (kto, dlaczego, okno).

---

## 8) FAQ

**Czy muszę mieć `.env.local` w CI?**
Nie — workflow generuje minimalną `.env` lokalnie; w repo trzymaj wzorzec.

**Co jeśli `origin/master` nie istnieje?**
Użyj `github.base_ref` dla PR lub `HEAD~1` na push (fallback w skryptach).

**Jak zmienić progi α/β/Z?**
Nie edytuj ręcznie — **kalibracja** aktualizuje `.glx/spec_state.json`. Możesz tymczasowo „freeze”.

---

## 9) Aneks A — mapa zadań do ról

| Warstwa   | Hook                | Job w CI          | Plik/komenda                             |
| --------- | ------------------- | ----------------- | ---------------------------------------- |
| Lint/Typy | `pre-commit`        | `lint_type`       | `ruff check .`, `mypy .`                 |
| Testy     | –                   | `test`            | `pytest --cov`                           |
| Mozaika/Δ | `post-commit` (A/M) | `delta_build`     | `mosaic.hybrid_ast_mosaic from-git-dump` |
| Gate I*   | `pre-push`          | `invariants_gate` | `glitchlab.core.invariants --ci-gate`    |
| SAST      | `pre-commit` (lite) | `sast_bridge`     | `security.sast_bridge scan`              |
| Artefakty | `post-commit`/ZIP   | `upload-artifact` | `.glx/last/**`, `backup/AUDIT_*.zip`     |

---

## 10) Aneks B — minimalny `.pre-commit-config.yaml` (opcjonalnie)

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9
    hooks: [ { id: ruff, args: [ "--fix" ] }, { id: ruff-format } ]
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.11.2
    hooks: [ { id: mypy, additional_dependencies: [ "types-Pillow" ] } ]
  - repo: local
    hooks:
      - id: glx-delta-tokens
        name: glx-delta-tokens
        entry: python -m delta.tokens --changed-only
        language: system
        pass_filenames: false
      - id: glx-sast-lite
        name: glx-sast-lite
        entry: python -m security.sast_bridge scan --fast --severity-threshold MEDIUM
        language: system
        pass_filenames: false
```

---

> **Kontrakt:** Niniejszy dokument jest **źródłem prawdy** dla integracji **hooków Git** oraz **workflow GitHub Actions**. Każda zmiana interfejsów (ścieżki artefaktów `.glx/*`, argumenty CLI, struktura raportów) musi być odzwierciedlona tutaj i w odpowiednich dokumentach (`docs/12_invariants.md`, `docs/13_delta_algebra.md`, `docs/41_pipelines.md`, `docs/60_security.md`).
