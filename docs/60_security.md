# Bezpieczeństwo (polityki, SAST, sandbox, supply-chain)

**Plik:** `docs/60_security.md`
**Status:** produkcyjny (SSOT dla bezpieczeństwa kodu i łańcucha dostaw)
**Powiązania:** `docs/12_invariants.md`, `docs/13_delta_algebra.md`, `docs/14_mosaic.md`, `docs/30_sast_bridge.md`, `docs/41_pipelines.md`, `docs/50_ci_ops.md`, `docs/70_observability.md`, `docs/90_playbooks.md`
**Komponenty kodu:** `security/sast_bridge.py`, `analysis/policy.json`, `.githooks/*`, `mosaic/*`, `core/invariants.py`

---

## 0) Cel i zasady

GlitchLab utrzymuje bezpieczeństwo na **trzech poziomach**:

1. **Kod i zmiany (Δ-first):** statyczne reguły + kontekst AST/Mozaiki;
2. **Wykonanie i rozszerzalność:** sandbox pluginów/filtrów + polityki *fail-closed*;
3. **Łańcuch dostaw:** SBOM, audyt zależności, podpisy i kontrola wydań.

**Zasady nadrzędne**

* **Fail-closed:** w razie wątpliwości → blokuj (PR/hook/CI).
* **SSOT:** jedna polityka (`analysis/policy.json`) i jeden model zdarzeń (BUS/EGDB).
* **Minimalizm uprawnień:** tylko niezbędne capabilities (procesy, pliki, sieć).
* **Δ-first:** priorytetyzacja ryzyka oparta o **tokeny Δ** i **powiązanie AST↔Mozaika**.
* **Ślad audytowy:** każdy bieg generuje dowód: `.glx/security/**`, `backup/AUDIT_*.zip`.

---

## 1) Model zagrożeń (skrót)

**Zasoby:** kod źródłowy, presety, artefakty `.glx/*`, tajemnice (klucze/tokeny), pipeline CI/CD, GUI/APP.
**Powierzchnie ataku:** PR/commit (złe zmiany), pluginy/filtry (kod nieufny), zależności PyPI, wycieki sekretów, konfiguracja CI.
**Adwersarze:** przypadkowa regresja bezpieczeństwa, contributor z błędnym kodem, łańcuch dostaw (kompromitacja paczki), użytkownik wprowadzający filtr naruszający polityki.

---

## 2) Polityka bezpieczeństwa — SSOT

Główny plik polityki (ładowany przez `spec/loader.py` i `security/sast_bridge.py`):

```json
{
  "severity_threshold": "MEDIUM",
  "allow_paths": ["glitchlab/**", "filters/**"],
  "deny_paths": ["tests/**", "bench/**", "samples/**"],
  "suppress_tags": ["no-sec", "pyvulnscan:ignore"],
  "public_api_roots": ["glitchlab.core", "glitchlab.gui"],
  "gates": {
    "require_tests_on_security_fix": true,
    "human_review_for": ["CRITICAL", "secrets"]
  },
  "crypto": {
    "disallow": ["md5", "sha1", "pickle", "yaml.load"],
    "enforce_tls_verify": true
  },
  "sandbox": {
    "plugins_exec_mode": "isolated_process",
    "allow_network": false,
    "allow_fs_write": ["./.glx/**", "./backup/**"],
    "env_passthrough": ["PYTHONUTF8"]
  }
}
```

> **Źródło prawdy:** dopuszczalne API, kategorie ryzyka, *deny/allow lists*, reguły sandboxa i bramki. Zmiany polityki są wersjonowane i walidowane w CI.

---

## 3) SAST Bridge — skan, normalizacja i gating

Patrz szczegóły w `docs/30_sast_bridge.md`. Poniżej integracja bezpieczeństwa i bramek:

### 3.1 Wejścia

* SARIF/JSON: Bandit, Semgrep, Gitleaks, pip-audit/OSV + `pyvulnscan` (lokalny AST-lint).
* Context: indeks AST (UID), Δ (git-diff/AST-diff), Mozaika Φ(Δ), polityka (SSOT).

### 3.2 Normalized Finding (NF)

Minimalny rekord (kanoniczny):

```json
{
  "id": "NF-…",
  "tool": "bandit|semgrep|gitleaks|pip-audit|pyvulnscan",
  "rule_id": "B602|R002|…",
  "cwe": "CWE-78",
  "severity": "LOW|MEDIUM|HIGH|CRITICAL",
  "confidence": 0.0,
  "category": "RCE|SQLi|Secrets|TLS|Crypto|…",
  "message": "…",
  "location": {"file": "x.py", "line": 123},
  "ast_ref": "AST::<uid>",
  "mosaic_cell": "MOZ::<cell-id>",
  "delta_bind": "added|modified|unchanged",
  "evidence": {"snippet": "…"},
  "suppression": {"state": "none|local|policy", "reason": ""},
  "fingerprint": "sha256(…)"
}
```

### 3.3 Priorytetyzacja i scoring

```
risk_score = f(severity, confidence, reachability, delta_weight, public_api_weight)
```

* **delta_weight** ↑ dla znalezisk na ścieżce zmian, **public_api_weight** ↑ dla API eksportowanego.
* Progi gate: **α/β/Z** trzymane w `.glx/spec_state.json` (kalibracja robust-stats).

### 3.4 Bramki bezpieczeństwa (hooki/CI)

* `pre-commit`: blokuje **CRITICAL** i **secrets** (Gitleaks).
* `pre-push`/CI (`invariants_gate`): blokuje **`score>β`** lub **jakiekolwiek I***.
* Przy **secrets** generuje **procedurę remediacji** (patrz niżej).

### 3.5 Artefakty

* `.glx/security/merged.sarif`, `.glx/security/queue.json` (PQ), `.glx/security/fix_candidates.json` (opcjonalnie).
* W CI publikowane jako `glx-sast` (artefakt do pobrania).

---

## 4) Sekrety — wykrywanie i remediacja

**Źródło:** Gitleaks + heurystyki (prefiksy kluczy, entropia).
**Reguła:** *zero tolerance w kodzie* (z wyjątkiem bezpiecznych fixture’ów w testach z markerem).

### 4.1 Remediacja (automaty)

* **Wycofanie z historii** (git filter-repo) — jeśli świeży wyciek.
* **Rotacja klucza** w KMS/źródle — link/iac w playbookach.
* **Refactor**: odczyt z `.env`/secret store; dodanie do `.gitignore`.
* **Testy**: stub w miejscu sekretu, kontrakt w testach.

### 4.2 Lokalna adnotacja tłumienia (wyjątkowe)

```python
# glx:sec-suppress: reason="fixture key for unit test", scope="tests only"
```

> **Uwaga:** tłumienia są raportowane w **SAST Bridge** i wymagają **review**.

---

## 5) Sandbox pluginów/filtrów

Filtry (np. w `filters/`) i panele ładowane przez GUI/APP działają w **kwarantannie**.

### 5.1 Model wykonania

* **Oddzielny proces** (spawn), wymuszone **capabilities**: brak sieci, kontrolowany FS, limit CPU/ram.
* **IPC**: in-proc **BUS proxy** z walidacją typów (schematy).
* **Środowisko**: czyste `env`, tylko whitelisted zmienne (np. `PYTHONUTF8`).

### 5.2 Pseudokod uruchomienia

```python
def run_plugin_isolated(entry, args):
    caps = Caps(network=False, fs_write=[".glx/**","backup/**"], time_limit=5, mem_mb=512)
    with IsolatedProcess(caps) as proc:
        return proc.run(entry, args)
```

### 5.3 Lista zabronionych wzorców (lint SAST-lite)

* `exec`, `eval`, `os.system`, `subprocess.*(shell=True)`
* `pickle.loads`, `yaml.load` (bez SafeLoader)
* `requests.*(verify=False)`, TLS bez weryfikacji
* bezpośrednie użycie `hashlib.md5/sha1` do bezpieczeństwa

---

## 6) Inwarianty bezpieczeństwa (I1–I4)

Rozszerzenie `docs/12_invariants.md` o implikacje bezpieczeństwa:

* **I1 (monotoniczność Δ):** żadna poprawka nie zwiększa *Δ-ryzyka* powiązanego z tokenami (np. `ΔIMPORT` do niezaufanych przestrzeni).
* **I2 (Φ/Ψ bliskie domknięcie):** refaktor nie może „zgubić” wiązań AST↔Mozaika (unikamy ukrytych “shadow flows”).
* **I3 (komutacja z Δ):** mapa Δ w Mozaice musi odpowiadać AST-diff — inaczej sygnał do **Guard** (blokada).
* **I4 (public API stability):** zmiany w eksportach muszą przejść adapter lub *guarded deprecation*; w przeciwnym razie ryzyko *confused deputy*.

Bramki **I*** są egzekwowane w: `pre-push` i w jobie CI `invariants_gate`.

---

## 7) Supply-chain security (SBOM, zależności, release)

### 7.1 Zależności

* **Pinning:** `pyproject.toml` (zakresy) + opcjonalny `constraints.txt` na CI.
* **Audyt:** `pip-audit`/OSV (część SAST Bridge).
* **Blokada instalacji niezweryfikowanych wheels** (opcjonalnie: `--require-hashes`).

### 7.2 SBOM i podpisy

* **SBOM (opcjonalnie):** generowany z `pipdeptree`/`cyclonedx-bom`.
* **Podpis tagów i wydań:** kluczem GPG (maintainer build).
* **Protected branches / CODEOWNERS:** polityka wymagająca review bezpieczeństwa.

### 7.3 Wydania (patrz `docs/80_release.md`)

* Kanały **beta/stable**, changelog z Conventional Commits, artefakty binarne GUI (opcjonalnie) + checksumy.

---

## 8) Dzienniki, PII i retencja

* **Redakcja** tajemnic w logach (maskowanie wartości na podstawie *hintów*).
* **Retencja** artefaktów `.glx/security/**`: domyślnie 90 dni (konfigurowalne).
* **AUDIT ZIP:** `backup/AUDIT_*.zip` zawiera metadane przebiegów (czas, gałąź, wersja polityki).
* **Zasada minimalizacji PII:** GUI/APP nie zbiera danych osobowych; jeżeli panel dodaje takie pola, musi wyraźnie je oznaczyć i wyłączyć z eksportu.

---

## 9) Integracja z CI/Ops (skrót)

Patrz `docs/50_ci_ops.md`. Najważniejsze punkty:

* **Hooki:** `pre-commit` (SAST-lite + secrets), `pre-push` (I* + gate α/β/Z), `post-commit` (artefakty Δ).
* **Actions:** job `sast_bridge` → artefakty `glx-sast`; job `invariants_gate` → bramka finalna.
* **Komentarz do PR:** podsumowanie tokenów Δ, naruszeń I*, NF>TH i checklist remediacji.

---

## 10) Checklisty bezpieczeństwa

### 10.1 PR Security Gate (dla recenzenta)

* [ ] Brak `exec/eval/os.system/subprocess(shell=True)`
* [ ] Brak `pickle/yaml.load` (bez SafeLoader)
* [ ] TLS: brak `verify=False`
* [ ] Crypto: brak `md5/sha1` do bezpieczeństwa
* [ ] Brak sekretów w kodzie i historii commita
* [ ] Zmiany w public API: adapter/test/migracja
* [ ] Wynik `invariants_gate`: ✅ (score ≤ β i brak I*)

### 10.2 Release Readiness

* [ ] SBOM wygenerowany (jeśli wymagany)
* [ ] pip-audit/OSV: brak CRITICAL/HIGH
* [ ] CI: pełny zielony, artefakty `.glx/*` dołączone
* [ ] Changelog (Conventional Commits)
* [ ] Tag podpisany (opcjonalnie GPG)

---

## 11) Kontrakt BUS/EGDB (bezpieczeństwo)

**Tematy BUS (publikowane/odbierane):**

* `security.alert` — CRITICAL/Secrets + lokalizacja + Δ-kontekst
* `guard.policy` — aktualizacja polityki (freeze/unfreeze progów)
* `heal.*` — (jeśli włączony Self-Healing) propozycje napraw + weryfikacja

**EGDB (minimum):**

* `security_findings` (NF), `security_queue` (PQ), `heal_candidates`, `heal_outcomes` (jeśli Self-Healing)

---

## 12) Przykłady CLI

### 12.1 Ręczny bieg SAST Bridge (lokalnie)

```bash
python -m glitchlab.security.sast_bridge scan \
       --out .glx/security \
       --severity-threshold MEDIUM
```

### 12.2 Gate inwariantów (lokalnie lub w CI)

```bash
python -m glitchlab.core.invariants --ci-gate \
       --spec .glx/spec_state.json \
       --delta .glx/delta_report.json
```

---

## 13) Incydenty i playbooki

Szczegóły w `docs/90_playbooks.md`. Szybka mapa:

* **Wycieki sekretów:** *revoke/rotate → filter-repo → PR z refaktorem → audyt*.
* **Dryf progów:** *freeze thresholds → zbiór etykiet → recalibrate → unfreeze*.
* **Zależność z CVE:** *pip-audit → fix candidate (bump + changelog) → gate I**.

---

## 14) Roadmap bezpieczeństwa (MVP → +)

1. **MVP (już):** SAST-lite w hookach, SAST Bridge w CI, gate I*, sandbox procesowy.
2. **+1:** SBOM (CycloneDX), podpisy wydań, CODEOWNERS + branch protection.
3. **+2:** Self-Healing *security-aware* (automatyczne PR z poprawkami ∧ testami).
4. **+3:** Telemetria bezpieczeństwa w HUD (trend FPR/MTTR-Sec, top patterns).

---

## 15) Źródła prawdy (do utrzymania)

* **Polityka:** `analysis/policy.json`
* **Bramki:** `glitchlab/core/invariants.py` + `.glx/spec_state.json`
* **SAST Bridge:** `security/sast_bridge.py` (+ adaptery)
* **Hooki/CI:** `.githooks/*`, `.github/workflows/ci.yml`
* **Dokumentacja:** niniejszy plik + `docs/30_sast_bridge.md`, `docs/50_ci_ops.md`

> Każda zmiana w interfejsach/ścieżkach musi być odbita w dokumentacji i testach kontraktowych.
