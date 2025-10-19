# GlitchLab — **SAST-Bridge** (Security Findings ⇄ AST/Mozaika ⇄ Generator) · `docs/30_sast_bridge.md`

> Specyfikacja **normatywna** modułu SAST-Bridge. Moduł **ujednolica** wyniki skanerów bezpieczeństwa (SARIF/JSON), **wiąże** je z AST i Mozaiką (Φ/Ψ, Δ), **priorytetyzuje** oraz przygotowuje **kontrakty napraw** dla generatora (FixCandidate).
> Spójne z: `docs/11_spec_glossary.md`, `docs/12_invariants.md`, `docs/13_delta_algebra.md`, `docs/20_bus.md`, `docs/21_egdb.md`.

---

## 0) Zakres i konwencje

* Słowa **MUST / SHOULD / MAY** są normatywne (RFC 2119).
* SAST-Bridge działa jako **usługa** (daemon) lub **CLI**.
* Nośniki danych: **SARIF 2.1.0** (preferowane), JSON specyficzny dla narzędzia (adaptery).
* Wszystkie ładunki **MUST** walidować się względem schematów w `/spec/schemas/`.

---

## 1) Rola i cele

1. **Ingest**: Bandit, Semgrep, Gitleaks, pip-audit/OSV, pyvulnscan (AST-lint lokalny).
2. **Normalizacja ➜ NF** (Normalized Finding) + **deduplikacja**.
3. **Wiązanie do AST/Mozaiki** (Φ/Ψ) i Δ (`added|modified|unchanged`).
4. **Priorytetyzacja** (risk_score) z **reachability** i wagami polityki.
5. **Kontrakt naprawy** (**FixCandidate**) dla generatora + walidacja **I1–I4**.
6. **Integracja** z **BUS** (tematy `SAST.*`) i **EGDB** (`sast_*`, patrz `docs/21_egdb.md`).

---

## 2) Architektura (przepływ)

```
[Skanery] → [Adapters] → [Normalizer] → [Dedup] → [AST/Φ bind] → [Δ-bind] → [Prioritizer]
      SARIF/JSON       NF(row)         NF(uniq)     ast_ref, cell      added/mod      PQ
                                             ↘ - (strzała skośna w prawy dół)
                                       [EGDB.persist] ← [BUS.publish]
                                                 ↘ - (strzała skośna w prawy dół)
                                      [FixPlanner → FixCandidate → Gates I1–I4 → Verify]
```

---

## 3) Wejścia / Wyjścia

### 3.1 Wejścia

* **Snapshot kodu** (drzewo plików + hash per plik, SHA-256) — **MUST**.
* **Δ (git diff)** z PR/commitu — **SHOULD** (dla `delta_bind` i wag).
* **Artefakty SAST** (SARIF/JSON) — **MUST**.
* **Kontekst AST** (indeks węzłów + UID) — **MUST**, gdy chcemy `ast_ref`.
* **Polityka** (progi/wagi/allow/deny) — **MUST**.

### 3.2 Wyjścia (logiczne)

* **NF** (Normalized Findings) — strumień i tabela `sast_findings`.
* **PQ** (Prioritized Queue) — pamięć operacyjna + `sast_fix_queue`.
* **FixCandidate** — `SAST.FixRequested`/`SAST.FixProposed` + `sast_fix_queue`/`sast_fix_verification`.
* **Raporty**: `.glx/security/merged.sarif`, `.glx/security/queue.json`.

---

## 4) Model danych (kontrakty)

### 4.1 Normalized Finding (NF)

```json
{
  "id": "NF-<ulid>",                     "tool": "bandit|semgrep|gitleaks|pip-audit|osv|pyvulnscan",
  "rule_id": "B602|p/requests.verify_false|...", "cwe": "CWE-78",
  "severity": "LOW|MEDIUM|HIGH|CRITICAL", "confidence": 0.0,
  "category": "RCE|SQLi|Secrets|TLS|Crypto|Deserialization|SSRF|Defaults",
  "message": "…",
  "location": { "file": "path.py", "line": 123, "col": 8, "endLine": 123 },
  "ast_ref": "AST::<node-uid>",          "mosaic_cell": "MOZ::<cell-id>",
  "delta_bind": "added|modified|unchanged",
  "evidence": { "snippet": "line text", "trace": [] },
  "suppression": { "state": "none|local|policy", "reason": "" },
  "fingerprint": "sha256(snippet|rule|path|norm)"
}
```

**Wymagania:** `id`, `tool`, `rule_id`, `severity`, `confidence`, `location`, `fingerprint` **MUST**.

### 4.2 Prioritized Queue (PQ) — rekord

```json
{
  "nf_id": "NF-…",
  "risk_score": 0.0,
  "reasons": [
    {"feature": "severity", "value": "HIGH", "weight": 1.0, "contrib": 0.4},
    {"feature": "reachability", "value": 0.8, "weight": 0.7, "contrib": 0.56},
    {"feature": "delta_weight", "value": 1.2, "weight": 0.3, "contrib": 0.36}
  ],
  "queue": "block|review|fixable|triage"
}
```

### 4.3 FixCandidate (kontrakt)

```json
{
  "nf_id": "NF-…",
  "pattern": "subprocess shell=True",
  "fix_hint": "set shell=False & pass argv list",
  "patch_spec": {
    "file": "x.py",
    "hunk": "@@ -123,7 +123,7 @@\n- subprocess.call(cmd, shell=True)\n+ subprocess.call(['sh', '-c', cmd])\n"
  },
  "tests_required": ["unit:security::subprocess_no_shell"],
  "constraints": {"I1..I4": "must-hold", "Δ-limit": "no public API break"},
  "review_gate": "human|auto"
}
```

---

## 5) Adaptery skanerów (ingest)

| Narzędzie      | Format     | Mapowanie → NF (skrót)                                                              |
| -------------- | ---------- | ----------------------------------------------------------------------------------- |
| **Bandit**     | SARIF/JSON | `ruleId→rule_id`, `level→severity`, `message→msg`, `locations→location`, heur.`cwe` |
| **Semgrep**    | SARIF/JSON | `check_id→rule_id`, `extra.metadata.cwe`→`cwe`, `severity`, `start/end`             |
| **Gitleaks**   | JSON       | `RuleID→rule_id`, `Secret`→`evidence.snippet`, `File`, `StartLine`                  |
| **pip-audit**  | JSON       | `vuln.id→rule_id`, `spec→package`, `fix_version`, `advisory.cwe[]`                  |
| **OSV**        | JSON       | `id→rule_id`, `affected[].ranges`                                                   |
| **pyvulnscan** | JSON       | Nasz AST-lint: `rule_id`, `category`, `evidence`, `confidence`                      |

**Zasady:**

* Adaptery **MUST** zachować `raw` (oryginał) w telemetrii (BUS/EGDB artifacts).
* Jeśli brak `cwe` → **SHOULD** uzupełnić z mapy reguł (`spec/policies/sast_rules.yaml`).
* `severity` skaluje się do {LOW,MEDIUM,HIGH,CRITICAL} wg tablicy konwersji.

---

## 6) Normalizacja i deduplikacja

**Klucz dedupu:**

```
(file, norm_line±N, norm(rule_id), fp(snippet|rule|path|norm))
```

* `norm_line±N` = linia ±2 (konfigurowalne).
* Fuzja duplikatów: `severity = max()`, `confidence = average()`, `evidence.trace = union`.
* `fingerprint` **MUST** być stabilny w czasie przy braku zmian semantycznych (tolerancja na rewrap/whitespace).

**Pseudokod:**

```python
for finding in ingest_all():
    nf = normalize(finding)
    key = dedup_key(nf)
    if key in cache:
        cache[key] = fuse(cache[key], nf)
    else:
        cache[key] = nf
emit(cache.values())
```

---

## 7) Wiązanie z AST i Mozaiką (Φ/Ψ, Δ)

1. **AST-bind**: `ast_ref = nearest(node ∈ {Call, Attribute, Assign, Import})` (UID z indeksu).
2. **Φ-projekcja → Mozaika**: `mosaic_cell = argmax_c weight(Φ(ast_ref)→cell)` (`phi_links`, patrz EGDB).
3. **Δ-bind**:

   * jeśli `file` w Δ: `delta_bind = added|modified` (wg hunka/zakresu linii), inaczej `unchanged`.
   * **delta_weight** = `1.0` (unchanged) / `1.2` (modified) / `1.5` (added) — **konfigurowalne**.

---

## 8) Priorytetyzacja (risk_score)

**Formuła (domyślna):**

```
risk_score = w_s*sev + w_c*conf + w_r*reach + w_d*delta_weight + w_p*public_api + w_k*category_bias + w_sec*secret_flag
```

gdzie:

* `sev ∈ {0.25,0.5,0.75,1.0}`, `conf ∈ [0,1]`, `reach ∈ [0,1]` (statyczne dojście z public API/endpointów),
* `public_api ∈ {0,1}`, `secret_flag ∈ {0,1}`.

**Domyślne wagi (policy):**

| cecha         | waga |
| ------------- | ---- |
| severity      | 1.0  |
| confidence    | 0.7  |
| reachability  | 0.7  |
| delta_weight  | 0.3  |
| public_api    | 0.3  |
| category_bias | 0.2  |
| secret_flag   | 1.2  |

**Próg → kolejka:**

* `score ≥ 1.8` → **block** (CRITICAL)
* `1.2 ≤ score < 1.8` → **review**
* `0.8 ≤ score < 1.2` → **fixable**
* `< 0.8` → **triage**

Progi **MAY** adaptować się wg `spec_thresholds` (kwantyle per moduł).

---

## 9) Generator poprawek (FixPlanner)

**Tabela reguł (skrót):**

| Wzorzec                       | Fix (hint)                                      |                                                        |
| ----------------------------- | ----------------------------------------------- | ------------------------------------------------------ |
| `subprocess.*(…, shell=True)` | `shell=False`, argv list                        |                                                        |
| `yaml.load(...)`              | `yaml.safe_load(..., Loader=SafeLoader)`        |                                                        |
| `requests.*(verify=False)`    | `verify=True` + CA bundle / `Session`           |                                                        |
| `hashlib.(md5                 | sha1)(...)`                                     | `sha256()`/`blake2b()`, tolerancja dla checksum testów |
| `pickle.loads`/`dumps`        | JSON/dataclass/pydantic                         |                                                        |
| SQL + f-string/format         | `cursor.execute(sql, params)`                   |                                                        |
| Gitleaks: sekrety w kodzie    | Usunięcie, rotacja, przeniesienie do `.env`/KMS |                                                        |
| pip-audit/OSV: wersje         | Bump + changelog link + constraints/test smoke  |                                                        |

**Kontrakt FixCandidate** → patrz §4.3. Każdy **FixProposed** **MUST** przejść bramki **I1–I4** i podażać `tests_required`.

---

## 10) Inwarianty (I1–I4) i gates

* **I1 — ΔZ monotonicity**: łatka nie może zwiększać ΔZ w obszarze dotkniętym (mierzonym Mozaiką).
* **I2 — Φ/Ψ near-closure**: projekcje po łatce muszą pozostać w progu (α/β/Z).
* **I3 — Commutation**: `|| Φ(Δ_AST) – Δ_MOZ || ≤ β`.
* **I4 — Public API stability**: brak złamań podpisów i kontraktów o oznaczonym stabilnym poziomie.

**Gates w CI**: naruszenia tworzą rekordy w `invariants_results` i **MUST** blokować merge zgodnie z polityką.

---

## 11) BUS — tematy i ładunki

* `SAST.ScanRequested@v1 { scope, delta?, policy }`
* `SAST.FindingsReady@v1 { findings: [NF], meta }`
* `SAST.Prioritized@v1 { queue: [ {nf_id, risk_score, queue} ] }`
* `SAST.FixRequested@v1 { fc: FixCandidate }`
* `SAST.FixProposed@v1 { patch, rationale, invariants_check }`
* `SAST.FixValidated@v1 { status, tests, delta_phi_psi }`

**Idempotencja:** `event.id`/`idempotency_key` **MUST** być respektowane przez ingest do EGDB.

---

## 12) EGDB — integracja (minimum)

* **NF → `sast_findings`** (ON CONFLICT `fingerprint` DO NOTHING/UPDATE).
* **PQ/Fix** → `sast_fix_queue`, `sast_fix_verification`.
* **Powiązania**: `ast_ref` ↔ `ast_nodes.uid`, `mosaic_cell` ↔ komórka Mozaiki (`phi_links`).
* **Widoki**: `vw_sast_backlog`, `vw_heal_success_by_kind` (por. `docs/21_egdb.md`).

---

## 13) Konfiguracja (`tile.yaml`)

```yaml
module: SAST-Bridge
inputs:
  scanners: [bandit, semgrep, gitleaks, pip-audit, osv, pyvulnscan]
  formats: [sarif, json]
policy:
  severity_threshold: MEDIUM
  allow_paths: ["src/**"]
  deny_paths: ["tests/**", "examples/**"]
  suppress_tags: ["pyvulnscan: ignore", "no-sec"]
prioritization:
  delta_weight: 1.2
  public_api_weight: 1.3
  weights:
    severity: 1.0
    confidence: 0.7
    reachability: 0.7
    delta_weight: 0.3
    public_api: 0.3
    category_bias: 0.2
    secret_flag: 1.2
outputs:
  sarif_merged: ".glx/security/merged.sarif"
  queue: ".glx/security/queue.json"
gates:
  require_tests: true
  human_review_for: ["CRITICAL", "secrets"]
```

---

## 14) CLI i usługa

**CLI (dev):**

```bash
glx-sast scan --inputs out/*.sarif --repo . --delta HEAD~1..HEAD --policy spec/policies/sast.yaml \
  --egdb postgres://... --emit-bus
```

**Daemon (prod):**

* Subskrybuje `SAST.ScanRequested@v1`, zapisuje do EGDB, publikuje `SAST.FindingsReady@v1` i `SAST.Prioritized@v1`.
* Na żądanie: generuje **FixCandidate** i publikuje `SAST.FixRequested@v1`.

---

## 15) Bezpieczeństwo

* **MUST NOT** wykonywać skanowanego kodu.
* **MUST** cenzurować wartości sekretów — przechowuj **fingerprint** i `_redacted`.
* **MUST** działać w środowisku izolowanym (fs read-only dla repo; brak sieci dla adapterów, chyba że pobiera CVE).
* **Patches** **MUST NOT** wprowadzać `shell=True`, `pickle`, niezweryfikowanego TLS.

---

## 16) Obserwowalność i metryki

* **MTTR-Sec**, **Fix-Adoption-Rate**, **False-Positive-Rate**, **Regressions-after-Patch**.
* **Δ-Risk-Density** per moduł, **Top patterns** (np. `verify=False`).
* **Coverage**: % plików ze skanem, % NF z `ast_ref`.
* Eksport do EGDB (`runs`, `steps`, widoki §6 `docs/21_egdb.md`).

---

## 17) Testy (plan)

* **Unit**: normalizery, dedup, mapy reguł→CWE/kategoria.
* **Fixtures**: złote pliki SARIF dla Bandit/Semgrep/Gitleaks/pip-audit/OSV.
* **AST-bind**: syntetyczne pliki z `Call/Assign/Import` (libcst).
* **E2E**: ingest→NF→PQ→FixCandidate→I1–I4→Verify (na repo fixture).
* **Security**: wstrzyknięcia sekretów — weryfikacja redakcji.
* **Stabilność FP**: ten sam NF ≠ duplikat po reflow/komentarzach.

---

## 18) Wersjonowanie i kompatybilność

* **Tematy BUS**: `@vN` w nazwie — zmiana niekompatybilna → nowa wersja.
* **JSON Schemas**: `spec/schemas/sast_nf.v1.json`, `sast_fix_candidate.v1.json`.
* **Adaptery**: semver per adapter (`bandit@1.0.0`).
* **Migracje EGDB**: Alembic, patrz `docs/21_egdb.md`.

---

## 19) Quickstart (minimalny)

```bash
# 1) Uruchom skanery (przykładowo)
bandit -r src -f sarif -o .glx/scan_bandit.sarif
semgrep --config p/r2c-ci --sarif -o .glx/scan_semgrep.sarif
gitleaks detect --report-format sarif --report-path .glx/scan_gitleaks.sarif
pip-audit -r requirements.txt -f json -o .glx/scan_pip_audit.json

# 2) SAST-Bridge → NF/PQ/Fix
glx-sast scan --inputs .glx/scan_*.sarif .glx/scan_*.json --repo . --delta HEAD~1..HEAD \
  --policy spec/policies/sast.yaml --egdb postgres://… --emit-bus
```

---

## 20) Checklist wdrożeniowy

* [ ] Schematy JSON (`sast_nf.v1.json`, `sast_fix_candidate.v1.json`).
* [ ] Adaptery: Bandit/Semgrep/Gitleaks/pip-audit/OSV/pyvulnscan.
* [ ] Deduplikacja + fingerprint stabilny.
* [ ] AST/Φ bind + Δ-bind.
* [ ] Priorytetyzacja + polityka progów (integrowana z `spec_thresholds`).
* [ ] BUS publish + EGDB persist.
* [ ] FixPlanner (co najmniej 5 reguł wysokiego ROI) + Gates I1–I4.
* [ ] Testy unit/E2E + fixtury SARIF.
* [ ] CI job: uruchomienie w PR, artefakty `.glx/security/*`.

---

## 21) Status pliku

✅ **Final (Spec — SAST-Bridge)**
Zmiany w kontraktach **MUST** aktualizować odpowiednie schematy i wersje tematów BUS.
