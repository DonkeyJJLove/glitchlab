### `docs/22_analytics.md`

# GlitchLab Analytics (Δ-First Intelligence Plane)

**Cel:** warstwa analityczna spajająca rdzeń (core/analysis/mosaic) z **AgentemAI** i HUD.  
Działa *diff-first* (Δ), zasila **EGDB** ustrukturyzowanymi sygnałami, publikuje zdarzenia na **BUS**, egzekwuje **inwarianty** i przygotowuje kontekst dla napraw (SAST-Bridge / Self-Healing).

---

## 1. Zakres i rola

Analytics to ciągły proces:
1) **Ekstrakcja** cech i tokenów Δ z kodu i artefaktów,  
2) **Kalibracja** progów (α/β/Z) oraz detekcja driftu,  
3) **Korelacja** w EGDB (zdarzenia ↔ komórki mozaiki ↔ pliki/UID AST),  
4) **Publikacja** decyzji/alertów na BUS (gates, rekomendacje, propozycje napraw),  
5) **Zasilenie** AgentaAI kontekstem (RAG), oraz HUD danymi na żywo.

**Źródła prawdy:** `/spec/invariants.yaml`, `/spec/schemas/*`, **EGDB**, artefakty `.glx/*`.

---

## 2. Struktura w repo

```

analysis/
ast_delta.py            # AST-diff → tokeny Δ (ADD_FN, MODIFY_SIG, ΔIMPORT, …)
ast_index.py            # indeks węzłów (Define/Use/Call), UID-y
ast_mosaic_analyzer.py  # wiązanie AST ↔ Mozaika (Φ), heatmapy Δ
diff.py                 # Δ obrazów (jeśli dotyczy) + statystyki
exporters.py            # DTO/bundle do HUD i do EGDB
formats.py              # heurystyki formatów
git_io.py               # snapshoty/diffy repo
impact.py               # reachability/taint heuristics, wpływ zmian
metrics.py              # entropia, krawędzie, RMS… (0–1)
mosaic_adapter.py       # adapter rastrów/komórek mozaiki
phi_psi_bridge.py       # sprzężenie Φ/Ψ (AST↔Mozaika)
reporting.py            # raporty JSON/MD dla CI/HUD
spectral.py             # FFT + pasma
policy.json             # lokalne reguły Analytics (dodatkowe do /spec)

````

Powiązania:
- **BUS**: `bus.py` (tematy `analytics.*`, `invariants.*`, `delta.*`, `heal.*`),
- **EGDB**: tabele i widoki z `docs/21_egdb.md` (rozszerzamy o prefiksy `analytics_*`),
- **SAST-Bridge**: konsument sygnałów Δ, producent FixCandidate.

---

## 3. Dane i artefakty

### 3.1. Wejścia
- Git diff (`git_io.py`) + AST (`ast_delta.py`, `ast_index.py`),  
- Mozaika i projekcje (`mosaic_adapter.py`, `ast_mosaic_analyzer.py`),  
- Metryki obrazu/kodu (`metrics.py`, `spectral.py`),  
- Polityki (`/spec/invariants.yaml`, `analysis/policy.json`).

### 3.2. Wyjścia (artefakty `.glx/*`)
- `.glx/metrics.parquet` — wektor cech Δ per commit/PR,  
- `.glx/delta_report.json` — tokeny Δ, fingerprint, naruszenia I*, score,  
- `.glx/spec_state.json` — aktualne progi (kwantyle, MAD, EWMA), drift,  
- `.glx/mosaic_*.png` — heatmapy Δ (Φ(Δ_AST)),  
- `.glx/commit_analysis.json` — telemetria pętli (czasy, liczniki, ostrzeżenia).

---

## 4. Model obliczeń (Δ-first)

### 4.1. Tokeny i fingerprint
- `ast_delta.py` → lista tokenów: `ADD_FN | RENAME | MODIFY_SIG | ΔIMPORT | ΔTYPE_HINTS | ΔTESTS | …`  
- `fingerprint` = stabilny skrót (histogram + bigramy tokenów).

### 4.2. Sprzężenie Φ/Ψ
- **Φ:** AST→Mozaika — mapujemy `ast_ref` na komórki i budujemy heatmapę Δ,  
- **Ψ:** Mozaika→AST — wykorzystujemy maskę/ROI do zawężenia AST do obszarów aktywnych Δ.

### 4.3. Wektor cech (features)
Przykład: `ΔLOC, ΔCC, ΔImports, ΔFanIn/Out, ΔTests, SSIM_drop, PSNR, secrets_hits, SAST_hits, reachability, public_api_weight, delta_weight`.

### 4.4. Score i bramki
- `score = wᵀ·z(Δ)` (z-score/robust-z z MAD),  
- Progi **α/β/Z** z **kwantyli** strumieniowych (t-digest/P²) + EWMA/MAD,  
- Gates: `score ≤ α → ok`, `α<score≤β → review`, `>β → block`, `>Z → hard block`.

---

## 5. Integracja z AgentemAI (RAG + decyzje)

Analytics publikuje do AgentaAI:
- **Kontekst RAG**: top-K fragmentów AST/diff/zdarzeń EGQL związanych z danym naruszeniem,  
- **Hipotezy**: „dlaczego score>β” (feature attribution),  
- **Zlecenia**: `heal.proposal` (gdy reguła → szybki auto-fix) lub `heal.request_review`.

AgentAI odsyła:
- `heal.proposal` (patch/test/config) + `rationale`,  
- `heal.verify.*` (wyniki sandbox/CI) — trafiają do EGDB i HUD.

---

## 6. Schemat zdarzeń (BUS)

**Publikowane:**
- `analytics.delta.ready { sha, tokens, fingerprint, features, score }`  
- `analytics.invariants.violation { kind:I1..I4, evidence, heatmap_ref }`  
- `analytics.spec.thresholds.update { alpha, beta, zeta, drift? }`  
- `analytics.security.alert { tool, rule_id, file, line, severity, ast_ref }`

**Konsumowane:**
- `core.step.finished`, `run.start/done/error`, `sast.findings.ready`, `heal.*`

---

## 7. EGDB (rozszerzenia)

Proponowane tabele (uzupełnienie `docs/21_egdb.md`):

```sql
CREATE TABLE analytics_commits (
  sha TEXT PRIMARY KEY, ts REAL, branch TEXT, author TEXT,
  fingerprint TEXT, score REAL, alpha REAL, beta REAL, zeta REAL
);

CREATE TABLE analytics_features (
  sha TEXT, name TEXT, value REAL,
  PRIMARY KEY (sha, name)
);

CREATE TABLE analytics_tokens (
  sha TEXT, token TEXT, count INTEGER,
  PRIMARY KEY (sha, token)
);

CREATE TABLE analytics_invariants (
  id TEXT PRIMARY KEY, sha TEXT, kind TEXT, evidence_json TEXT, level TEXT, ts REAL
);
````

Widoki:

* `vw_delta_hotspots` (komórki o najwyższej energii ΔZ),
* `vw_gate_failures` (naruszenia I* per moduł),
* `vw_spec_drift` (historia progów i freeze okien).

---

## 8. API (przykład, Python)

```py
# analytics/publisher.py (przykład wzorca)
from typing import Dict, Any
from glitchlab.bus import Bus

def publish_delta_ready(bus: Bus, sha: str, tokens: Dict[str, int], fp: str, feats: Dict[str, float], score: float):
    bus.emit("analytics.delta.ready", {
        "sha": sha, "tokens": tokens, "fingerprint": fp,
        "features": feats, "score": score,
    })

def publish_invariant(bus: Bus, sha: str, kind: str, evidence: Dict[str, Any], level: str = "warn"):
    bus.emit("analytics.invariants.violation", {
        "sha": sha, "kind": kind, "evidence": evidence, "level": level,
    })
```

---

## 9. CI/Hooki (jak używać)

* **pre-commit**: szybkie `ast_delta` + reguły bezpieczeństwa (Δ-tokens, secrets),
* **post-commit**: zapis `.glx/delta_report.json`, `.glx/spec_state.json`, publikacja `analytics.delta.ready`,
* **pre-push**: bramki α/β/Z (fail-closed), publikacja `analytics.invariants.violation`.

---

## 10. HUD (panele)

* **Delta Inspector**: histogram tokenów, fingerprint, heatmapa Δ, lista naruszeń I*, linki do plików/linii,
* **Spec Monitor**: α/β/Z w czasie, drift, freeze okna, „dlaczego zablokowano”.

---

## 11. Rozszerzalność i bezpieczeństwo

* Adaptery źródeł (nowe metryki/źródła),
* Polityki w `analysis/policy.json` + `/spec/invariants.yaml`,
* Sandbox dla analizy zewnętrznych pluginów (oddzielny proces, ograniczone capabilities),
* Telemetria prywatna → redakcja PII i sekrety.

---

## 12. Minimalny scenariusz E2E

1. Commit → **post-commit** uruchamia Analytics (Δ, tokeny, features, score),
2. Publikacja `analytics.delta.ready` + zapis artefaktów `.glx/*`,
3. Gate w **pre-push/CI**; gdy `score>β` → `analytics.invariants.violation`,
4. AgentAI proponuje patch/test; weryfikacja → wynik do EGDB + HUD.

---

**Koniec dokumentu.**

```
::contentReference[oaicite:0]{index=0}
```