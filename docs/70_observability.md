````markdown
# docs/70_observability.md
# Observability & EGQL w GlitchLab (BUS → EGDB → HUD)

> Ten dokument definiuje **jak mierzymy, logujemy i korelujemy** zdarzenia w GlitchLab: od hooków i rdzenia (Core/Analysis/GUI), przez **BUS**, do **EGDB** i **HUD**. Opisuje też **EGQL** – lekki język zapytań po zdarzeniach/Δ-metrykach, oraz referencyjne **widoki** i **artefakty `.glx/*`**.

---

## 1) Cele

- **Widoczność end-to-end:** commit → Δ-analiza (AST/Mozaika) → walidatory I1–I4 → decyzja (guard/heal) → GUI/HUD.
- **Spójny model danych:** ten sam kontrakt dla hooków, rdzenia i GUI, z wersjonowanymi schematami.
- **Korelacje i dowody:** zapytania EGQL/SQL, ścieżki przyczynowe (PATH), fingerprint Δ, heatmapy.
- **Produkcyjna prostota:** domyślnie **SQLite** (w `.glx/grammar/`), opcjonalnie Postgres. Repo już przewiduje tę topologię (patrz layout `.glx/grammar`). :contentReference[oaicite:0]{index=0}

---

## 2) Model Obserwowalności

### 2.1 Jednostki

- **Event** (zdarzenie): `ts, topic, kind, sha, branch, src, data{...}` – minimalna cegiełka strumienia.
- **Metric** (metryka): punkt pomiarowy (np. SSIM/PSNR, ΔCC, ΔTokens).
- **Artifact**: plik wynikowy (HUD/CI), trzymany poza DB, z metadanymi w DB (ścieżka, hash).
- **Run/Span/Trace**: korelacja eventów w czasie (commit/pipeline/test).

### 2.2 Taksonomia topiców (BUS)

Minimalny rdzeń (wycinek), jak w `.glx/bus.yaml`:

```yaml
topics:
  - run.start              # początek wykonania (GUI/CLI/CI)
  - run.done               # zakończenie (status=ok)
  - run.error              # zakończenie (status=error)
  - code.delta             # podpis Δ (tokens/fingerprint/loc)
  - invariants.violation   # I1–I4 + dowód
  - security.alert         # SAST/secret/deps (NF/PQ/FC)
  - heal.proposal          # propozycja naprawy (patch/test/config)
  - heal.verify.done       # wynik sandbox/CI/mutation
````

Tematy te są już zakotwiczone w repozytoryjnym opisie BUS. 

### 2.3 Rejestr schematów (SSOT)

Rejestr plików schematów (URI/nazwa/wersja) trzymamy w `.glx/schemas/registry.json` (kanoniczne nazwy i wersje). Dzięki temu walidatorzy wiedzą **jakiego** JSON-payloadu oczekiwać (np. `code.ast.built-1.json`, `git.delta.ready-1.json`). 

---

## 3) Instrumentacja & Emiter

### 3.1 Format logów (JSON Lines)

**Wszystkie** komponenty logują w JSON (UTF-8), minimalny kontrakt:

```json
{
  "ts": "2025-10-04T21:23:02.550Z",
  "topic": "code.delta",
  "kind": "report|define|measure|violation|proposal|verify",
  "sha": "9d9f399",
  "branch": "master",
  "src": "analysis.ast_delta",
  "corr": "c4a1e2f5...",     // correlation id (run/commit)
  "data": { "... domain fields ..." }
}
```

> Stan repozytorium/CI przechowujemy też jako **artefakty** (np. `.glx/commit_snippet.txt`, `analysis/logs/commit_*.json`). 

### 3.2 Emiter (Python, skrót)

```python
from typing import TypedDict, Any
import json, time, uuid, sys

class Event(TypedDict, total=False):
    ts: str; topic: str; kind: str; sha: str; branch: str; src: str; corr: str; data: dict[str, Any]

def emit(ev: Event, sink=sys.stdout):
    if "ts" not in ev: ev["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
    if "corr" not in ev: ev["corr"] = uuid.uuid4().hex
    sink.write(json.dumps(ev, ensure_ascii=False) + "\n")
```

---

## 4) EGDB – schemat i widoki

### 4.1 Tabele rdzeniowe

* `glx_events(topic, ts, kind, sha, branch, src, corr, data JSONB)`
* `glx_metrics(ts, sha, name, value, scope, aux JSONB)`
* `glx_artifacts(ts, sha, kind, path, hash, meta JSONB)`
* `glx_deltas(ts, sha, file, tokens JSONB, fingerprint, loc_add, loc_del, aux JSONB)`

### 4.2 Widoki referencyjne (skrót)

> Repo utrzymuje plik **SQL widoków** w przestrzeni `.glx/grammar/views.sql` (ładowany w migracji **DDL → views → seed**).

* `vw_runs_recent` – grupowanie `run.*` w interwałach z KPI (czas, status).
* `vw_invariants_violations` – ostatnie naruszenia I1–I4 z dowodami.
* `vw_hotspots` – pliki o wysokim **temporal coupling** (co-change) i Δ-energii.
* `vw_quality_trends` – serie czasowe SSIM/PSNR/ΔTests/ΔCC.

---

## 5) EGQL – zapytania domenowe

EGQL to **lekki DSL** nad EGDB (parsuje do SQL/JSONPath). Przykłady używane już w repo i dokumentacji:

* **Ścieżka bez błędu (SLA 5s):**
  `PATH TOPIC:run.start >> TOPIC:run.done WHERE window<=5000 AND NOT EXISTS TOPIC:run.error` 
* **Hotspoty (β > α):**
  `FIND tiles WHERE ΔH > ΔS ORDER BY ΔH DESC LIMIT 20` 
* **Naruszenia I2 w 24h (SQL-alias):**

  ````sql
  SELECT * FROM glx_events
  WHERE topic = 'validator.violation.I2'
    AND ts >= datetime('now','-24 hours');
  ``` :contentReference[oaicite:7]{index=7}

  ````

> **Konwencje:** `FIND`, `PATH`, `WHERE`, `WINDOW`, `LIMIT`, oraz aliasy `TOPIC:...`. Parser EGQL działa po stronie CLI/GUI i odwzorowuje się na widoki/tabele EGDB.

---

## 6) Metryki i Δ-dowody dla HUD

HUD korzysta z dwóch paneli:

* **Delta Inspector:** histogram `Δ-tokenów`, fingerprint, heatmapy Δ (AST↔Mozaika), przegląd **„co się naprawdę zmieniło”**.
* **Spec Monitor:** progi **α/β/Z**, drift (Page-Hinkley), tryb `freeze`, log przyczyn; aktualne wartości trzymamy m.in. w `.glx/spec_state.json` i logach commitów. 

Warstwy mozaiki i pseudometryki (SSIM, PSNR, dΦ) są opisane w materiale „Mozaikowe Drzewo AST…”, łącznie z warunkami spójności i kosztami planu – to *źródło prawdy* dla interpretacji wizualizacji.

---

## 7) Kolektor i ścieżka danych

1. **Hooki Git** publikują *delta-skrót* i artefakty audytowe (ZIP). 
2. **Core/Analysis/GUI** emitują zdarzenia na **BUS** (lokalny in-proc albo NATS/Kafka).
3. **Indexer** konsumuje BUS → zapisuje do **EGDB** zgodnie z rejestrem schematów. 
4. **HUD/CLI/CI** zadają zapytania **EGQL/SQL** i renderują dowody (heatmapy, PATH).
5. **Artefakty** (raporty, mozaiki, metryki) lądują w `.glx/*` i są linkowane w PR/GUI. 

---

## 8) Kalibracja, drift i retencja

* **Kalibracja progów (α/β/Z):** KWANTYLE + EWMA/MAD (aktualizowane strumieniowo), stan w `.glx/spec_state.json`.
* **Drift (Page-Hinkley):** wykrycie → **freeze thresholds** na okno N commitów, log przyczyn.
* **Retencja:** domyślnie 90 dni w SQLite, artefakty ZIP 180 dni; Postgres rekomendowany przy większym ruchu.
  Mechanizmy te są częścią „procedur operacyjnych (SOP)” w architekturze v5. 

---

## 9) Kontrakty jakości (I1–I4) w Observability

Wszystkie walidacje publikują **dowody** (np. PATH, fragmenty mozaiki, wartości progów):

* **I1** – typy/nośniki, brak wycieków poza ROI.
* **I2** – spójność warstw na granicach kafli (ε-bound).
* **I3** – lokalność/komutacja Δ (Φ(Δ_AST) ≈ Δ_MOZ).
* **I4** – monotoniczność celu (nie pogarszamy `𝒥`).
  Definicje i checklista są spisane w materiałach „Mozaikowe Drzewo AST…”.

---

## 10) Operacje (CLI/CI) – skrót

* `glx egql "<zapytanie>"  --db .glx/grammar/egdb.sqlite`
* `glx egdb migrate        --views .glx/grammar/views.sql`
* `glx report delta        --in analysis/logs --out .glx/delta_report.json`

> W **pipeline CI**: `lint → typecheck → tests → delta-tokens → invariants-check → build → artifacts` + publikacja artefaktów (heatmapy, JSONy) do PR/HUD; przykład workflow i hooków opisany w dokumentacji repo. 

---

## 11) Minimalne wymagania operacyjne

* **Idempotencja** kolektora (UPSERT po `(corr, topic, ts)`).
* **Walidacja schematów** wg rejestru (odrzuć payload niespójny).
* **Korelacja**: `corr` (run/commit) **wymagana** na `run.*`, `code.delta`, `invariants.*`.
* **Prywatność**: maskowanie ścieżek/sekretów w payloadach `security.alert`.
* **Degradacja**: gdy BUS/DB offline, zapisz bufor w `analysis/logs/commit_*.json` i wypchnij po powrocie (replay).

---

## 12) Załączniki i źródła

* `.glx/grammar/` – **views.sql**, **rules.yaml**, baza **egdb.sqlite** (layout repo). 
* `.glx/schemas/registry.json` – rejestr schematów. 
* `.glx/commit_snippet.txt`, `analysis/logs/commit_*.json` – ślady lokalne. 
* „Mozaikowe Drzewo AST – matematyka, formalizm i praktyka…” – definicje warstw, pseudometryk i kosztów planu (Φ/Ψ, I1–I4).
* `README.MD` – instalacja, hooki, EGQL przykłady i ścieżki `.glx/grammar`.

---

## 13) Checklista „prod-ready”

* [ ] EGDB zainicjalizowany (DDL → **views.sql** → seed). 
* [ ] BUS emituje: `run.*`, `code.delta`, `invariants.*`, `security.alert`, `heal.*`. 
* [ ] Walidacja payloadów (rejestr schematów aktywny). 
* [ ] HUD prezentuje **Delta Inspector** i **Spec Monitor** (progi/drift/freeze). 
* [ ] CI publikuje artefakty `.glx/*` do PR + komentarz z fingerprintem Δ. 

```

::contentReference[oaicite:26]{index=26}
```
