# GlitchLab — **EGDB (Event & Graph Database) Spec** · `docs/21_egdb.md`

> Ten dokument **normatywnie** definiuje **EGDB** — trwałą bazę faktów i grafów GlitchLab. EGDB jest źródłem prawdy dla: historii zdarzeń (BUS), Δ-Algebry, mapowań **Φ/Ψ** (AST ↔ Mozaika), bramek **I1–I4**, SAST-Bridge i Healer.
> Dokument spójny z:
> – `docs/11_spec_glossary.md` (terminy),
> – `docs/12_invariants.md` (I1–I4),
> – `docs/13_delta_algebra.md` (tokeny, fingerprint),
> – `docs/20_bus.md` (envelopa zdarzeń).

---

## 0) Zakres i konwencje

* Słowa **MUST / SHOULD / MAY** są normatywne (RFC 2119).
* **Backend referencyjny:** PostgreSQL 15+ (**MUST**). **Tryb developerski:** SQLite 3.41+ (**SHOULD**) — subset typu/indeksów.
* Wszystkie ładunki `payload_json` **MUST** przechodzić walidację względem schematów w `/spec/schemas/`.

---

## 1) Rola EGDB

1. **Chronologia zdarzeń** (event-sourcing BUS).
2. **Grafy kodu** (AST/Import/Call) i **projekcje Φ/Ψ**.
3. **Strumienie Δ-cech** (tokeny, fingerprint, metryki, mozaiki).
4. **Wyniki inwariantów** (I1–I4) i **gates**.
5. **SAST-Bridge**: Normalized Findings + FixQueue.
6. **Healer**: kandydaci, weryfikacje, outcomes.
7. **Panele HUD**: widoki/materialized views.

---

## 2) Model logiczny (ER — skrót)

```
events ───< artifacts
  │
  ├──< runs ───< steps
  │
  ├──< invariants_results
  ├──< delta_tokens ──< delta_fingerprints
  ├──< mosaic_deltas
  ├──< spec_thresholds, spec_drift
  ├──< sast_findings ──< sast_fix_queue ──< sast_fix_verification
  └──< heal_candidates ──< heal_verification ──< heal_outcomes

(code graph)
modules ──< files ──< ast_nodes ──< ast_edges(call|contain|import)
               │                    │
               └──────< phi_links(ast_node_id ↔ mosaic_cell_id)
```

---

## 3) Schemat bazowy (DDL — PostgreSQL)

> Utrzymuj wersję schematu w `egdb_schema_version`.

```sql
-- 3.0 Meta
CREATE TABLE egdb_schema_version (
  version     integer PRIMARY KEY,
  applied_utc timestamptz NOT NULL DEFAULT now(),
  comment     text
);

-- 3.1 Zdarzenia (BUS snapshot)
CREATE TABLE events (
  id              text PRIMARY KEY,             -- ULID (BUS.id)
  topic           text NOT NULL,                -- 'core.step.finished@v1'
  ts_utc          timestamptz NOT NULL,
  source          text NOT NULL,
  correlation_id  text NOT NULL,
  causation_id    text,
  repo_sha        text NOT NULL,
  repo_branch     text,
  env_json        jsonb,
  payload_json    jsonb NOT NULL,               -- zgodny ze schema
  artifacts_json  jsonb,                        -- ["path", ...]
  idempotency_key text,
  trust           text DEFAULT 'high'           -- 'high'|'low'
);
CREATE INDEX ix_events_topic_ts ON events(topic, ts_utc);
CREATE INDEX ix_events_sha ON events(repo_sha);
CREATE INDEX ix_events_corr ON events(correlation_id);
CREATE INDEX ix_events_payload_gin ON events USING GIN(payload_json);

-- 3.2 Artefakty (opcjonalna materializacja)
CREATE TABLE artifacts (
  id       bigserial PRIMARY KEY,
  event_id text REFERENCES events(id) ON DELETE CASCADE,
  kind     text NOT NULL,                       -- 'json'|'png'|'parquet'|...
  path     text NOT NULL,
  size_b   bigint,
  sha256   text
);
CREATE INDEX ix_artifacts_event ON artifacts(event_id);

-- 3.3 Run/Steps (agregaty wykonania)
CREATE TABLE runs (
  id        text PRIMARY KEY,                   -- ULID corr_id runu
  sha       text NOT NULL,
  started   timestamptz NOT NULL,
  finished  timestamptz,
  status    text CHECK (status IN ('ok','error','partial')) DEFAULT 'ok'
);
CREATE TABLE steps (
  id       bigserial PRIMARY KEY,
  run_id   text REFERENCES runs(id) ON DELETE CASCADE,
  name     text NOT NULL,
  idx      int  NOT NULL,
  dt_ms    int  NOT NULL,
  deltas   jsonb
);
CREATE INDEX ix_steps_runid ON steps(run_id);

-- 3.4 Inwarianty (I1–I4)
CREATE TABLE invariants_results (
  id          bigserial PRIMARY KEY,
  sha         text NOT NULL,
  gate        text NOT NULL,                    -- 'I1'..'I4'
  score       double precision NOT NULL,
  threshold   double precision NOT NULL,
  action      text NOT NULL,                    -- 'block'|'review'|'warn'|'pass'
  details     jsonb,
  created_utc timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX ix_invariants_sha_gate ON invariants_results(sha, gate);

-- 3.5 Δ-Algebra
CREATE TABLE delta_tokens (
  id          bigserial PRIMARY KEY,
  sha         text NOT NULL,
  file        text NOT NULL,
  token       text NOT NULL,                    -- 'ADD_FN','MODIFY_SIG',...
  weight      double precision NOT NULL,        -- wkład do ΔZ
  meta        jsonb,                            -- np. arityΔ, line, symbol
  created_utc timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX ix_dt_sha_file ON delta_tokens(sha, file);
CREATE INDEX ix_dt_token ON delta_tokens(token);

CREATE TABLE delta_fingerprints (
  sha          text PRIMARY KEY,
  histogram    jsonb NOT NULL,                  -- {token:count,...}
  embedding    jsonb,                           -- opcjonalny vektor/bigrams
  hash         text NOT NULL,                   -- stabilny fingerprint
  score_deltaZ double precision NOT NULL,       -- zagregowany score
  created_utc  timestamptz NOT NULL DEFAULT now()
);

-- 3.6 Mozaika Δ i projekcje Φ/Ψ
CREATE TABLE mosaic_deltas (
  id          bigserial PRIMARY KEY,
  sha         text NOT NULL,
  grid        jsonb NOT NULL,                   -- {rows, cols, cell_size}
  stats       jsonb,                            -- {coverage, density, ...}
  mask_uri    text,                             -- .glx/mosaic_delta.png/.npy
  explain     jsonb,                            -- [{token, weight}, ...]
  created_utc timestamptz NOT NULL DEFAULT now()
);

-- 3.7 Spec żywy (progi i drift)
CREATE TABLE spec_thresholds (
  id          bigserial PRIMARY KEY,
  module      text NOT NULL,                    -- 'core','gui','analysis'|path
  alpha       double precision NOT NULL,
  beta        double precision NOT NULL,
  zeta        double precision NOT NULL,
  basis       text NOT NULL,                    -- 'ewma'|'quantile'
  state_uri   text,                             -- .glx/spec_state.json
  sha         text,
  created_utc timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE spec_drift (
  id          bigserial PRIMARY KEY,
  module      text NOT NULL,
  metric      text NOT NULL,
  detector    text NOT NULL,                    -- 'page-hinkley'|'adwin'
  freeze      boolean NOT NULL DEFAULT true,
  details     jsonb,
  created_utc timestamptz NOT NULL DEFAULT now()
);

-- 3.8 SAST-Bridge
CREATE TABLE sast_findings (
  id            text PRIMARY KEY,               -- 'NF-...'
  sha           text NOT NULL,
  tool          text NOT NULL,
  rule_id       text NOT NULL,
  cwe           text,
  severity      text CHECK (severity IN ('LOW','MEDIUM','HIGH','CRITICAL')) NOT NULL,
  confidence    double precision NOT NULL,
  category      text,
  message       text,
  location      jsonb NOT NULL,                 -- {file,line,col,endLine}
  ast_ref       text,
  mosaic_cell   text,
  delta_bind    text CHECK (delta_bind IN ('added','modified','unchanged')),
  evidence      jsonb,
  suppression   jsonb,
  fingerprint   text NOT NULL UNIQUE,
  created_utc   timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX ix_sast_sha_sev ON sast_findings(sha, severity);

CREATE TABLE sast_fix_queue (
  id             bigserial PRIMARY KEY,
  nf_id          text REFERENCES sast_findings(id) ON DELETE CASCADE,
  pattern        text,
  fix_hint       text,
  patch_spec     jsonb,
  tests_required jsonb,
  constraints    jsonb,                         -- {"I1..I4":"must-hold",...}
  review_gate    text CHECK (review_gate IN ('human','auto')) DEFAULT 'human',
  status         text CHECK (status IN ('queued','proposed','accepted','rejected')) DEFAULT 'queued',
  created_utc    timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE sast_fix_verification (
  id           bigserial PRIMARY KEY,
  fix_id       bigint REFERENCES sast_fix_queue(id) ON DELETE CASCADE,
  sandbox_ok   boolean,
  ci_ok        boolean,
  invariants   jsonb,
  notes        text,
  ended_utc    timestamptz
);

-- 3.9 Healer
CREATE TABLE heal_candidates (
  id            text PRIMARY KEY,               -- ULID
  sha           text NOT NULL,
  branch        text,
  kind          text CHECK (kind IN ('code_patch','test_rewrite','config_fix')) NOT NULL,
  scope_json    jsonb,                          -- files/lines/modules
  rationale_json jsonb,
  confidence    double precision,
  deltas_json   jsonb,                          -- ΔS/ΔH/ΔZ summary
  created_utc   timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE heal_verification (
  id            text PRIMARY KEY,               -- ULID
  candidate_id  text REFERENCES heal_candidates(id) ON DELETE CASCADE,
  sandbox_ok    boolean,
  ci_ok         boolean,
  mutation_score double precision,
  canary_ok     boolean,
  notes         text,
  ended_ts      timestamptz
);

CREATE TABLE heal_outcomes (
  id            text PRIMARY KEY,               -- ULID
  candidate_id  text REFERENCES heal_candidates(id) ON DELETE CASCADE,
  result        text CHECK (result IN ('applied','rejected','rolled_back')) NOT NULL,
  reason        text,
  pr_url        text
);

-- 3.10 Grafy kodu (AST/Import/Call)
CREATE TABLE modules (
  id       bigserial PRIMARY KEY,
  name     text UNIQUE NOT NULL,                -- 'glitchlab.core.pipeline'
  path     text NOT NULL
);

CREATE TABLE files (
  id        bigserial PRIMARY KEY,
  module_id bigint REFERENCES modules(id) ON DELETE CASCADE,
  path      text UNIQUE NOT NULL,
  sha256    text
);

CREATE TABLE ast_nodes (
  id        bigserial PRIMARY KEY,
  file_id   bigint REFERENCES files(id) ON DELETE CASCADE,
  uid       text UNIQUE NOT NULL,               -- 'AST::<node-uid>'
  kind      text NOT NULL,                      -- 'FunctionDef'|'ClassDef'|...
  name      text,
  line      int,
  col       int,
  meta      jsonb
);

CREATE TABLE ast_edges (
  id        bigserial PRIMARY KEY,
  src_id    bigint REFERENCES ast_nodes(id) ON DELETE CASCADE,
  dst_id    bigint REFERENCES ast_nodes(id) ON DELETE CASCADE,
  etype     text CHECK (etype IN ('contain','call','import')) NOT NULL,
  meta      jsonb
);
CREATE INDEX ix_ast_edges_src ON ast_edges(src_id);
CREATE INDEX ix_ast_edges_dst ON ast_edges(dst_id);

CREATE TABLE phi_links (
  id            bigserial PRIMARY KEY,
  ast_node_id   bigint REFERENCES ast_nodes(id) ON DELETE CASCADE,
  mosaic_cell   text NOT NULL,                  -- 'MOZ::<cell-id>'
  weight        double precision NOT NULL       -- siła projekcji Φ
);
```

> **SQLite wariant:** użyj `TEXT`/`REAL` zamiast `jsonb` (JSON przechowywany jako TEXT), pomiń GIN, zachowaj klucze główne i indeksy podstawowe.

---

## 4) Indeksowanie i partycjonowanie

* **PostgreSQL MUST:**

  * GIN na `events.payload_json`.
  * BTREE `(topic, ts_utc)`, `(sha, gate)` itd.
  * **Partycje** dzienne na `events.ts_utc` i tygodniowe na `delta_tokens.created_utc` (Range partitioning).
* **SQLite SHOULD:** indeksy BTREE na kolumnach używanych w HUD/CI.

---

## 5) EGQL — język zapytań domenowych (minimalne jądro)

**Cel:** prosty DSL tłumaczony na SQL dla HUD/CI/analityki.

### 5.1 Składnia (EBNF skrót)

```
query      := select WS from WS where? WS group? WS order? WS window?
select     := "SELECT" ( "events" | "tokens" | "graph" | "invariants" )
from       := "FROM" ( "sha(" SHA ")" | "branch(" NAME ")" )
where      := "WHERE" cond { ("AND" | "OR") cond }
cond       := key OP value | "topic~" REGEX | "token=" NAME | "gate=" NAME
group      := "GROUP BY" key { "," key }
order      := "ORDER BY" key ("ASC"|"DESC")?
window     := "SINCE" ISO8601 | "LAST" INT ( "m" | "h" | "d" )
```

### 5.2 Przykłady

* **Ścieżka przyczyn (events) od commitu:**

```
SELECT events FROM sha(8f1c…) WHERE topic~ "core\..*" SINCE 2025-10-10T00:00Z
```

* **Tokeny Δ dla pliku w ostatniej godzinie:**

```
SELECT tokens FROM branch(main) WHERE file="core/pipeline.py" LAST 1h
```

* **Naruszenia I3 dla modułu `core`:**

```
SELECT invariants FROM branch(main) WHERE gate=I3 AND payload.module="core" LAST 7d
```

* **Podgraf wywołań od funkcji `run`:**

```
SELECT graph FROM sha(8f1c…) WHERE start="glitchlab.core.pipeline:run" depth<=3
```

> Mapowanie EGQL→SQL realizuje `glitchlab.egql.compiler` (adapter dla PG/SQLite).

---

## 6) Widoki i materializacje (HUD/CI)

```sql
-- Ostatnie uruchomienie na sha
CREATE MATERIALIZED VIEW vw_last_run_by_sha AS
SELECT sha, max(finished) AS last_finished, count(*) AS runs
FROM runs GROUP BY sha;

-- Heatmapa naruszeń bramek
CREATE MATERIALIZED VIEW vw_invariants_heatmap AS
SELECT gate, date_trunc('day', created_utc) AS day, count(*) AS cnt
FROM invariants_results GROUP BY gate, day;

-- Backlog SAST (otwarte)
CREATE VIEW vw_sast_backlog AS
SELECT f.sha, f.severity, count(*) AS cnt
FROM sast_findings f
LEFT JOIN sast_fix_queue q ON q.nf_id = f.id
WHERE q.id IS NULL
GROUP BY f.sha, f.severity;

-- Skuteczność Healer
CREATE VIEW vw_heal_success_by_kind AS
SELECT kind, avg( (o.result='applied')::int ) AS success_rate, count(*) AS n
FROM heal_candidates c
JOIN heal_outcomes o ON o.candidate_id = c.id
GROUP BY kind;

-- Gęstość Δ tokenów per moduł
CREATE VIEW vw_delta_density_by_module AS
SELECT m.name AS module, count(*) AS tokens
FROM delta_tokens dt
JOIN files f ON f.path = dt.file
JOIN modules m ON m.id = f.module_id
GROUP BY m.name;
```

> **REFRESH** materializowanych widoków po każdej serii `post-commit`/CI lub co 5 minut.

---

## 7) Ingest z BUS → EGDB (idempotencja)

**Zasada:** każde zdarzenie **MUST** upsertować się po `id` (ULID) i/lub `idempotency_key`.

Pseudokod:

```python
def persist(evt):
    with tx():
        upsert_events(evt)          # ON CONFLICT (id) DO UPDATE minimalne
        for a in evt.artifacts:     # opcjonalnie
            upsert_artifact(evt.id, a)
        route(evt)                  # demultiplekser: do runs/steps/delta/... zgodnie z topic
```

**Błędy walidacji** → zapis do `bus.dlq@v1` + tabela `events_dlq` (opcjonalnie).

---

## 8) Życie danych, retencja, archiwa

* **Events:** partycje dzienne, retencja 90 dni w ONLINE (**SHOULD**), starsze → **archiwum Parquet** (`/archives/egdb/events_YYYYMM.parquet`) (**MUST**).
* **Graf kodu:** bez retencji; wersjonuj po `sha` lub „aktywny stan” (ostatnie HEAD per plik).
* **SAST/Healer:** min. 180 dni (audyt).
* **Kopia zapasowa:** `pg_dump -Fc` codziennie; test odtwarzania tygodniowo.
* **PII/sekrety:** payload **MUST NOT** przenosić sekretów (por. `docs/20_bus.md §8`).

---

## 9) Migracje i zgodność

* Narzędzie migracji: **Alembic** (**MUST**), katalog `db/migrations`.
* Tabela `egdb_schema_version` aktualizowana atomowo w transakcji.
* Migracje **MUST** być odwracalne (downgrade), chyba że uzasadniono inaczej.
* Zmiany niekompatybilne → nowy `@vN+1` w tematach i nowe widoki.

---

## 10) Wydajność i SLO

* **SLO ingest:** p95 < 200 ms zapis `events` (poza I/O artefaktów).
* **SLO zapytań HUD:** p95 < 400 ms dla widoków *nie-materializowanych* (filtry po `sha/day`).
* **Wskazówki:**

  * GIN na JSON tylko na tabeli `events`.
  * Agregaty tokenów: materializacja + TTL 1 min.
  * Używać `COPY`/batch dla masowych insertów tokenów.

---

## 11) Φ/Ψ — mapowania operacyjne

* **`phi_links`** wiąże `ast_nodes` ↔ `mosaic_cell` z **`weight`** (siła projekcji).
* **Test I3 (komutacja z Δ)** — przykład zapytania:

```sql
-- różnica między Φ(Δ_AST) i Δ_MOZ (heurystycznie):
WITH ast_delta AS (
  SELECT n.id, sum(dt.weight) AS w
  FROM delta_tokens dt
  JOIN ast_nodes n ON n.file_id = (SELECT id FROM files WHERE path = dt.file)
  WHERE dt.sha = $1 GROUP BY n.id
),
phi_delta AS (
  SELECT p.mosaic_cell, sum(a.w * p.weight) AS w_moz
  FROM ast_delta a JOIN phi_links p ON p.ast_node_id = a.id
  GROUP BY p.mosaic_cell
)
SELECT md.id, abs(md.stats->>'density')::float - coalesce(p.w_moz,0) AS diff
FROM mosaic_deltas md
LEFT JOIN phi_delta p ON p.mosaic_cell = (md.grid->>'cell_id')
WHERE md.sha = $1;
```

> Wynik agreguj do wskaźnika `|| Φ(Δ_AST) – Δ_MOZ ||` (por. `docs/12_invariants.md`, I3).

---

## 12) Δ-Algebra w EGDB (przykładowe zapytania)

* **Fingerprint i „gorące” moduły:**

```sql
SELECT m.name, sum(dt.weight) AS z
FROM delta_tokens dt
JOIN files f ON f.path = dt.file
JOIN modules m ON m.id = f.module_id
WHERE dt.sha = $1
GROUP BY m.name ORDER BY z DESC LIMIT 10;
```

* **Histogram tokenów do HUD:**

```sql
SELECT token, count(*) AS cnt
FROM delta_tokens WHERE sha = $1
GROUP BY token ORDER BY cnt DESC;
```

---

## 13) SAST-Bridge w EGDB

* **Unikalność findingu:** `fingerprint UNIQUE` (deduplikacja między skanami).
* **Backlog napraw:** widok `vw_sast_backlog` (patrz §6).
* **Śledzenie weryfikacji:** `sast_fix_verification` z `sandbox_ok`, `ci_ok`, `invariants`.
* **Gates:** naruszenia **MUST** tworzyć rekord w `invariants_results` (źródło: `sast.findings.ready@v1` + polityka).

---

## 14) Healer — pełny łańcuch

* **Kandydat** → `heal_candidates` (rationale, scope, Δ).
* **Weryfikacja** → `heal_verification` (sandbox/CI/mutation/canary).
* **Outcome** → `heal_outcomes` (applied/rejected/rolled_back).
* **Metryki sukcesu** → `vw_heal_success_by_kind`.

---

## 15) End-points (SQL gotowce dla HUD/CI)

* **Panel Δ Inspector**
  `SELECT * FROM delta_fingerprints WHERE sha = $sha;`
  `SELECT * FROM vw_delta_density_by_module WHERE module LIKE $prefix;`

* **Panel Spec Monitor**
  `SELECT * FROM spec_thresholds ORDER BY created_utc DESC LIMIT 100;`
  `SELECT * FROM spec_drift WHERE freeze = true ORDER BY created_utc DESC;`

* **Panel Security**
  `SELECT severity, count(*) FROM vw_sast_backlog GROUP BY severity;`

---

## 16) Testy, dane przykładowe

Minimalne fixtury (PG):

```sql
INSERT INTO runs(id, sha, started, status) VALUES ('01J…RUN', '8f1c…', now(), 'ok');
INSERT INTO steps(run_id, name, idx, dt_ms) VALUES ('01J…RUN','pipeline.build_ctx',0,123);

INSERT INTO delta_tokens(sha,file,token,weight,meta)
VALUES ('8f1c…','core/pipeline.py','MODIFY_SIG',1.0,'{"arity_delta":1,"line":123}');

INSERT INTO delta_fingerprints(sha,histogram,embedding,hash,score_deltaZ)
VALUES ('8f1c…','{"MODIFY_SIG":1}',NULL,'fp:abcd1234',1.0);

INSERT INTO invariants_results(sha,gate,score,threshold,action)
VALUES ('8f1c…','I2',0.87,0.80,'review');
```

Testy **MUST** obejmować: idempotencję ingestu, integralność FK, wydajność zapytań HUD.

---

## 17) Backup/Restore

* **Backup:** `pg_dump -Fc glitchlab_egdb > backups/egdb_$(date +%F).dump` (MUST).
* **Restore:** `pg_restore -d glitchlab_egdb backups/egdb_<date>.dump` + `ANALYZE`.
* **Parquet archive:** miesięczne `COPY (SELECT …) TO PROGRAM 'parquet-writer …'` lub ETL batch.

---

## 18) Checklist wdrożeniowy

* [ ] Uruchom migracje (Alembic) → `egdb_schema_version`.
* [ ] Włącz partycjonowanie `events`, `delta_tokens`.
* [ ] Skonfiguruj ingest BUS→EGDB z idempotencją.
* [ ] Zbuduj widoki (§6) + plan `REFRESH`.
* [ ] Włącz politykę retencji + archiwum Parquet.
* [ ] Dodaj testy E2E ingest→HUD (fixtury §16).
* [ ] Monitoruj SLO (§10) i metryki DB (lag, p95, bloat).

---

## 19) Status pliku

✅ **Final (Spec — EGDB)**
Zmiany w schemacie **MUST** przechodzić przez migracje i aktualizować `egdb_schema_version`.
