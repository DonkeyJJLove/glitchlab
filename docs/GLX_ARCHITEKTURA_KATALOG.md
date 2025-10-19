# GLX :: Katalog architektury

> Ten dokument to **hipertekstowy katalog architektury GlitchLab** przygotowany jako **szablon**.  
> **Nie wykonuje** analizy kodu ani załączonych ZIP-ów. Wszystkie sekcje mają **miejsca na automatyczne wypełnienie** (Δ) przez Twoje skrypty GLX.

---

## Spis treści

1. [Mapa systemu (HLD)](#mapa-systemu-hld)
2. [Repozytorium (Repo GLX)](#repozytorium-repo-glx)
   - [pre-commit.py](#pre-commitpy)
   - [pre-diff.py](#pre-diffpy)
   - [.glx/commit_analysis.json](#glxcommitanalysisjson)
   - [post-commit.py](#post-commitpy)
   - [AUDIT_*.zip](#audit_zip)
3. [GLX BUS — tiles, guards, HUD](#glx-bus--tiles-guards-hud)
   - [git-analytics tile](#git-analytics-tile)
   - [code-ast service](#code-ast-service)
   - [refactor-planner](#refactor-planner)
   - [validators I1–I4](#validators-i1i4)
   - [HUD/Reports](#hudreports)
4. [EGDB — Event Grammar DB (ERD + Schemata)](#egdb--event-grammar-db-erd--schemata)
5. [Warstwa GUI (komponenty i przepływy)](#warstwa-gui-komponenty-i-przepływy)
6. [Pipelines: Filters / Analysis / Mosaic (Φ/Ψ)](#pipelines-filters--analysis--mosaic-φψ)
7. [CI/Ops i cykl commitu — sekwencja](#ciops-i-cykl-commitu--sekwencja)
8. [Kontrakty danych (JSON Schema) — szablony](#kontrakty-danych-json-schema--szablony)
9. [Reguły walidacji (I1–I4) — szablon](#reguły-walidacji-i1i4--szablon)
10. [Obserwowalność i metryki (HUD)](#obserwowalność-i-metryki-hud)
11. [Załączniki i tagi automaty](#załączniki-i-tagi-automaty)

---

## Mapa systemu (HLD)

Poniższa mapa jest **kanoniczną** wersją Twojego grafu — rozszerzoną o węzły pomocnicze.  
> Źródło: ręcznie utrzymywany schemat (bez analizy kodu).

```mermaid
flowchart LR
  subgraph Repo [Repo GLX]
    HookPre[pre-commit.py] --> PreDiff[pre-diff.py]
    PreDiff --> GLXglx[.glx/commit_analysis.json]
    HookPost[post-commit.py] --> AuditZIP[AUDIT_*.zip]
  end

  subgraph Bus [GLX BUS]
    GA[git-analytics tile]:::tile
    AST[code-ast service]:::tile
    REF[refactor-planner]:::tile
    VAL[validators I1–I4]:::guard
    HUD[HUD/Reports]:::tile
  end

  subgraph EGDB[EGDB: Event Grammar DB]
    EVT[(glx_events)]:::db
    CFG[(glx_config)]:::db
    TOP[(glx_topics)]:::db
    GDelta[(glx_deltas)]:::db
    GRM[(glx_grammar_events)]:::db
  end

  Repo -->|HEAD, Δ| GA
  GA -->|git.delta.ready| HUD
  AST -->|code.ast.built| HUD
  REF -->|refactor.plan.ready| HUD
  VAL -->|fail-closed| HUD
  HUD -->|publish| EVT

  classDef tile fill:#0b7285,stroke:#083344,color:#fff;
  classDef db fill:#4c6ef5,stroke:#233,color:#fff;
  classDef guard fill:#e03131,stroke:#300,color:#fff;
```
**Dowiązania (nawigacja):**  
[Repo GLX](#repozytorium-repo-glx) • [GLX BUS](#glx-bus--tiles-guards-hud) • [EGDB](#egdb--event-grammar-db-erd--schemata)

---

## Repozytorium (Repo GLX)

### `pre-commit.py`
**Rola:** Hak inicjujący proces przygotowania Δ/HEAD.  
**Wejścia:** staged changes.  
**Wyjścia:** wywołanie `pre-diff.py`.  
**Dane:** Δ (A/M/D, loc_add/del), branch, parent.  
**Powiązania:** → [`pre-diff.py`](#pre-diffpy)

**Miejsca na automatyczne wypełnienie (Δ):**
- Δ plików: `<!-- @auto:repo.precommit.delta -->`
- Statystyki LOC: `<!-- @auto:repo.precommit.loc -->`

---

### `pre-diff.py`
**Rola:** Buduje artefakt różnic i metadanych commitu.  
**Wyjścia:** [`/.glx/commit_analysis.json`](#glxcommitanalysisjson).

**Kontrakt (szablon):**
```json
{
  "commit": { "hash": "Δ", "parent": "Δ", "author": "Δ", "ts": "Δ" },
  "delta": { "files": "Δ", "summary": "Δ" },
  "head": { "branch": "Δ" }
}
```

---

### `.glx/commit_analysis.json`
**Rola:** Artefakt repo z danymi Δ/HEAD.  
**Konsumenci:** [`git-analytics tile`](#git-analytics-tile).

**Podgląd (opcjonalny):**
```
<!-- @auto:repo.glx.commit_analysis.preview -->
```

---

### `post-commit.py`
**Rola:** Pakuje audyt w `AUDIT_*.zip`.  
**Wejścia:** HEAD po commit.  
**Wyjścia:** [`AUDIT_*.zip`](#audit_zip).

---

### `AUDIT_*.zip`
**Rola:** Artefakt archiwalny; raporty, logi, zrzuty.  
**Lista zawartości (opcjonalna):**
```
<!-- @auto:repo.audit_zip.contents -->
```

---

## GLX BUS — tiles, guards, HUD

### git-analytics tile
**Rola:** Parsuje `.glx/commit_analysis.json`; emituje `git.delta.ready`.  
**Wejścia:** Δ/HEAD.  
**Wyjścia:** zdarzenie bus.  
**Kontrakt zdarzenia (szablon):**
```json
{
  "type": "git.delta.ready",
  "commit": "Δ",
  "delta": { "files": "Δ" },
  "summary": { "add": "Δ", "del": "Δ" }
}
```

---

### code-ast service
**Rola:** Buduje AST (zakres wg Δ lub całość).  
**Wyjścia:** `code.ast.built`.  
**Kontrakt (szablon):**
```json
{
  "type": "code.ast.built",
  "commit": "Δ",
  "scope": ["Δ"],
  "metrics": { "nodes": "Δ", "functions": "Δ", "classes": "Δ" }
}
```

---

### refactor-planner
**Rola:** Generuje plan refaktorów (na bazie Δ/AST).  
**Wyjścia:** `refactor.plan.ready`.  
**Kontrakt (szablon):**
```json
{
  "type": "refactor.plan.ready",
  "commit": "Δ",
  "items": [{"path":"Δ","action":"Δ","rationale":"Δ"}]
}
```

---

### validators I1–I4
**Rola:** Bramki jakości; publikują `fail-closed` przy naruszeniu.  
**Kontrakt (szablon):**
```json
{
  "type": "fail-closed",
  "commit": "Δ",
  "stage": "I1|I2|I3|I4",
  "errors": [{"code":"Δ","msg":"Δ","path":"Δ"}]
}
```

---

### HUD/Reports
**Rola:** Konsolidacja i publikacja do `glx_events`.  
**Operacja:** `publish` (patrz [EGDB](#egdb--event-grammar-db-erd--schemata)).

---

## EGDB — Event Grammar DB (ERD + Schemata)

**ERD (szablon):**
```mermaid
erDiagram
  glx_events {
    bigint id PK
    timestamp ts
    varchar source
    varchar type
    varchar commit
    json payload
    varchar topic
    varchar severity
  }
  glx_config {
    varchar key PK
    json value
    varchar scope
    timestamp updated_at
  }
  glx_topics {
    varchar topic PK
    varchar producer
    varchar schema_ref
    varchar retention
  }
  glx_deltas {
    varchar commit
    varchar path
    varchar status
    int loc_add
    int loc_del
    varchar hash_before
    varchar hash_after
  }
  glx_grammar_events {
    varchar type PK
    varchar version
    json schema
    varchar producer
    varchar consumers
  }

  glx_events ||--o{ glx_topics : "topic"
```

**Miejsca na schematy (JSON Schema):**
- `glx_events`: `<!-- @auto:egdb.schema.events -->`
- `glx_deltas`: `<!-- @auto:egdb.schema.deltas -->`
- `glx_grammar_events`: `<!-- @auto:egdb.schema.grammar -->`

---

## Warstwa GUI (komponenty i przepływy)

> Nazwy przykładowe zgodne z konwencją GLX (szablon — bez analizy kodu).

```mermaid
flowchart TD
  subgraph GUI[GUI Layer]
    LP[LayerPanel] --> LM[LayerManager]
    LM --> CC[CanvasContainer]
    CC --> CMP[Compositor]
    HUDG[HUDView] -->|events| HUDS[HUDStore]
  end

  subgraph Services
    EB[EventBus]
    PR[PipelineRunner]
  end

  EB <-->|pub/sub| HUDS
  PR --> CMP
  LM --> PR
  style GUI fill:#0b7285,stroke:#083344,color:#fff
```

**Miejsca na opis komponentów GUI:**
- `<!-- @auto:gui.components -->`

---

## Pipelines: Filters / Analysis / Mosaic (Φ/Ψ)

### Przepływ filtrów (szablon)
```mermaid
flowchart LR
  IN[Input Image] --> F1[filter: rgb_offset]
  F1 --> F2[filter: amp_mask_mul]
  F2 --> F3[filter: scanlines]
  F3 --> OUT[Output]
```

### Sprzęg AST ⇄ Mozaika (Φ/Ψ) — koncepcja
```mermaid
graph LR
  ASTN[AST Nodes] -->|map Φ| MZ[Mosaic Tiles]
  MZ -->|feedback Ψ| ASTN
```
**Miejsca na metryki:** `<!-- @auto:mosaic.metrics -->`

---

## CI/Ops i cykl commitu — sekwencja

```mermaid
sequenceDiagram
  participant Dev as Developer
  participant Git as Repo (hooks)
  participant GA as git-analytics
  participant AST as code-ast
  participant REF as refactor-planner
  participant VAL as validators I1–I4
  participant HUD as HUD/Reports
  participant DB as EGDB

  Dev->>Git: git commit
  Git->>Git: pre-commit.py
  Git->>Git: pre-diff.py → .glx/commit_analysis.json
  Git-->>Dev: commit ok
  Git->>Git: post-commit.py → AUDIT_*.zip

  GA->>HUD: git.delta.ready
  AST->>HUD: code.ast.built
  REF->>HUD: refactor.plan.ready
  VAL->>HUD: fail-closed?
  HUD->>DB: publish(events)
```

---

## Kontrakty danych (JSON Schema) — szablony

> Wklej/utrzymuj poniższe schematy jako **źródło prawdy**.  
> Dodaj wersjonowanie w `glx_grammar_events`.

- `git.delta.ready` — `<!-- @auto:schema.git.delta.ready -->`  
- `code.ast.built` — `<!-- @auto:schema.code.ast.built -->`  
- `refactor.plan.ready` — `<!-- @auto:schema.refactor.plan.ready -->`  
- `fail-closed` — `<!-- @auto:schema.fail.closed -->`  
- `publish` — `<!-- @auto:schema.publish -->`  

---

## Reguły walidacji (I1–I4) — szablon

- **I1 — Spójność Δ:** `<!-- @auto:rules.I1 -->`
- **I2 — Syntaktyka/kompilowalność:** `<!-- @auto:rules.I2 -->`
- **I3 — Standardy i konwencje:** `<!-- @auto:rules.I3 -->`
- **I4 — Kontrakty interfejsów:** `<!-- @auto:rules.I4 -->`

> Tryb **fail-closed**: publikuj `fail-closed` przy pierwszym naruszeniu; zatrzymaj przepływ do czasu korekty.

---

## Obserwowalność i metryki (HUD)

Tabela metryk do uzupełnienia przez automaty:

| Warstwa | Metryka | Opis | Miejsce na wypełnienie |
|---|---|---|---|
| Repo | Δ files (A/M/D) | Liczba plików w delcie | <!-- @auto:obs.repo.delta --> |
| Repo | LOC ± | Dodane/usunięte linie | <!-- @auto:obs.repo.loc --> |
| AST | Nodes / Func / Class | Rozmiary AST | <!-- @auto:obs.ast.sizes --> |
| REF | Plan items | Liczba propozycji | <!-- @auto:obs.ref.count --> |
| VAL | I1–I4 pass/fail | Stan bramek | <!-- @auto:obs.val.state --> |
| HUD | T_pub | Czas do publikacji | <!-- @auto:obs.hud.tpub --> |

---

## Załączniki i tagi automaty

- **Wtyczki/tagi dla Twoich skryptów GLX (przykładowe):**
  - `<!-- @auto:... -->` — miejsce wstrzyknięcia danych z audytu/ZIP/analiz.
  - `<!-- @auto:diagram:XYZ -->` — renderuj dodatkowe grafy Mermaid.
  - `<!-- @auto:contract:EVENT -->` — wstaw JSON Schema zdarzenia.

**Instrukcja integracji (pseudo):**
```
glx_doc_inject --file docs/GLX_ARCHITECTURE.md \
  --source /mnt/data/AUDIT_YYYYMMDD.zip \
  --tag @auto:repo.glx.commit_analysis.preview \
  --value "$(jq '.delta | .files' .glx/commit_analysis.json)"
```

---

### Nota końcowa
- Ten katalog jest **template-first**: bez ryzyka pomyłek wynikających z domysłów.  
- Aby go „ożywić”, podłącz swoje encje GLX (ZIP/HASH/MAIL) i wypełnij sekcje oznaczone jako **Δ** lub `@auto:`.
