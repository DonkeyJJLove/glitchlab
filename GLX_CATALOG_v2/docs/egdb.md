# 03 — EGDB: Event Grammar DB (ERD + schematy)

## ERD
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

  glx_events }o--|| glx_topics : "topic"
```
%% Mermaid Styles
classDef tile fill:#0b7285,stroke:#083344,color:#fff;
classDef db fill:#4c6ef5,stroke:#233,color:#fff;
classDef guard fill:#e03131,stroke:#300,color:#fff;


## Schematy (JSON Schema)

- `glx_events`: `<!-- @auto:egdb.schema.events -->`
- `glx_deltas`: `<!-- @auto:egdb.schema.deltas -->`
- `glx_grammar_events`: `<!-- @auto:egdb.schema.grammar -->`
