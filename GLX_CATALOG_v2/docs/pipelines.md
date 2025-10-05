# 05 — Pipelines: Filters / Analysis / Mosaic (Φ/Ψ)

## Filters
```mermaid
flowchart LR
  IN[Input Image] --> F1[filter: rgb_offset]
  F1 --> F2[filter: amp_mask_mul]
  F2 --> F3[filter: scanlines]
  F3 --> OUT[Output]
```
%% Mermaid Styles
classDef tile fill:#0b7285,stroke:#083344,color:#fff;
classDef db fill:#4c6ef5,stroke:#233,color:#fff;
classDef guard fill:#e03131,stroke:#300,color:#fff;


## Sprzęg AST ⇄ Mozaika (Φ/Ψ)
```mermaid
graph LR
  ASTN[AST Nodes] -->|map Φ| MZ[Mosaic Tiles]
  MZ -->|feedback Ψ| ASTN
```
**Metryki:** `<!-- @auto:mosaic.metrics -->`
