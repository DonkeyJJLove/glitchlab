# 04 — Warstwa GUI (komponenty i przepływy)

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
%% Mermaid Styles
classDef tile fill:#0b7285,stroke:#083344,color:#fff;
classDef db fill:#4c6ef5,stroke:#233,color:#fff;
classDef guard fill:#e03131,stroke:#300,color:#fff;


**Opis komponentów:**  
`<!-- @auto:gui.components -->`
