### A) Architektura GUI – widok makro (C4-ish)

```mermaid
flowchart LR
  %% NODES
  subgraph CORE["Core / Root"]
    APP["app.py<br/>AppShell"]:::core
    EB["event_bus.py<br/>EventBus"]:::bus
    LOADER["panel_loader.py"]:::core
    STATE["state.py"]:::core
  end

  subgraph GUI["GUI (Views + Widgets + Panels)"]
    MENU["views/menu.py"]:::ui
    NB["views/notebook.py"]:::ui
    VIEWPORT["views/viewport.py"]:::ui
    LAYERP["views/layer_panel.py"]:::ui
    HUDV["views/hud.py"]:::ui

    CC["widgets/canvas_container.py"]:::uiw
    IC["widgets/image_canvas.py"]:::uiw
    MV["widgets/mosaic_view.py"]:::uiw
    PF["widgets/param_form.py"]:::uiw
    PPV["widgets/pipeline_preview.py"]:::uiw

    PANELS["panels — plugins"]:::panel
  end

  subgraph SRV["Services"]
    PR["services/pipeline_runner.py"]:::svc
    CMP["services/compositor.py"]:::svc
    LM["services/layer_manager.py"]:::svc
    FILES["services/files.py"]:::svc
    PRESET["services/presets.py"]:::svc
    MASKS["services/masks.py"]:::svc
    IH["services/image_history.py"]:::svc
    LAYOUT["services/layout.py"]:::svc
  end

  subgraph STORES["Stores / DB"]
    HUDS["widgets/hud.py<br/>HUDStore"]:::store
    EGDB[(EGDB<br/>runtime_events<br/>grammar_events)]:::db
  end

  %% FLOWS
  APP --> MENU
  APP --> NB
  APP --> VIEWPORT
  APP -->|discover| LOADER
  APP --> STATE

  MENU --> EB
  NB --> PANELS
  LAYERP --> LM
  HUDV --> HUDS
  VIEWPORT --> CC
  CC --> IC
  CC --> MV
  PF --> EB
  PPV --> EB

  EB <-->|pub/sub| HUDS
  EB <-->|pub/sub| PR
  EB <-->|pub/sub| LM
  EB --> EGDB

  LM --> PR
  PR --> CMP
  CMP --> VIEWPORT
  FILES --> PR
  PRESET --> PR
  MASKS --> PR
  IH --> PR
  LAYOUT --> VIEWPORT

  %% STYLES
  classDef core fill:#0b7285,stroke:#083344,color:#fff;
  classDef bus  fill:#1c7ed6,stroke:#123a62,color:#fff;
  classDef ui   fill:#3fba8a,stroke:#0a3,color:#062;
  classDef uiw  fill:#2f9e44,stroke:#0a3,color:#fff;
  classDef panel fill:#845ef7,stroke:#423,color:#fff;
  classDef svc  fill:#f08c00,stroke:#742,color:#000;
  classDef store fill:#495057,stroke:#222,color:#fff;
  classDef db   fill:#4c6ef5,stroke:#233,color:#fff;
```

---

### B) GUI – wewnętrzne przepływy (Views/Widgets/Panels)

```mermaid
flowchart TD
  subgraph VIEWS["Views"]
    MENU[menu.py]:::ui
    LEFTTB[left_toolbar.py]:::ui
    NB[notebook.py]:::ui
    TABG[tab_general.py]:::ui
    TABF[tab_filter.py]:::ui
    TABP[tab_preset.py]:::ui
    LAYERP[layer_panel.py]:::ui
    HUDV[hud.py]:::ui
    BOTTOM[bottom_panel.py]:::ui
    STATUS[statusbar.py]:::ui
  end

  subgraph WIDGETS["Widgets"]
    CC[canvas_container.py]:::uiw
    IC[image_canvas.py]:::uiw
    LC[layer_canvas.py]:::uiw
    IL[image_layers.py]:::uiw
    MV[mosaic_view.py]:::uiw
    OR[overlay_renderer.py]:::uiw
    PF[param_form.py]:::uiw
    PPV[pipeline_preview.py]:::uiw
    TOOLS[tools/*]:::uiw
  end

  subgraph PANELS["Panels (pluginy)"]
    P_ID[panel_default_identity.py]:::panel
    P_RGBO[panel_rgb_offset.py]:::panel
    P_MOSH[panel_block_mosh.py]:::panel
    P_DEPTH[panel_depth_displace.py]:::panel
    P_SPECT[panel_spectral_shaper.py]:::panel
    P_PSORT[panel_pixel_sort_adaptive.py]:::panel
  end

  MENU --> NB
  LEFTTB --> NB
  NB --> TABG
  NB --> TABF
  NB --> TABP
  NB -->|mount| PANELS
  LAYERP --> CC
  HUDV --> BOTTOM
  BOTTOM --> STATUS

  PANELS --> PF
  PF --> PPV
  PF --> CC
  CC --> IC
  CC --> LC
  LC --> IL
  CC --> MV
  OR --> CC
  TOOLS --> CC

  classDef ui fill:#3fba8a,stroke:#0a3,color:#062;
  classDef uiw fill:#2f9e44,stroke:#0a3,color:#fff;
  classDef panel fill:#845ef7,stroke:#423,color:#fff;
```

---

### C) Services – zależności i punkty integracji

```mermaid
flowchart LR
  EB[event_bus.py<br/>EventBus]:::bus
  PR[pipeline_runner.py]:::svc
  CMP[compositor.py]:::svc
  LM[layer_manager.py]:::svc
  FILES[files.py]:::svc
  MASKS[masks.py]:::svc
  PRESET[presets.py]:::svc
  IH[image_history.py]:::svc
  LAYOUT[layout.py]:::svc
  EGDB[(EGDB)]:::db

  EB <-->|pub/sub| PR
  EB <-->|pub/sub| LM
  EB --> EGDB

  LM --> PR
  PR --> CMP
  FILES --> PR
  MASKS --> PR
  PRESET --> PR
  IH --> PR
  LAYOUT --> CMP

  classDef bus fill:#1c7ed6,stroke:#123a62,color:#fff;
  classDef svc fill:#f08c00,stroke:#742,color:#000;
  classDef db fill:#4c6ef5,stroke:#233,color:#fff;
```

---

### D) Scenariusz: „Zastosuj preset → render na Canvas” (sequence)

```mermaid
sequenceDiagram
  autonumber
  participant User
  participant Menu as views/menu.py
  participant EB as core/EventBus
  participant PR as services/PipelineRunner
  participant CMP as services/Compositor
  participant CC as widgets/CanvasContainer
  participant HUDS as HUDStore
  participant HUDV as views/hud.py
  Note over User,HUDV: Operacja: wybór presetu i zastosowanie filtra

  User->>Menu: click "Apply Preset"
  Menu->>EB: publish preset.apply(preset_id)
  EB->>PR: event preset.apply
  PR->>CMP: build pipeline + run
  CMP-->>PR: frame/result
  PR-->>EB: event pipeline.done(result_meta)
  EB->>HUDS: publish metric ΔS/ΔH/ΔZ, times
  HUDS-->>HUDV: state update (α,β,Z, hotspoty)
  PR-->>CC: push new frame
  CC-->>User: redraw viewport
```

## Co ten plan „zamyka” i gdzie go użyć

* **Pokrywa wszystkie warstwy:** *Core/Root → GUI (views/widgets/panels) → Services → Stores/EGDB*.
* **Oddaje realne zależności** z Twojego repo (m.in. `pipeline_runner`, `compositor`, `layer_manager`, `canvas_container`, `hud`).
* **Kanał zdarzeń BUS** jest jawny (pub/sub), a **HUDStore** i **EGDB** mają osobne węzły (brak krawędzi do subgrafów).
* **Gotowe do README.md/ARCHITECTURE.md** – bez walki z parserem GitHuba.

**Opis komponentów:**  
`<!-- @auto:gui.components -->`
