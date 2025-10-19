# Specyfikacja kafelków GLX

> **Cel dokumentu** – spójne wytyczne dla ludzi i narzędzi (parser, GUI, CI) dotyczące:
>
> 1. Struktury katalogów pojedynczego **kafelka** (module‑tile)
> 2. Pliku **`tile.yaml`** jako „źródła prawdy”
> 3. Znaczników **`# glx:*`** w kodzie źródłowym
> 4. Algorytmu **Mapowania #tag → Zdarzenie → Δ (AST & Mozaika)**
>
> Dokument jest podzielony na dwie części:
>
> * **Część I – Migracja i konwencje kafelków** (rozdz. 1–10)
> * **Część II – Specyfikacja mapowania** (rozdz. A–J)

---

## Część I – Migracja i konwencje kafelków

### 1. Struktura katalogu kafelka

```
core/
  pipeline.py            # ALGORYTM (root) – zostaje nietknięty
  __init__.py
  ports/                 # PORTY = exposed services (interfejs)
    core_exec.py        # request/reply do pipeline
    core_events.py      # publikacja core.stage.*, core.metrics.*
  services/              # LED = owned services (wewnętrzna orkiestracja)
    bus_adapter.py      # wpięcie do szyny (Event/Data)
    telemetry_sink.py   # lekka telemetria
  tile.yaml              # manifest kafelka (patrz § 2)
```

Analogicznie dla `mosaic/`:

```
mosaic/
  hybrid_ast_mosaic.py   # ALGORYTM (root)
  ports/                 # np. mosaic.view.update
  services/              # np. fuzja analysis.git + ast.delta
  tile.yaml
```

---

### 2. Plik `tile.yaml` – manifest kafelka

```yaml
# core/tile.yaml

tile: core
algorithm:
  entry_file: pipeline.py
  entry_points:
    - apply_pipeline(img_u8, ctx, steps, **kw)
ports:
  exposed:                        # wystawiane na zewnątrz
    - name: core.exec
      style: request_reply        # RPC na szynie
      request:  core.exec.request # topic IN
      reply:    core.exec.reply   # topic OUT
      schema:   vnd.glx/core.exec@1
      fallback_plan: F_PIPE_EXEC  # polityka błędu
    - name: core.events
      style: pubsub
      publish:
        - core.stage.start
        - core.stage.end
        - core.metrics.step
        - image.result
      schema: vnd.glx/core.events@1
  led:                            # prowadzone wewnętrznie
    - name: core.run
      subscribe: [run.start, run.progress]
      publish:   [run.done, run.error]
      schema:    vnd.glx/run.*@1

policy:
  fail_fast: true
  qos:
    event_plane: best_effort
    data_plane: reliable

mosaic_profile:
  expect_alpha: high     # wysokie S
  expect_beta:  low      # niskie H (mało mostów)
  depth: medium
```

> **Do czego służy manifest?** Jest jedynym „źródłem prawdy” dla analizy AI, narzędzia `busctl`, a w przyszłości GUI.

---

### 3. Znaczniki **`# glx:*`** w algorytmie i portach

#### 3.1 `core/pipeline.py`

```python
# glx:tile=core
# glx:port.exposed=core.events
# glx:port.exposed=core.exec
# glx:port.led=core.run
# glx:topic.publish=core.stage.start,core.stage.end,core.metrics.step,image.result
# glx:topic.request_reply=core.exec.request->core.exec.reply
# glx:fallback.plan=F_PIPE_EXEC

def apply_pipeline(img_u8, ctx, steps, **kw):
    # glx:event=enter_scope:apply_pipeline   # AST(+1,+1,+1); Mozaika(+1,0,0)
    ...
    # glx:event=link:core.stage.start        # H += w(call/import)=1.5
    ...
    # glx:event=exit_scope
```

#### 3.2 `core/ports/core_exec.py`

```python
# glx:tile=core
# glx:port.exposed=core.exec
# glx:topic.subscribe=core.exec.request
# glx:topic.publish=core.exec.reply
# glx:schema=vnd.glx/core.exec@1

def on_request(evt):                 # request → reply
    # glx:event=enter_scope:on_request
    ok, out = safe_run_apply_pipeline(evt.payload)
    topic = "core.exec.reply"
    payload = {"ok": ok, "output": out, "corr": evt.correlation_id}
    bus.publish(topic, payload, source="core")
    # glx:event=exit_scope
```

#### 3.3 `core/ports/core_events.py`

```python
# glx:port.exposed=core.events
# glx:topic.publish=core.stage.start,core.stage.end,core.metrics.step,image.result

def emit_stage(name, i, status="start"): ...

def emit_metrics(i, metrics_in, metrics_out): ...
```

> **Dlaczego?** Algorytm pozostaje czysty; publikacje/prenumeraty obsługują lekkie funkcje portów.

---

### 4. Szyna danych i adapter busa

* **Event‑plane**: `gui/event_bus` (Tk `after_idle`).
* **Data‑plane**: lekki broker in‑proc (duże payload‑y).
* **Adapter** `core/services/bus_adapter.py` spina porty z bus‑em.

```python
# glx:tile=core
# glx:port.binding=event|data

class BusAdapter:
    def bind_core_exec(self):
        bus.subscribe("core.exec.request", core_exec.on_request)

    def bind_core_events(self):
        core_events.install(bus.publish)  # emit_* użyją przekazanej publish
```

---

### 5. Fallback‑i (determinista polityk per port)

```yaml
fallbacks:
  F_PIPE_EXEC:
    - match: { when: "violations has 'schema_error' or payload.steps is null" }
      do:
        - emit: { topic: "analysis.request.snapshot", plane: data,
                  payload: { reason: "missing_steps" }, expects_reply: true }
        - emit: { topic: "core.exec.reply", plane: event,
                  payload: { ok: false, error: "analysis-recover" } }
    - match: { when: "payload.timeout_ms > 5000" }
      do:
        - emit: { topic: "run.error", plane: event,
                  payload: { error: "timeout" } }
```

> Plan fallback jest definiowany przy porcie – system automatycznie wie, co robić dalej.

---

### 6. Tag → Zdarzenie → Δ – zasady dla portów

* `glx:port.exposed=NAME` przy `enter_scope` pliku/klasy portu →
  **AST** (+1,+1,+1), **Mozaika** (+1,0,0)
* `glx:topic.subscribe=X` → `use(X)` → ΔH += 1 × w(sub)
* `glx:topic.publish=Y`   → `define(Y)` + `link(self→Y)` → ΔH += w(pub)
* `glx:topic.request_reply=A->B` → `define(A), define(B), link(A→B)` → ΔH += 1 + 1 + w(rpc)=2
* `glx:fallback.plan=F_*` → meta‑define (ΔH += 1)

**Wagi domyślne** – import/call/RPC = 1.5; schema/kontrakt = 2.0; telemetry = 0.5.

---

### 7. Minimalne zmiany w istniejącym kodzie

1. Nie ruszaj `core/pipeline.py` – dodaj tylko tagi i wywołania portów.
2. Utwórz `core/ports/core_exec.py` i `core/services/bus_adapter.py`.
3. Dodaj manifest `core/tile.yaml`.
4. Podłącz adapter i zbieraj `run.*` → `core.events` → GUI/analysis.
5. Powtórz dla `mosaic/` (port `mosaic.view.update`, manifest).

---

### 8. Human‑in‑the‑Loop (HITL)

* **CLI `busctl`** – szybkie CRUD portów i manifestu
* **GUI (przyszłość)** – panel „Ports”, walidacja schem, auto‑Δ.

---

### 9. Wpływ na metryki S/H/Z

* **S ↑** – jawne porty i zakresy usług.
* **H ↓** w algorytmach – mosty przeniesione do portów.
* **Z ≈** – root‑algorytmy zostają na tym samym poziomie abstrakcji.

---

### 10. Checklist migracji (TL;DR)

| # | Zadanie                          | Gotowe? |
| - | -------------------------------- | :-----: |
| 1 | `core/ports/` + `bus_adapter.py` |    ☐    |
| 2 |  `core/tile.yaml`                |    ☐    |
| 3 |  Tagi w `pipeline.py` i portach  |    ☐    |
| 4 |  Wpięcie adaptera do bus‑a       |    ☐    |
| 5 |  Manifest + port w `mosaic/`     |    ☐    |

---

## Część II – Specyfikacja mapowania #tag → Zdarzenie → Δ

### A. Notacja i założenia

* **Δ = (ΔS, ΔH, ΔZ)** – przyrosty struktury **S**, semantyki **H** i głębokości/abstrakcji **Z**.
* **Warstwy**:

  * **AST** – zdarzenia z rzeczywistego kodu.
  * **Mozaika** – meta‑graf (kafelkowy centric view).
* **Normalizacja**: po zliczeniu Δ, obliczamy

  * **α = S / (S+H)**,
  * **β = H / (S+H)** (gdy S+H > 0),
    * **Z** skalujemy do ⟨0;1⟩ per plik/katalog.
* **Źródła zdarzeń**: tagi, sygnatury AST, runtime (EventBus `run.*`).

---

### B. Rdzeń mapowania (kanon)

| Tag                             | Zdarzenie           | Δ (AST)      | Δ (Mozaika)   | Uwagi                    |
| ------------------------------- | ------------------- | ------------ | ------------- | ------------------------ |
| `glx:event=enter_scope:Name`    | `enter_scope(Name)` | (+1,+1,+1)   | (+1,0,0)      | Nowy blok/zakres         |
| `glx:event=exit_scope`          | `exit_scope()`      | (0,0,-1)     | (0,0,0)       | Zamknięcie zakresu       |
| `glx:event=define:Sym`          | `define(Sym)`       | (0,+1,0)     | (0,+1,0)      | Deklaracja/alias         |
| `glx:event=use:Sym`             | `use(Sym)`          | (0,+1,0)     | (0,+1,0)      | Referencja               |
| `glx:event=link:A->B`           | `link(A,B)`         | (0,+1,0)     | (0,+1,0)×w(B) | Most semantyczny         |
| `glx:event=bucket_jump:b->b'`   | `bucket_jump`       | (0,0,b'-b)   | (0,0,b'-b)    | Skok abstrakcji          |
| `glx:event=reassign:node@K->K'` | `reassign`          | (+1,+h,0)    | (+1,+h,0)     | h = liczba nowych mostów |
| `glx:event=contract:K,k=n`      | `contract`          | (-(n-1),0,0) | j.w.          | Scalanie węzłów          |
| `glx:event=expand:K,k=n`        | `expand`            | (+(n-1),0,0) | j.w.          | Rozszerzenie             |
| `glx:contracts.publish=…`       | —                   | —            | —             | Kontrakt; runtime doda Δ |

**Domyślne wagi `w(topic)`**: 1 (typowy), 1.5 (import/call), 2 (schema), 0.5 (telemetria).

---

### C. Warianty mapowania

1. **Arność & typ** – link może ważyć więcej przy większej arności.
2. **Kontekst** – premiujemy publiczne API (+S).
3. **Runtime `run.*`** – `run.progress` inkrementuje H o ε = 0.1 (lub wariant – patrz pytanie na końcu).
4. **Soft‑import/fallback** – nie liczymy Δ, jeśli import się nie udał.
5. **Wątki & locki** – każde pierwsze użycie → +S.
6. **Skoki abstrakcji (`bucket_jump`)** – zmiana Z.
7. **Zdarzenia złożone** – `define+link`, `use+link`.
8. **GUI Heurystyka** – `run.*` w panelach GUI → wagi ×1.25.

---

### D. Algorytm agregacji (pseudokod)

```python
S = H = Z = 0
for tag in ordered_tags:
    ... # patrz pełny kod w repo
return normalise(S, H, Z)
```

Pełny kod referencyjny → `analysis/delta_accumulator.py`.

---