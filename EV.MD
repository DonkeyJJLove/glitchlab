

* **Led/Owned** (moduł *prowadzi*): wewnętrzne orkiestracje i wykonanie.
* **Exposed/Facade** (moduł *udostępnia*): stabilne porty/kontrakty na zewnątrz.

Poniżej finalny, spójny układ „kafelkowy” z portami, bez przenoszenia algorytmów z rootów plików.

---

# 1) Struktura modułu (kafelek)

```
core/
  pipeline.py                # ALGORYTM (root)  ← zostaje
  __init__.py
  ports/                     # PORTY = exposed services (interfejs)
    core_exec.py            # request/reply do pipeline
    core_events.py          # publikacja core.stage.*, core.metrics.*
  services/                  # LED = owned services (wewnętrzna orkiestracja)
    bus_adapter.py          # wpięcie do szyny (Event/Data)
    telemetry_sink.py       # lekka telemetria
  tile.yaml                  # manifest kafelka (patrz niżej)
```

Analogicznie np. `mosaic/`:

```
mosaic/
  hybrid_ast_mosaic.py       # ALGORYTM (root)  ← zostaje
  ports/                     # exposed services (np. mosaic.view.update)
  services/                  # owned (np. fuzja analysis.git + ast.delta)
  tile.yaml
```

---

# 2) Manifest kafelka (mozaikowy protokół kontekstu)

`core/tile.yaml` — **źródło prawdy** dla ludzi, AI i narzędzi:

```yaml
tile: core
algorithm:
  entry_file: pipeline.py
  entry_points:
    - apply_pipeline(img_u8, ctx, steps, **kw)
ports:
  exposed:                                  # to wystawiamy na zewnątrz
    - name: core.exec
      style: request_reply                   # RPC na szynie
      request:  core.exec.request            # topic IN
      reply:    core.exec.reply              # topic OUT
      schema:   vnd.glx/core.exec@1
      fallback_plan: F_PIPE_EXEC             # polityka błędu
    - name: core.events
      style: pubsub
      publish:
        - core.stage.start
        - core.stage.end
        - core.metrics.step
        - image.result
      schema: vnd.glx/core.events@1
  led:                                      # to moduł prowadzi wewnątrz
    - name: core.run                        # orkiestracja run.* z GUI
      subscribe: [run.start, run.progress]
      publish:   [run.done, run.error]
      schema:    vnd.glx/run.*@1

policy:
  fail_fast: true
  qos:
    event_plane: best_effort
    data_plane: reliable
mosaic_profile:
  expect_alpha: high     # S wys.
  expect_beta:  low      # H niskie (mało mostów bezpośrednich)
  depth: medium
```

> **AI/analiza** czyta `tile.yaml` + #tagi → buduje Δ i graf mozaiki. GUI może też wykorzystać manifest do auto-podpowiedzi.

---

# 3) #Tagi przy algorytmie i portach

W **`core/pipeline.py`** (algorytm zostaje, dodajemy metadane):

```python
# glx:tile=core
# glx:port.exposed=core.events
# glx:port.exposed=core.exec
# glx:port.led=core.run
# glx:topic.publish=core.stage.start,core.stage.end,core.metrics.step,image.result
# glx:topic.request_reply=core.exec.request->core.exec.reply
# glx:fallback.plan=F_PIPE_EXEC

def apply_pipeline(img_u8, ctx, steps, **kw):
    # glx:event=enter_scope:apply_pipeline   # AST(+1,+1,+1) ; Mozaika(+1,0,0)
    ...
    # glx:event=link:core.stage.start        # H += w(call/import)=1.5
    ...
    # glx:event=exit_scope
```

W **`core/ports/core_exec.py`** (exposed/facade):

```python
# glx:tile=core
# glx:port.exposed=core.exec
# glx:topic.subscribe=core.exec.request
# glx:topic.publish=core.exec.reply
# glx:schema=vnd.glx/core.exec@1

def on_request(evt):                      # request → reply
    # glx:event=enter_scope:on_request
    ok, out = safe_run_apply_pipeline(evt.payload)
    topic = "core.exec.reply"
    payload = {"ok": ok, "output": out, "corr": evt.correlation_id}
    bus.publish(topic, payload, source="core")
    # glx:event=exit_scope
```

W **`core/ports/core_events.py`**:

```python
# glx:port.exposed=core.events
# glx:topic.publish=core.stage.start,core.stage.end,core.metrics.step,image.result
def emit_stage(name, i, status="start"): ...
def emit_metrics(i, metrics_in, metrics_out): ...
```

> Dzięki temu **algorytm** jest czysty, a publikacje przechodzą przez lekkie funkcje portów. ΔH liczymy z #tagów w portach (mosty), a ΔS z zakresów/struktur.

---

# 4) Szyna danych (bez zmiany GUI) — „ports” wiążą się z bus

* **Event plane**: dalej `gui/event_bus` (tk.after_idle), wpięty przez adapter.
* **Data plane**: broker in-proc (duże payloady).
* **Binding**: `core/services/bus_adapter.py` łączy `ports/*` z odpowiednią szyną.

```python
# glx:tile=core
# glx:port.binding=event|data
class BusAdapter:
    def bind_core_exec(self):
        bus.subscribe("core.exec.request", core_exec.on_request)
    def bind_core_events(self):
        core_events.install(bus.publish)  # emit_* użyją przekazanej publish
```

---

# 5) Fallback = determinista polityk per PORT (nie „uprzejmość”)

`F_PIPE_EXEC` (w `.glx/bus.yaml` lub `tile.yaml`):

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
        - emit: { topic: "run.error", plane: event, payload: { error: "timeout" } }
```

> **Port** niesie plan fallback → system wie „co dalej zrobić”, a nie tylko informuje usera.

---

# 6) Mapowanie #tag → Zdarzenie → Δ (doprecyzowanie dla portów)

* `glx:port.exposed=NAME` przy `enter_scope` pliku/klasy portu →
  **AST**: (+1,+1,+1), **Mozaika**: (+1,0,0)
* `glx:topic.subscribe=X` → `use(X)` → ΔH += 1 × w(sub)
* `glx:topic.publish=Y` → `define(Y)` + `link(self→Y)` → ΔH += w(pub)
* `glx:topic.request_reply=A->B` → `define(A), define(B), link(A→B)` → ΔH += 1 + 1 + w(rpc)=2
* `glx:fallback.plan=F_*` → meta-define (ΔH+=1)
  **Wagi domyślne**: import/call/rpc=1.5; schema/contract=2.0; telemetry=0.5.

---

# 7) Minimalne zmiany w istniejącym kodzie (bez ruszania algorytmu)

* **Zostawiasz** `core/pipeline.py` jak jest; tylko dodajesz #tagi + używasz funkcji z `ports/core_events.py` do publikacji.
* **Dodajesz** `core/ports/core_exec.py` (exposed RPC) oraz `core/services/bus_adapter.py` (wiązanie do bus).
* **Manifest** `core/tile.yaml` spina całość; narzędzie `busctl` potrafi z manifestu wygenerować wpisy do `.glx/bus.yaml`.

---

# 8) Human-in-the-Loop (HITL) — edycja portów i zdarzeń

* **`busctl`**: `glx bus port add core core.exec --style request_reply`
  `glx bus port bind core core.exec.request --schema vnd.glx/core.exec@1`
  `glx bus emit core.exec.request '{...}' --await-reply`
* **GUI (przyszłość)**: panel „Ports” — edycja `tile.yaml` + walidacja schem; auto-Δ dla mozaiki.

---

# 9) Co to daje w metrykach S/H/Z

* **S↑**: porty i zakresy usług są jawne (czysta struktura).
* **H↓ w algorytmach**: mosty zewnętrzne lądują w `ports/` (nie w `pipeline.py`).
* **Z** stałe: root-algorytmy zostają na tym samym poziomie abstrakcji; porty wnoszą własne ΔZ tylko przy RPC.

---

# 10) Krótka checklista migracji

1. Utwórz `core/ports/` i `core/services/bus_adapter.py`.
2. Dodaj `core/tile.yaml` (jak wyżej).
3. Otaguj `pipeline.py` i nowo dodane pliki portów.
4. Podłącz adapter busa i zbierz `run.*` → `core.events` → GUI/analysis.
5. Dodaj `mosaic/tile.yaml` i prosty port `mosaic.view.update`.

---

# A. Notacja i założenia

* **Δ = (ΔS, ΔH, ΔZ)** — przyrosty struktury, semantyki i głębokości/abstrakcji.
* **Warstwy**:

  * **AST** (kod realny) — zdarzenia wynikające z tagów + analizy kodu.
  * **Mozaika** (meta-graf) — te same zdarzenia odwzorowane na węzły meta i centroidy.
* **Jednostka zliczania**: domyślnie **1** per zdarzenie; warianty pozwalają skalić o arność (np. liczba argumentów) lub wagę tematu.
* **Normalizacja**: po zliczeniu Δ → liczymy **α = S/(S+H), β = H/(S+H)** (gdy S+H>0), Z normalizujemy do [0,1] per plik/katalog.
* **Źródła zdarzeń**:

  1. **Tagi** `# glx:…` (statyczne).
  2. **Sygnatury AST** (import/call/def/use – jeśli włączone).
  3. **Runtime** (EventBus `run.*`) — mapowane do ΔH (mosty wykonania).

---

# B. Rdzeń mapowania (deterministyczny)

Poniższa tabela to **kanon**. Warianty (sekcja C) nadpisują lub modulują wartości.

| #tag (w komentarzu)                     | Zdarzenie logiczne     | Δ (AST)      | Δ (Mozaika)              | Uwagi                                           |
| --------------------------------------- | ---------------------- | ------------ | ------------------------ | ----------------------------------------------- |
| `glx:event=enter_scope:Name`            | `enter_scope(Name)`    | (+1,+1,+1)   | (+1,0,0)                 | Nowy blok/zakres; +H w AST za definicję symbolu |
| `glx:event=exit_scope`                  | `exit_scope()`         | (0,0,−1)     | (0,0,0) lub kontrakcja λ | Zmniejsza Z; w Mozaice opcjonalna kontrakcja    |
| `glx:event=define:Symbol`               | `define(Symbol)`       | (0,+1,0)     | (0,+1,0)                 | Deklaracja/alias/soft-import                    |
| `glx:event=use:Symbol`                  | `use(Symbol)`          | (0,+1,0)     | (0,+1,0)                 | Referencja/wykorzystanie                        |
| `glx:event=link:A->B`                   | `link(A,B)`            | (0,+1,0)     | (0,+1,0)×w(B)            | Most semantyczny; w(B) – waga tematu            |
| `glx:event=bucket_jump:b->b'`           | `bucket_jump(b→b')`    | (0,0,b'−b)   | (0,0,b'−b)               | Skok abstrakcji                                 |
| `glx:event=reassign:node@K->K'`         | `reassign(node,K→K')`  | (+1,+h,0)    | (+1,+h,0)                | h = liczba nowych mostów po migracji            |
| `glx:event=contract:K,k=n`              | `contract(K,n)`        | (−(n−1),0,0) | (−(n−1),0,0)             | Scalanie n węzłów                               |
| `glx:event=expand:K,k=n`                | `expand(K,n)`          | (+(n−1),0,0) | (+(n−1),0,0)             | Rozszerzenie n węzłów                           |
| `glx:contracts.publish=topicA,topicB`   | — (kontrakt)           | —            | —                        | Sam kontrakt ≠ zdarzenie; runtime doda Δ        |
| `glx:mosaic.S=x,y` / `glx:mosaic.H=x,y` | — (deklaracja profilu) | —            | —                        | Profil do walidacji/priorytetyzacji             |

**Wagi tematyczne mostów (domyślne)**:
`w(topic)` = 1 (zwykły), **1.5** (import/call/API), **2** (schema/kontrakt), **0.5** (telemetria/log).

---

# C. WARIANTY mapowania (rozszerzenia)

## C1. Arność i typ operacji

* **`define`**: ΔH = 1 per symbol; dla aliasów `import as` → ΔH += liczba aliasów.
* **`use`**: ΔH = liczba użytych symboli w linii (gdy parser AST włączony); inaczej =1.
* **`link`**: ΔH = w(topic) × *arity*, gdzie *arity* = liczba elementów w payloadzie (np. liczba argumentów wywołania).

## C2. Kontekst miejsca (plik/klasa/funkcja)

* Wejście w publiczne API (`glx:ast.fn=*` *i* `public_api` w GLX-CTX) → **premia S:+1** przy `enter_scope`.
* Metody prywatne (`_name`) → Δ bez premii.

## C3. Runtime EventBus (`run.*`)

Mapujemy obserwacje czasu wykonania na ΔH:

* `run.start` → ΔH += 1 (inicjacja mostu GUI→Core).
* `run.progress` → ΔH += **ε** (mały przyrost, np. 0.1), kumulowany; przekroczenia 1.0 zaokrąglamy do 1 na raport.
* `run.done` → ΔH += 1 (domknięcie mostu) + ΔS += 1 (utrwalenie krawędzi).
* `run.error` → ΔH += 1, **znacznik naruszenia** (I*), bez ΔS.

## C4. Soft-import / fallback

* `glx:contracts.note=soft-import` przy `try/except` → **nie liczymy ΔS/ΔH** dla ścieżki fallback; liczymy tylko dla ścieżki „happy-path” (import udany).
* Przy awarii importu w runtime → rejestrujemy `define(use)` *shadow* w Mozaice (bez S).

## C5. Wątki i blokady (execution structure)

* Uruchomienie wątku (`thread_start`) → ΔS += 1.
* Zastosowanie locka (`lock_guard`) → ΔS += 1 (pierwsze użycie w danym zakresie), dalsze użycia w tym samym zakresie → 0.

## C6. Skoki abstrakcji λ / bucket

* Jeżeli funkcja deklaruje `glx:mosaic.S` i `glx:mosaic.H` **oraz** `bucket_jump`, wtedy:

  * podczas `enter_scope` → ΔZ += `bucket_target − bucket_current` (jeśli jawnie zadano).
  * w przeciwnym razie ΔZ z domyślnej głębokości bloków (AST).

## C7. Zdarzenia złożone (kompozyty)

* **`define+link`** (np. rejestracja API i od razu most): ΔH = 1 (define) + w(topic) (link) — **połączone** w jedną paczkę.
* **`use+link`** (wywołanie funkcji obcej): ΔH = 1 (use) + w(call)=1.5.
* **`reassign`** w obrębie tego samego centroidu: ΔH = h (zmiana mostów lokalnych); **ΔS = 0**.

## C8. Heurystyka: panele GUI ↔ filtry

* Tag `glx:contracts.publish` zawierający `run.*` w klasach `gui/services/*` → **wzmocnienie** w(topic) = 1.25 (GUI jest „przekaźnikiem” jakości).

---

# D. Liczenie Δ — pseudokod agregatora

```python
def accumulate_deltas(tags, ast_info=None, runtime_events=None, weights):
    S=H=Z=0
    bucket = 0  # aktualny poziom, opcjonalnie z GLX-CTX
    for tag in tags:  # uporządkowane po liniach
        k, v = tag.key, tag.val  # np. ('event', 'enter_scope:run')
        if k == 'event':
            kind, *args = parse_event(v)
            if kind == 'enter_scope':
                S += 1; H += 1; Z += 1;  # AST
                S += 1                   # Mozaika: struktura meta
            elif kind == 'exit_scope':
                Z -= 1
            elif kind == 'define':
                H += 1
            elif kind == 'use':
                H += 1
            elif kind == 'link':
                topic = args[-1] if '->' in v else 'generic'
                H += weights.topic(topic) * weights.arity(args)
            elif kind == 'bucket_jump':
                dz = bucket_delta(args); Z += dz; bucket += dz
            elif kind == 'reassign':
                h = estimate_new_links(args); S += 1; H += h
            elif kind == 'contract':
                n = int(args[0].split('=')[1]); S -= (n-1)
            elif kind == 'expand':
                n = int(args[0].split('=')[1]); S += (n-1)

    # AST heurystyki (opcjonalne)
    if ast_info:
        H += ast_info.num_calls * weights.topic('call')
        H += ast_info.num_imports * weights.topic('import')

    # Runtime (EventBus)
    for ev in runtime_events or []:
        if ev.topic == 'run.start': H += 1
        elif ev.topic == 'run.progress': H += weights.epsilon
        elif ev.topic == 'run.done': H += 1; S += 1
        elif ev.topic == 'run.error': H += 1; mark_violation('runtime_error')

    return S, H, Z
```

**Parametry wag (domyślnie):**

```python
weights = {
  'topic': lambda t: {'import':1.5, 'call':1.5, 'schema':2, 'telemetry':0.5}.get(t, 1.0),
  'arity': lambda args: max(1, infer_arity(args)),
  'epsilon': 0.1
}
```

---

# E. Profile mapowania — rodziny przypadków

## E1. GUI → Core (np. `pipeline_runner`)

* Tagi: `enter_scope`, `link:gui.event_bus.EventBus.publish->core.pipeline.apply_pipeline`, `contracts.publish=run.*`.
* Oczekiwane Δ: **S:** +2 (thread+lock), **H:** ≥3 (publish/use/call), **Z:** +1 na `_runner`.

## E2. Adaptery analizy (np. `analysis/mosaic_adapter.py`)

* Tagi: dużo `define/use/link` na symbolach analitycznych i mapach.
* Δ: **H** dominuje; **S** niewielkie (struktura narzędzia).

## E3. Moduły Mozaiki (np. `mosaic/hybrid_ast_mosaic.py`)

* Tagi: `contract/expand`, `reassign`, `bucket_jump`.
* Δ: **S** zmienne (kontrakcje/ekspansje), skoki **Z** widoczne.

## E4. Filtry (`filters/*.py`)

* Tagi: `enter_scope`, `define` (parametry), `use` (mat/np/cv).
* Δ: **H** ~ liczba punktów styku z bibliotekami; **S** umiarkowane.

---

# F. Walidacje i reguły jakości (automaty do hooków)

* **I1**: sparowanie `enter_scope` ↔ `exit_scope` na poziomie funkcji/klasy; brak pary → ostrzeżenie.
* **I2**: `α+β == 1` (gdy `S+H>0`); odchyłka > 1e-9 → błąd normalizacji.
* **I3**: `contracts.publish` bez żadnego `_emit`/`bus.publish` → ostrzeżenie.
* **I4**: `link` do nieistniejącego symbolu (brak `define`/`import`) → [EVIDENCE GAP].
* **I5 (runtime)**: sekwencja `run.start` → `run.done|run.error` wymagana; brak domknięcia → alert.

---

# G. Produkty wyjściowe (formaty)

* `delta_stream.jsonl` — zdarzenie/linia: `{"file":"…","line":N,"event":"link","delta":{"S":0,"H":1.5,"Z":0},"layer":"AST|Mosaic"}`
* `delta_totals.json` — sumy Δ per plik/katalog; `alpha_beta.json` — α,β,Z po normalizacji.
* `violations.json` — lista naruszeń I1–I5 z kontekstem (plik, linia, opis).

---

# H. Przykład „na żywo” — fragment `pipeline_runner.py` (skrót)

* `# glx:event=enter_scope:PipelineRunner` → (+1,+1,+1 ; +1,0,0)
* `# glx:mosaic.S=thread_ctx,lock_guard` → (meta) informacja profilowa (brak Δ)
* `# glx:event=enter_scope:run` → (+1,+1,+1 ; +1,0,0)
* `# glx:event=link:gui.event_bus.EventBus.publish->core.pipeline.apply_pipeline` → (0,+1.5,0 ; 0,+1.5,0)
* `_emit` z `# glx:event=use:bus.publish` → (0,+1,0 ; 0,+1,0)
* `_runner` `# glx:event=use:apply_pipeline` → (0,+1.5,0 ; 0,+1.5,0)
* `# glx:event=exit_scope` ×3 (dla metod) → (0,0,−1 ; 0,0,0) każdorazowo

---

# I. Rozszerzalność: nowe #tagi

| Proponowany tag             | Sens                       | Δ domyślnie                                      |                            |
| --------------------------- | -------------------------- | ------------------------------------------------ | -------------------------- |
| `glx:event=io_read:path`    | użycie I/O (wejście)       | H += 0.5                                         |                            |
| `glx:event=io_write:path`   | użycie I/O (wyjście)       | H += 0.5; S += 0 (lub +1 jeśli tworzy strukturę) |                            |
| `glx:event=policy:GateName` | reguła/polityka na wejściu | H += 1                                           |                            |
| `glx:security=sandboxed     | trusted`                   | znacznik zaufania                                | Δ brak; wpływ na walidacje |
| `glx:perf=hotpath`          | gorąca ścieżka             | Δ brak; wpływ na wagi w(topic)=1.75              |                            |

---

# J. Minimalna specyfikacja parsera (CSV)

* **Regex**: `^\s*#\s*glx:(?P<key>[\w\.\*]+)=(?P<val>.+)$`
* **Lista**: CSV bez spacji (`a,b,c`), liczby i identyfikatory bez cudzysłowów; wartości z `:` parsujemy jako `kind:arg`.
* **Kolejność**: sekcja/linia źródłowa wyznacza kolejność zdarzeń (stabilność liczenia Z).

---

## Krótki „takeaway”

* Masz **kanon** mapowania (B), **warianty** (C) i **agregator** (D).
* Walidacje (F) spinają to z hookami; produkty (G) są gotowe do wpięcia w `analysis/` i `mosaic/`.
* Przykład (H) pokazuje, że `pipeline_runner.py` naturalnie „nabija” H (mosty GUI↔Core) i trochę S (struct exec).

---