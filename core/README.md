# GlitchLab Core — runtime, graf procesu i mozaika

> Warstwa **core** rejestruje i uruchamia filtry, normalizuje presety v2, zbiera metryki/diff i buduje DAG procesu oraz artefakty dla HUD/GUI. Zapewnia deterministykę, spójne API i kanały telemetrii.

---

## Kluczowe moduły

* `registry.py` – **jedno źródło prawdy** dla filtrów (`@register`, `get`, `meta`, aliasy).
* `pipeline.py` – normalizacja presetów, `Ctx`, wykonanie kroków, metryki, diff, logi; zapisy do `ctx.cache`.
* `graph.py` – DAG: węzły kroków z deltami metryk; eksport JSON do HUD (`ast/json`).
* `mosaic.py` – siatka square/hex; projekcja metryk blokowych → overlay RGB.
* `astmap.py` – **Python AST → graf** (funkcje/klasy, `calls`/`contains`) + projekcja na mozaikę; `export_ast_json`.
* `metrics/basic.py`, `metrics/compare.py` – entropia, edge density, contrast RMS, PSNR/SSIM.
* `utils.py`, `roi.py`, `symbols.py` – konwersje, maski, symbole → maski.

Szczegóły interfejsów: [ARCHITECTURE.md](ARCHITECTURE.md).

---

## API w pigułce

```python
from glitchlab.core.pipeline import normalize_preset, build_ctx, apply_pipeline
from glitchlab.core.registry import available, get

cfg = normalize_preset({
  "version": 2,
  "seed": 7,
  "amplitude": {"kind":"none","strength":1.0},
  "edge_mask": {"thresh":60,"dilate":0,"ksize":3},
  "steps": [{"name": available()[0], "params": {}}]
})

out = apply_pipeline(img_u8, build_ctx(img_u8, seed=cfg["seed"], cfg=cfg), cfg["steps"])
```

**Filter API v2:** `fn(img_u8, ctx, **params) -> np.ndarray`
Wspólne parametry: `mask_key | use_amp | clamp`. RNG wyłącznie z `ctx.rng`.

---

## Telemetria i HUD

* **Per stage:**
  `stage/{i}/in|out|diff|t_ms`, `stage/{i}/metrics_in|metrics_out|diff_stats`, `stage/{i}/fft_mag|hist`, `stage/{i}/mosaic|mosaic_meta`.
* **Diag per filtr:** `diag/<filter>/…`
* **Global:** `ast/json`, `format/jpg_grid`, `run/id`, `cfg/*`.

> GUI jest **czytnikiem cache** — nie liczy metryk, tylko wyświetla to, co przygotuje core/analysis.

---

## AST → graf → mozaika (metapoziom)

`core/astmap.py` dostarcza:

* `build_ast(source)` – parsowanie Pythona do AST.
* `ast_to_graph(tree)` – węzły `function|class` z metrykami: `weight`, `branching`, `fan_in`, `fan_out`.
* `project_ast_to_mosaic(graph, mosaic)` – przypisanie metryk do komórek (domyślnie `R←branching`, `G←weight`, `B←fan_out`).
* `export_ast_json(graph, ctx, cache_key="ast/json")` – JSON przyjazny HUD.

Przykład projekcji (pseudokod):

```python
from glitchlab.core import mosaic, astmap

tree = astmap.build_ast(open("some_module.py").read())
graph = astmap.ast_to_graph(tree)
mz = mosaic.mosaic_map((H,W), mode="square", cell_px=32)
overlay = astmap.project_ast_to_mosaic(graph, mz)  # uint8 RGB
```

---

## Deterministykę i zakresy typów

* Wejście/wyjście filtrów: `uint8 RGB (H,W,3)`.
* Wnętrze: `float32 [0,1]`.
* **Bez** SciPy/OpenCV.

---

## Status i rozszerzanie

* Dodanie nowej metryki → `metrics/basic.py` (+ opcjonalnie `analysis/metrics.py`).
* Dodanie operatorów (opcjonalny runtime node-style): `artifact.py` / `operator.py`.

---

