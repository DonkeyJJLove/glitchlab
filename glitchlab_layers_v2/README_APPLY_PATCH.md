# GlitchLab Layers Patch (v2)

## Zawartość
- `glitchlab_layers_v2.patch` — patch do `git apply`
- `new_files/` — nowe pliki (fallback)
  - `glitchlab/gui/services/compositor.py`
  - `glitchlab/gui/services/layer_manager.py`
- `ANALYSIS_SUMMARY.md` — skrót analizy
- `README_APPLY_PATCH.md` — jak zastosować

## Zastosowanie
1. Przejdź do **roota repo** (gdzie leży katalog `glitchlab/`).
2. Upewnij się, że drzewo jest czyste: `git status`.
3. Zastosuj patch: `git apply /ścieżka/do/glitchlab_layers_v2.patch`
4. W razie konfliktów, skopiuj pliki z `new_files/` i ręcznie wprowadź zmiany z hunks do:
   - `glitchlab/gui/app.py`
   - `glitchlab/gui/widgets/canvas_container.py`
   - `glitchlab/gui/services/pipeline_runner.py`
5. Uruchom aplikację i sprawdź:
   - otwarcie obrazu → tworzy warstwę `Background` (jeśli nie, dodaj po `Open` wywołanie `layer_mgr.add_layer(...)`),
   - uruchomienie filtra → wynik trafia do warstwy aktywnej lub nowej (`target="new"`),
   - viewport renderuje kompozyt warstw; HUD czyta diagnostykę z `ctx.cache`.
