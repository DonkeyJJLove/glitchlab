# GlitchLab — Analiza (obszar wyświetlania obrazu) — skrót

**Cel:** Ustabilizować i rozbudować obszar wyświetlania (viewport/canvas) z naciskiem na
warstwy, kompozycję, diagnostykę i przyszłe 3D.

## Pliki i role (GUI / Services / Core)
- `gui/views/viewport.py` — widok główny obrazu (kontener dla CanvasContainer).
- `gui/widgets/image_canvas.py` — niskopoziomowy renderer (PIL↔ImageTk, zoom/pan).
- `gui/widgets/canvas_container.py` — kontroler renderu; integracja narzędzi i subskrypcji EventBus.
- `gui/services/pipeline_runner.py` — uruchamia core, aktualizuje stan po zakończeniu.
- **NOWE:** `gui/services/compositor.py` — czysta kompozycja warstw (NumPy).
- **NOWE:** `gui/services/layer_manager.py` — stan warstw + orkiestracja kompozycji.

## Kontrakty i przepływy
- Format graniczny GUI↔Core: `np.ndarray uint8 RGB` (H,W,3).
- Core zapisuje telemetrię do `ctx.cache`; HUD czyta po kluczach.
- Viewport renderuje **kompozyt warstw** z `LayerManager.get_composite_for_viewport()`.
- `PipelineRunner` wynik kieruje do **warstwy aktywnej** (lub tworzy nową).
- Repaint przez `ui.layers.changed` (EventBus).

## Naprawiane ryzyka (wg raportu IDE i obserwacji)
- Typy i argumenty: wymuszony `np.uint8` i (H,W,3) w kompozytorze.
- Atrybuty poza `__init__`: wzorce inicjalizacji `AppState` i LayerManager.
- Niejednoznaczne ścieżki danych (image_in/out vs. result): konsolidacja na warstwach.

## Co dalej (krótko)
- Dodać toolbara (ikony) publikującego `ui.tool.*` i obsługę w `CanvasContainer`.
- Panel `layer_panel.py` (lista warstw, krycie, blend).
- Cache kompozytu + inwalidacja na zmianach.
