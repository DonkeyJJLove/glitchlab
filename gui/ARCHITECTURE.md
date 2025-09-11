# GlitchLab GUI v2 — Architektura

## 1) Cele i założenia

* **Rewolucja vs stan obecny:** stały layout, **dockowalne/pływające panele**, duży podgląd, środkowy mini-graf, mocne HUD-diagnostyki.
* **Modułowość:** panele per filtr + **GenericFormPanel** jako fallback (autogeneracja z `registry.meta`).
* **Jeden język telemetrii:** GUI czyta obrazy/metryki wyłącznie przez **klucze HUD** w `ctx.cache` (z core/analysis).
* **Deterministyka i A/B:** szybkie porównania, snapshoty, porządek presetów v2.
* **Bez zewn. frameworków UI:** czysty Tkinter/ttk + Pillow.

## 2) Layout i nawigacja

```
┌──────── top-bar ───────┐  File | Preset | Filter | Run | Compare | HUD
├──── left (preview) ────┬──────────── right (Parameters) ────────────────┐
│  ImageCanvas + Overlays│  PanelLoader: Filter Panel / GenericFormPanel  │
│  (toggle full-frame)   │  Amplitude / Edge / Mosaic (stałe sekcje)      │
├──────── bottom HUD (3 sloty) ───────────────────────────────────────────┤
│  Masks & Amplitude   |   Filter Diagnostics    |   Graph & Metrics Log  │
└─────────────────────────────────────────────────────────────────────────┘
```

* **Dock/float:** prawa kolumna i sloty HUD mogą być **odłączone** do `Toplevel` (pływające panele), z możliwością powrotu (dock).
* **Mini-graf:** w trzecim slocie HUD (lub jako „float”) — zasilany `ast/json` z `core.graph`.
* **Full-frame:** szybkie podglądy (overlay/porównanie A↔B).

## 3) System paneli

### 3.1 PanelBase (kontrakt)

```pseudocode
class PanelBase:
    # public
    def build(self, parent: tk.Widget) -> tk.Frame: ...
    def get_params(self) -> dict: ...
    def set_params(self, params: dict) -> None: ...         # opcjonalne
    def validate(self) -> list[str]: ...                    # opcjonalne
    def set_context_provider(self, provider: Callable[[], PanelContext]) -> None: ...
    on_change: Optional[Callable[[], None]] = None          # wywoływane przy zmianie pól
```

`PanelContext(mask_keys: list[str], ... )` — lekkie dane runtime (np. dropdown `mask_key`).

### 3.2 PanelLoader

* **Ścieżka:** `glitchlab/gui/panel_loader.py`
* **Zasada:** jeśli istnieje panel dedykowany (`panels/registry.py` → `register_panel("filter", PanelCls)`), użyj go; inaczej **GenericFormPanel** (czyta `registry.meta()` i *sam* buduje formularz).

### 3.3 GenericFormPanel (fallback)

* Typy: `int/float/bool/str/enum`.
* Wspólne pola (`mask_key`, `use_amp`, `clamp`) wyświetlane standardowo; `mask_key` wypełniane z `PanelContext`.
* Obsługa natywnych defaultów z rejestru; synchronizacja `on_change`.

### 3.4 Rejestr paneli

```python
# gui/panels/base.py
def register_panel(filter_name: str, cls: Type[PanelBase]) -> None: ...
def get_panel(filter_name: str) -> Type[PanelBase] | None: ...
```

* Dodawanie panelu = 1 linia w module panelu, plus import w `gui/panels/__init__.py`.

## 4) Widoki i widżety

* **ImageCanvas** (`widgets/image_canvas.py`): płynne zoom/pan, overlay warstw (maska, amplitude, mosaic, diff, FFT). Obsługuje tryb „full-frame”.
* **GraphView** (`widgets/graph_view.py`): rysuje DAG z `ast/json`.
* **MosaicView** (`widgets/mosaic_view.py`): plastry mozaiki (square/hex), nakładanie kolorów z blokowych metryk.
* **HudView** (`widgets/hud.py`): trzy sloty, każdy może pokazać różny „kanał” (maska/amp/diag/FFT/hist/diff/graph).

> Każdy widżet jest **czytnikiem cache**: przyjmuje listę kluczy (`stage/{i}/in|out|diff`, `format/jpg_grid`, `ast/json`…), umie pokazać pierwszy dostępny.

## 5) Źródła danych i przepływ

### 5.1 Wejścia GUI

* plik obrazu (Pillow)
* preset v2 (YAML → `core.pipeline.normalize_preset`)
* pojedynczy filtr + `PanelBase.get_params()`

### 5.2 Pipeline run

* `build_ctx(img, seed, cfg)`
* `apply_pipeline(img, ctx, steps, fail_fast, metrics)`
* (opcjonalnie) `analysis/exporters.export_hud_bundle(ctx)`

### 5.3 HUD kanały (klucze w `ctx.cache`)

* `stage/{i}/in`, `stage/{i}/out`, `stage/{i}/diff`, `stage/{i}/diff_stats`
* `stage/{i}/metrics_in`, `stage/{i}/metrics_out`
* `stage/{i}/fft_mag`, `stage/{i}/hist`
* `stage/{i}/mosaic`, `stage/{i}/mosaic_meta`
* `format/jpg_grid`, `format/notes`
* `ast/json`, `run/id`, `cfg/*`, `run/snapshot`

GUI **nie interpretuje** macierzy — tylko je wyświetla.

## 6) Docking i pływające panele

* **Dock**: `ttk.PanedWindow` (pion/poziom), sloty: `right_params`, `hud_slot_1..3`.
* **Undock**: konwersja zawartości slotu do `Toplevel` (z tytułem i przyciskiem „Dock back”).
* **Persistencja**: `~/.glitchlab/layout.json` — zapis/odczyt geometrii okna, pozycji splitterów, stanu paneli (dokowane/float, rozmiary).
* **Focus & z-index**: `Toplevel.attributes('-topmost', False)` + „Bring to front”.

## 7) Styl i użyteczność

* **ttk Theme**: ciemny (kolory z `controls.py`), spójne pady, siatka, etykiety wyrównane.
* **Skróty klawiszowe**: `Ctrl+O/S`, `Ctrl+R` (Run), `F` (full-frame), `1/2/3` (sloty HUD), `Ctrl+E` (export bundle).
* **Aktywne hinty**: pasek statusu pokazuje: rozmiar obrazu, seed, czas ostatniego runu, liczbę ostrzeżeń.
* **Konsekwentne nazwy**: parametry mają „kanoniczne” opisy z rejestru (`meta().doc`, `defaults`).

## 8) Struktura pakietu GUI (docelowa)

```
glitchlab/gui/
  __init__.py
  app.py                 # Application frame (bez mainloop)
  panel_base.py          # PanelBase + PanelContext
  panel_loader.py        # fabryka paneli
  controls.py            # drobne widżety formularzy
  generic_form_panel.py  # fallback (autogenerowany)
  widgets/
    image_canvas.py      # podgląd, zoom/pan, overlay
    graph_view.py        # DAG z ast/json
    mosaic_view.py       # mozaika i legendy
    hud.py               # 3-slotowy HUD z routingiem źródeł
  panels/
    __init__.py          # importy rejestrujące panele
    base.py              # register_panel/get_panel
    anisotropic_contour_warp_panel.py
    pixel_sort_adaptive_panel.py
    spectral_shaper_panel.py
    phase_glitch_panel.py
    block_mosh_grid_panel.py
    depth_displace_panel.py
    depth_parallax_panel.py
    rgb_offset_panel.py
  state.py               # wybór obrazu, preset, steps, seed, hud-mapping
  docking.py             # DockManager (dock/float), persystencja layoutu
  exporters.py           # zapis/odczyt layoutu, eksport HUD-bundle (thin DTO)
  # main.py              # uruchamiacz (GENERUJEMY NA KOŃCU)
```

## 9) API (najważniejsze interfejsy)

```python
# gui/app.py
class App(tk.Frame):
    def set_image(self, pil_image: Image.Image) -> None: ...
    def set_preset_cfg(self, cfg: dict) -> None: ...
    def set_filter(self, name: str) -> None: ...
    def run_pipeline(self) -> None: ...
    def export_bundle(self) -> dict: ...   # thin DTO z kluczami do cache
```

```python
# gui/state.py
@dataclass
class UiState:
    image: Image.Image | None
    preset_cfg: dict | None
    single_filter: str | None
    filter_params: dict
    seed: int
    hud_mapping: dict[str, list[str]]  # slot -> lista kluczy cache do prób
```

```python
# gui/docking.py
class DockManager:
    def dock(self, slot_id: str, widget: tk.Widget) -> None: ...
    def undock(self, slot_id: str) -> None: ...
    def save_layout(self) -> dict: ...
    def load_layout(self, d: dict) -> None: ...
```

## 10) Integracja z core/analysis

* **Preset v2**: GUI zawsze woła `normalize_preset` (wymusza schemat).
* **Ctx/HUD**: `build_ctx` buduje amplitude/maski; `apply_pipeline` tworzy wpisy HUD.
* **Graph**: po run — `core.graph.build_and_export_graph` zapisuje `ast/json`.
* **Analysis widoki**: FFT/hist/diff są **tylko odczytywane** (GUI nie liczy metryk).

## 11) Migracja z obecnych plików

**Wykorzystujemy:**

* `panel_base.py` → zostaje (lekka kosmetyka).
* `panel_loader.py` → zostaje (dorzucamy `try: import panels`).
* `controls.py` → zostaje, poszerzamy o `spin`, `enum`, `slider`.
* `generic_form_panel.py` → zostaje, dopinamy wsparcie `enum` i `mask_key`.

**Do dodania:**

* `widgets/image_canvas.py`, `widgets/graph_view.py`, `widgets/mosaic_view.py`, `widgets/hud.py`
* `docking.py`, `state.py`, `exporters.py`
* panele brakujące (depth\_\*, rgb\_offset\_panel…).

**Do uproszczenia/usunięcia:**

* Z `app.py` przenieść logikę layout/undock do `docking.py` i widżety do `widgets/`.
* Z paneli dedykowanych usunąć powielony kod kontrolek (użyć `controls.py`).

## 12) Persistencja i konfiguracja

* `~/.glitchlab/ui.json`:

  * ostatnia ścieżka pliku, ostatni preset, seed,
  * layout: rozmiary splitterów, panele pływające (geometria, sloty),
  * mapowanie HUD-slotów na listy kluczy (preferencje użytkownika).

## 13) Wydajność

* **Cache obrazków**: przechowuj `ImageTk.PhotoImage` dla aktualnego powiększenia; odświeżaj tylko przy zmianie zoom/pan/overlay.
* **Miniatury**: downsample wątku głównego, ale limituj częstotliwość (debounce).
* **Batch repaint**: spinbox/slider emituje zmianę dopiero po „Idle 200ms”, by nie spamować pipeline.

## 14) Testowalność

* **Smoke GUI** (headless): uruchom `App` z dummy obrazem i presetem, sprawdź że 3 sloty HUD wypełniają się placeholderami.
* **Panel cięcia**: każdy panel ma `get_params()` deterministyczne (bez side-effectów).
* **Layout**: test zapisu/odczytu `DockManager.save_layout()`.

## 15) Roadmap implementacyjny (bez `main.py`)

1. **widgets/**: `image_canvas.py`, `hud.py`, `graph_view.py`, `mosaic_view.py`.
2. **docking.py** + integracja w `app.py` (przeniesienie logiki).
3. **state.py** + „Run” → pipeline + odświeżenie HUD.
4. **exporters.py**: `export_hud_bundle(ctx)` (thin DTO) + zapis layoutu.
5. **panels/**: wyrównanie istniejących + brakujące (`depth_*`, `rgb_offset`).
6. **hotkeys + full-frame preview**.
7. **(Na koniec)** `main.py`.

---

## Załącznik A — Minimalne szkielety (przykłady)

**widgets/image\_canvas.py**

```python
class ImageCanvas(tk.Canvas):
    def set_image(self, pil: Image.Image) -> None: ...
    def set_overlay(self, name: str, pil: Image.Image | None, alpha: float = 0.5) -> None: ...
    def zoom(self, factor: float) -> None: ...
    def pan(self, dx: int, dy: int) -> None: ...
    def toggle_fullframe(self) -> None: ...
```

**widgets/hud.py**

```python
class HudSlot(tk.Frame):
    def bind_sources(self, keys: list[str]) -> None: ...
    def render_from_cache(self, cache: dict) -> None: ...

class Hud(tk.Frame):
    def set_cache(self, cache: dict) -> None: ...
    def set_slot_mapping(self, mapping: dict[str, list[str]]) -> None: ...
```

**docking.py**

```python
class DockManager:
    def __init__(self, root: tk.Tk | tk.Toplevel, slots: dict[str, tk.Frame]) -> None: ...
    def undock(self, slot_id: str, title: str) -> None: ...
    def dock(self, slot_id: str) -> None: ...
    def save_layout(self) -> dict: ...
    def load_layout(self, d: dict) -> None: ...
```

---

## Załącznik B — Mapowanie HUD (domyślne)

```yaml
hud:
  slot1: ["stage/0/in", "stage/0/metrics_in", "format/jpg_grid"]
  slot2: ["stage/0/out", "stage/0/metrics_out", "stage/0/fft_mag"]
  slot3: ["stage/0/diff", "stage/0/diff_stats", "ast/json"]
```

---

## Załącznik C — Integracja z panelami filtrów

Panel wywołuje `on_change()` → App zbiera `get_params()` → aktualizuje **UiState** → naciśnięcie „Run” odpala pipeline → HUD/preview odświeżają się z `ctx.cache`.

---

### Podsumowanie

* Mamy prosty, **przewidywalny** GUI-stack (Tkinter), ale o **dużej ergonomii**: dock/float, pełny HUD, mini-graf, mozaika.
* System paneli jest *otwarty*: panele dedykowane lub autogenerowane.
* GUI nie liczy metryk; **tylko** wyświetla to, co wystawia core/analysis HUD-kanałami.
* `main.py` dostawimy na końcu — po dopięciu `widgets/`, docking i paneli.

