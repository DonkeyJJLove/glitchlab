# GlitchLab GUI — HUD, panele i pływające widoki

![UI](../screen.png)

> GUI jest **czytnikiem cache**. Renderuje obrazy i telemetrię przygotowaną przez core/analysis. Panele są modularne: dedykowane per filtr lub automatycznie generowane (GenericFormPanel).

---

## Layout

```
┌──────── top-bar ───────┐  File | Preset | Filter | Run | Compare | HUD
├──── left (preview) ────┬──────────── right (Parameters) ────────────────┐
│  ImageCanvas + Overlays│  PanelLoader: Filter Panel / GenericFormPanel  │
│  (toggle full-frame)   │  Amplitude / Edge / Mosaic (stałe sekcje)      │
├──────── bottom HUD (3 sloty) ───────────────────────────────────────────┤
│  Masks & Amplitude   |   Filter Diagnostics    |   Graph & Metrics Log  │
└─────────────────────────────────────────────────────────────────────────┘
```

* **Dock/float:** prawa kolumna i sloty HUD mogą być odpinane do `Toplevel` i dokowane z powrotem.
* **Mini-graf:** rysowany z `ast/json` (DAG procesu lub graf AST).
* **Full-frame:** przełącznik pełnoekranowego podglądu.

---

## Kanały, które GUI rozumie

* `stage/{i}/in|out|diff|t_ms`
* `stage/{i}/metrics_in|metrics_out|diff_stats`
* `stage/{i}/fft_mag|hist`
* `stage/{i}/mosaic|mosaic_meta`
* `diag/<filter>/…`
* `ast/json`, `format/jpg_grid`, `run/id`, `cfg/*`

> GUI **nie liczy** metryk — prezentuje to, co znajdzie po kluczach.

---

## Panele i formularze

* Jeśli istnieje panel dedykowany (`gui/panels/…`), zostanie użyty.
* W przeciwnym razie działa **GenericFormPanel** na podstawie `registry.meta(name)["defaults"]`.
* **PanelContext** dostarcza dynamiczne dane (np. listę `mask_key`).

---

## Skróty i ergonomia

* `Ctrl+O/S` (open/save), `Ctrl+R` (run), `F` (full-frame), `1/2/3` (sloty HUD), `Ctrl+E` (export bundle).
* Ciemny motyw ttk, spójne pady i siatka.
* Pasek statusu: rozmiar obrazu, seed, czas runu, liczba ostrzeżeń.

---

## Mapowanie mozaikowe (w GUI)

* **MosaicView** – pokazuje nakładkę z `stage/{i}/mosaic` i legendę; wspólna skala dla metryk blokowych i projekcji AST (`core/astmap.project_ast_to_mosaic`).
* **GraphView** – DAG procesu lub graf AST (z `ast/json`).

---

## Uruchomienie

```bash
python -m glitchlab.gui.app
```

Szczegóły integracji: [ARCHITECTURE.md](ARCHITECTURE.md) i [../core/ARCHITECTURE.md](../core/ARCHITECTURE.md).

---

## Plansza testowa

Do szybkich weryfikacji użyj `../glitchlab_testchart_v1.png` — wykrywa rezonanse, aliasy i „wycieki” efektów.

---