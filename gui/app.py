# glitchlab/gui/app.py
# -*- coding: utf-8 -*-
"""
GlitchLab GUI v4 — główny moduł aplikacji.

Sekcje:
1. bootstrap          – uruchamianie „z palca” (bez instalacji pakietu)
2. importy            – stdlib ▸ third-party ▸ lokalne (GlitchLab)
3. fallback-stuby     – minimalne zamienniki opcjonalnych zależności
4. AppState           – współdzielony stan
5. App                – okno główne (MENU ▸ UI ▸ BUS ▸ PIPELINE)
"""
from __future__ import annotations

# ──────────────────────── 1 • bootstrap ────────────────────────────────────
import os, sys, tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Any, Dict, Optional

if __package__ in (None, ""):
    PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if PROJ not in sys.path:
        sys.path.insert(0, PROJ)

# ──────────────────────── 2 • importy ──────────────────────────────────────
try:
    from PIL import Image, ImageOps            # pillow
except Exception:
    Image = ImageOps = None                    # type: ignore
try:
    import numpy as np                         # numpy
except Exception:
    np = None                                  # type: ignore

# core
try:
    from glitchlab.core.pipeline import build_ctx, apply_pipeline
except Exception:
    build_ctx = apply_pipeline = None          # type: ignore

# GUI – widoki / widgety
from glitchlab.gui.views.tab_general import GeneralTab, GeneralTabConfig
from glitchlab.gui.views.tab_filter  import TabFilter,  FilterTabConfig
from glitchlab.gui.views.tab_preset  import PresetsTab
from glitchlab.gui.views.statusbar   import StatusBar
from glitchlab.gui.widgets.canvas_container import CanvasContainer

# ──────────────────────── 3 • fallbacki ────────────────────────────────────
try:
    from glitchlab.gui.event_bus import EventBus
except Exception:                              # minimalny stub
    class EventBus:
        def __init__(self, *_a, **_k): ...
        def publish(self, t, p): print(t, p)
        def subscribe(self, *_a, **_k): ...
try:
    from glitchlab.gui.services.pipeline_runner import PipelineRunner
except Exception:
    PipelineRunner = None                      # type: ignore
try:
    from glitchlab.gui.views.bottom_panel import BottomPanel
except Exception:
    BottomPanel = None                         # type: ignore
try:
    from glitchlab.gui.docking import DockManager
except Exception:
    class DockManager:
        def __init__(self, *_a, **_k): ...
try:
    from glitchlab.gui.services.presets import PresetService, PresetServiceConfig
except Exception:
    class PresetService:
        def __init__(self, *_a, **_k): ...
    PresetServiceConfig = object              # type: ignore

# ──────────────────────── 4 • AppState ─────────────────────────────────────
class AppState:
    """Stan współdzielony między zakładkami/panelami."""
    def __init__(self) -> None:
        self.image: Optional["Image.Image"] = None
        self.masks: Dict[str, Any] = {}
        self.cache: Dict[str, Any] = {}
        self.preset_cfg: Optional[Dict[str, Any]] = None
        self.seed: int = 7

# ──────────────────────── 5 • App (UI + logika) ────────────────────────────
class App(tk.Frame):
    """Viewer ▸ Notebook ▸ BottomPanel ▸ StatusBar."""

    def __init__(self, master: tk.Tk, **kw: Any) -> None:
        super().__init__(master, **kw)

        # infrastruktura
        self.master = master
        self.bus    = EventBus(master)
        self.state  = AppState()
        self.runner = PipelineRunner(self.bus, master) if PipelineRunner else None

        # wspólna zmienna trybu (menu „Tools” ⇆ Toolbox)
        self.tool_var = tk.StringVar(value="pan")

        # buduj UI i podpinaj bus
        self._build_ui()
        self._wire_bus()

    # ────────────── MENU ──────────────
    def _build_menu(self) -> None:
        menubar = tk.Menu(self)
        self.master.configure(menu=menubar)

        # File
        m_file = tk.Menu(menubar, tearoff=False)
        m_file.add_command(label="Open Image…", command=self._open_image)
        m_file.add_separator()
        m_file.add_command(label="Exit", command=self.master.destroy)
        menubar.add_cascade(label="File", menu=m_file)

        # Presets
        m_preset = tk.Menu(menubar, tearoff=False)
        m_preset.add_command(label="Open Preset…",
                             command=lambda: self.bus.publish("ui.preset.open", {}))
        m_preset.add_command(label="Save Preset As…",
                             command=lambda: self.bus.publish(
                                 "ui.preset.save",
                                 {"cfg": self.state.preset_cfg or {}}
                             ))
        menubar.add_cascade(label="Presets", menu=m_preset)

        # Tools — radiobuttony zsynchronizowane z Toolboxem
        m_tools = tk.Menu(menubar, tearoff=False)
        for key, lbl in (
            ("pan",   "Pan"),
            ("zoom",  "Zoom"),
            ("ruler", "Ruler"),
            ("probe", "Probe"),
            ("pick",  "Pick Color"),
        ):
            m_tools.add_radiobutton(
                label=lbl,
                variable=self.tool_var,
                value=key,
                command=lambda n=key: self.bus.publish("ui.tools.select", {"name": n}),
            )
        menubar.add_cascade(label="Tools", menu=m_tools)

    # ────────────── UI ──────────────
    def _build_ui(self) -> None:
        """Składa całe główne okno:
           ┌───────────── PanedWindow ─────────────┐
           │  left (viewer + toolbox) │  right NB  │
           └───────────────────────────────────────┘
             ▼ opcjonalny Dock / BottomPanel
             ▼ StatusBar
        """
        # kontener App
        self.pack(fill="both", expand=True)

        # ===== MENUBAR =======================================================
        self._build_menu()

        # ===== GŁÓWNY PODZIAŁ (paned) =======================================
        self.main = ttk.Panedwindow(self, orient="horizontal")
        self.main.pack(fill="both", expand=True)

        # ------------------ LEWA STRONA: viewer + toolbox -------------------
        self.left = ttk.Frame(self.main)

        self.viewer = CanvasContainer(
            self.left,
            bus=self.bus,
            tool_var=self.tool_var,
        )
        self.viewer.pack(fill="both", expand=True)

        # ------------------ PRAWA STRONA: Notebook --------------------------
        self.right = ttk.Notebook(self.main)

        # karty
        self.tab_general = GeneralTab(
            self.right,
            ctx_ref=self.state,
            cfg=GeneralTabConfig(preview_size=240),
        )
        self.tab_filter = TabFilter(
            self.right,
            bus=self.bus,
            ctx_ref=self.state,
            cfg=FilterTabConfig(allow_apply=True),
        )
        self.tab_preset = PresetsTab(self.right, bus=self.bus)

        self.right.add(self.tab_general, text="General")
        self.right.add(self.tab_filter,  text="Filters")
        self.right.add(self.tab_preset,  text="Presets")

        # Dodajemy panele do okna dzielonego.
        # minsize → zapobiega zniknięciu widoku; weight ustawia proporcje.
        self.main.add(self.left,  weight=3)
        self.main.add(self.right, weight=6)

        # Po wyrenderowaniu okna ustawiamy początkową pozycję separatora
        def _init_sash():
            w = self.main.winfo_width()
            if w < 100:                  # okno jeszcze się nie narysowało
                self.after(30, _init_sash)
                return
            self.main.sashpos(0, int(w * 0.33))   # ~1/3 na viewer

        self.after_idle(_init_sash)

        # ===== DOLNY PANEL (opcjonalny) =====================================
        self.bottom = BottomPanel(self, bus=self.bus, default="hud") if BottomPanel else None
        if self.bottom:
            self.bottom.pack(fill="x", side="bottom")

        # ===== STATUS BAR ====================================================
        self.status = StatusBar(self, show_progress=True)
        self.status.pack(fill="x", side="bottom")
        self.status.bind_bus(self.bus)

        # ===== DOCKING =======================================================
        try:
            self.dock = DockManager(
                self.winfo_toplevel(),
                {"bottom": self.bottom or ttk.Frame(self), "right": self.right},
            )
        except Exception:
            self.dock = DockManager()    # type: ignore

        # ===== PRESET SERVICE ===============================================
        try:
            self.svc_presets = PresetService(
                self.master,
                self.bus,
                PresetServiceConfig(),
            )
        except Exception:
            self.svc_presets = PresetService(self.master, self.bus)  # type: ignore[arg-type]
# ───────────── BUS wiring ──────────────
    def _wire_bus(self) -> None:
        B = self.bus

        # start pojedynczego kroku pipeline (z FiltersTab)
        B.subscribe("ui.run.apply_filter",
                    lambda _t, p: self.run_step(
                        (p or {}).get("step") or self.tab_filter.get_current_step()
                    ))

        # progress → blokada viewer’a
        B.subscribe("run.progress",
                    lambda _t, d: self.viewer.set_enabled(
                        not (0 < float(d.get("value", 0)) < 1)
                    ))
        B.subscribe("run.done",  lambda *_: self.viewer.set_enabled(True))
        B.subscribe("run.error", lambda *_: self.viewer.set_enabled(True))

        # StatusBar
        B.subscribe("ui.status.set",
                    lambda _t, d: self.status.set_text(d.get("text", "")))
        B.subscribe("ui.cursor.pos",
                    lambda _t, d: self.status.set_right(
                        f"x={d.get('x','-')}, y={d.get('y','-')}" if d else ""
                    ))

    # ────────────── File → open image ──────────────
    def _open_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Open image",
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.webp;*.tif;*.tiff"),
                       ("All files", "*.*")])
        if not path:
            return
        if Image is None:
            messagebox.showerror("Pillow", "Pillow is required.")
            return
        try:
            img = ImageOps.exif_transpose(Image.open(path))            # type: ignore[arg-type]
            img = img.convert("RGB") if img.mode != "RGB" else img
            self.state.image = img
            self.viewer.set_image(img)
            self.status.set_text(f"Loaded: {os.path.basename(path)}")
        except Exception as ex:
            messagebox.showerror("Open image", str(ex))

    # ────────────── helpers ──────────────
    @staticmethod
    def _pil_to_u8(img: "Image.Image"):
        if np is None:
            raise RuntimeError("NumPy required")
        a = np.asarray(img, dtype=np.uint8)                         # type: ignore[arg-type]
        if a.ndim == 2:
            a = np.stack([a] * 3, axis=-1)
        if a.shape[-1] == 4:
            rgb, alpha = a[..., :3], a[..., 3:4].astype(np.float32) / 255.0
            a = (rgb * alpha).clip(0, 255).astype(np.uint8)
        return a

    def run_step(self, step: Dict[str, Any]) -> None:
        if not self.state.image:
            messagebox.showwarning("Run", "No image loaded."); return
        if not (build_ctx and apply_pipeline):
            messagebox.showwarning("Core", "core.pipeline unavailable."); return

        img_u8 = self._pil_to_u8(self.state.image)
        ctx = build_ctx(img_u8, seed=self.state.seed,                 # type: ignore[arg-type]
                        cfg=self.state.preset_cfg or {"version": 2, "steps": []})
        if self.state.masks:
            ctx.masks.update(self.state.masks)                        # type: ignore[attr-defined]

        # asynchronicznie (jeśli Runner dostępny)
        if self.runner:
            self.runner.run(img_u8, ctx, [dict(step)])                # type: ignore[arg-type]
            return

        # fallback: synchronicznie
        try:
            out = apply_pipeline(img_u8, ctx, [dict(step)],
                                 fail_fast=True, metrics=True)        # type: ignore[arg-type]
            self.bus.publish("run.done", {"output": out, "ctx": ctx})
        except Exception as e:
            self.bus.publish("run.error", {"error": str(e)})

# ────────────── entrypoint ──────────────
def main() -> None:
    root = tk.Tk()
    root.title("GlitchLab GUI v4")
    root.geometry("1200x800")
    App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
