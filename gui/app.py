# glitchlab/gui/app.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Any, Dict, Optional, List

# --- bootstrap for direct run ------------------------------------------------
if __package__ in (None, ""):
    THIS = os.path.abspath(__file__)
    GUI_DIR = os.path.dirname(THIS)
    PKG_DIR = os.path.dirname(GUI_DIR)
    PROJ = os.path.dirname(PKG_DIR)
    if PROJ not in sys.path:
        sys.path.insert(0, PROJ)

# --- soft deps ---------------------------------------------------------------
try:
    from PIL import Image, ImageOps
except Exception:
    Image = None  # type: ignore

try:
    import numpy as np
except Exception:
    np = None  # type: ignore

# --- core (właściwy import) --------------------------------------------------
try:
    from glitchlab.core.pipeline import build_ctx, apply_pipeline, normalize_preset
except Exception:
    build_ctx = None  # type: ignore
    apply_pipeline = None  # type: ignore
    normalize_preset = None  # type: ignore

# --- EventBus ----------------------------------------------------------------
try:
    from glitchlab.gui.event_bus import EventBus
except Exception:
    class EventBus:
        def __init__(self, *_a, **_k): pass

        def publish(self, topic: str, payload: Dict[str, Any]) -> None:
            print(f"[bus] {topic}: {payload}")

        def subscribe(self, topic: str, cb) -> None:
            # caller powinien wołać cb(topic, payload)
            setattr(self, f"_cb_{topic.replace('.', '_')}", cb)

# --- Runner (async wrapper) --------------------------------------------------
try:
    from glitchlab.gui.services.pipeline_runner import PipelineRunner
except Exception:
    PipelineRunner = None  # type: ignore

# --- Tabs --------------------------------------------------------------------
from glitchlab.gui.views.tab_general import GeneralTab, GeneralTabConfig
from glitchlab.gui.views.tab_filter import TabFilter, FilterTabConfig
from glitchlab.gui.views.tab_preset import PresetsTab

# --- Widgets (soft) ----------------------------------------------------------
try:
    from glitchlab.gui.widgets.image_canvas import ImageCanvas
except Exception:
    class ImageCanvas(ttk.Label):
        def set_image(self, pil_img):  # type: ignore
            try:
                from PIL import ImageTk
                self._photo = ImageTk.PhotoImage(pil_img)
                self.configure(image=self._photo, text="")
            except Exception:
                self.configure(text="(ImageCanvas unavailable)")

try:
    from glitchlab.gui.widgets.hud import Hud
    from glitchlab.gui.widgets.diag_console import DiagConsole
except Exception:
    class Hud(ttk.Frame):
        def set_cache(self, _cache: dict) -> None: pass

        def render_from_cache(self, _cache: dict) -> None: pass

# --- Docking (soft) ----------------------------------------------------------
try:
    from glitchlab.gui.docking import DockManager
except Exception:
    class DockManager:
        def __init__(self, *_a, **_k): pass

# --- Presets service ---------------------------------------------------------
try:
    from glitchlab.gui.services.presets import PresetService
except Exception:
    class PresetService:
        def __init__(self, *_a, **_k): pass


# ============================== STATE ========================================
class AppState:
    """Globalny stan (ctx_ref) widoczny dla zakładek/paneli."""

    def __init__(self) -> None:
        self.image: Optional["Image.Image"] = None
        self.masks: Dict[str, Any] = {}  # nazwy → np.ndarray (H,W) float32/u8
        self.cache: Dict[str, Any] = {}  # HUD kanały z pipeline
        self.meta: Dict[str, Any] = {}
        self.preset_text: str = ""  # surowy YAML/JSON (edytor)
        self.preset_cfg: Optional[Dict[str, Any]] = None  # znormalizowany preset
        self.seed: int = 7


# =============================== APP =========================================
class App(tk.Frame):
    """
    Left: Preview (ImageCanvas)
    Right: Notebook — General / Filters / Presets
    Bottom: HUD (jeśli widget zapewnia)
    Run: FilterTab emituje ui.run.apply_filter -> uruchamiamy core.pipeline
    """

    def __init__(self, master: tk.Tk, **kw: Any) -> None:
        super().__init__(master, **kw)
        self.master = master
        self.bus = EventBus(master)
        self.state = AppState()
        self.runner = PipelineRunner(self.bus, master) if PipelineRunner else None

        self._build_ui()
        self._wire_bus()

    def _attach_menubar(self, menubar: tk.Menu) -> None:
        top = self.winfo_toplevel()
        if isinstance(top, (tk.Tk, tk.Toplevel)):
            try:
                top.configure(menu=menubar)
            except Exception:
                top["menu"] = menubar

    # ---------------------------- BUILD UI -----------------------------------
    def _build_ui(self) -> None:
        # menu
        menubar = tk.Menu(self)
        # ... dodawanie File/Edit/etc ...
        self._attach_menubar(menubar)
        m_file = tk.Menu(menubar, tearoff=False)
        m_file.add_command(label="Open Image…", command=self._open_image)
        m_file.add_separator()
        m_file.add_command(label="Exit", command=self.master.destroy)
        menubar.add_cascade(label="File", menu=m_file)
        self.master.configure(menu=menubar)

        # main split
        main = ttk.Panedwindow(self, orient="horizontal")
        self.pack(fill="both", expand=True)

        left = ttk.Frame(main)
        right = ttk.Notebook(main)
        main.add(left, weight=3)
        main.add(right, weight=2)
        main.pack(fill="both", expand=True)

        # left: preview
        self.viewer = ImageCanvas(left)
        self.viewer.pack(fill="both", expand=True)

        # right: tabs
        self.tab_general = GeneralTab(
            right, ctx_ref=self.state,
            cfg=GeneralTabConfig(preview_size=220),
        )
        self.tab_filter = TabFilter(
            right, bus=self.bus,
            cfg=FilterTabConfig(allow_apply=True),
            ctx_ref=self.state,
        )
        self.tab_preset = PresetsTab(right, bus=self.bus)
        # Diagnostics tab
        self.tab_diag = DiagConsole(right)
        self.tab_diag.attach_bus(self.bus)
        right.add(self.tab_diag, text="Diagnostics")

        right.add(self.tab_general, text="General")
        right.add(self.tab_filter, text="Filters")
        right.add(self.tab_preset, text="Presets")

        # bottom HUD
        hud_frame = ttk.Frame(self)
        hud_frame.pack(fill="x")
        self.hud = Hud(hud_frame)
        try:
            self.hud.pack(fill="both", expand=True)
        except Exception:
            pass

        # docking: podaj prawdziwy root (Tk/Toplevel), nie Misc
        root_like = self.winfo_toplevel()
        try:
            self.dock = DockManager(root_like, {"hud": hud_frame, "right": right})
        except Exception:
            self.dock = DockManager()

        # status
        self.status = tk.StringVar(value="Ready")
        ttk.Label(self, textvariable=self.status, anchor="w").pack(fill="x")

        # preset service
        self.svc_presets = PresetService(self.master, self.bus)

    # ---------------------------- BUS WIRING ----------------------------------
    def _wire_bus(self) -> None:
        B = self.bus

        # uruchom pojedynczy krok (Filters tab)
        def _on_apply_filter(_t: str, payload: Dict[str, Any]) -> None:
            step = (payload or {}).get("step") or self.tab_filter.get_current_step()
            self.run_step(step)

        def _on_run_done(_t: str, data: Dict[str, Any]) -> None:
            out = (data or {}).get("output")
            ctx = (data or {}).get("ctx")
            if ctx is not None:
                try:
                    self.state.cache = dict(getattr(ctx, "cache", {}) or {})
                except Exception:
                    pass
                self._propagate_ctx(ctx)
                self._update_hud_from_cache(self.state.cache)
            self._show_output_array(out)
            self.status.set("Done")

        def _on_run_error(_t: str, data: Dict[str, Any]) -> None:
            messagebox.showerror("Run", f"{(data or {}).get('error')}")

        # presety
        def _on_preset_loaded(_t: str, d: Dict[str, Any]) -> None:
            self.state.preset_text = str((d or {}).get("text") or "")
            self.status.set("Preset loaded")

        def _on_preset_parsed(_t: str, d: Dict[str, Any]) -> None:
            cfg = (d or {}).get("cfg") or {}
            self.state.preset_cfg = cfg
            self.status.set("Preset OK")

        # subskrypcje
        try:
            B.subscribe("ui.run.apply_filter", _on_apply_filter)
        except Exception:
            pass
        try:
            B.subscribe("run.done", _on_run_done)
        except Exception:
            pass
        try:
            B.subscribe("run.error", _on_run_error)
        except Exception:
            pass
        try:
            B.subscribe("preset.loaded", _on_preset_loaded)
        except Exception:
            pass
        try:
            B.subscribe("preset.parsed", _on_preset_parsed)
        except Exception:
            pass

    # ---------------------------- ACTIONS -------------------------------------
    def _open_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Open image",
            filetypes=[
                ("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.webp;*.tif;*.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        if Image is None:
            messagebox.showerror("Pillow", "Pillow is required.")
            return
        try:
            img = Image.open(path)
            img = ImageOps.exif_transpose(img)
            if img.mode != "RGB":
                img = img.convert("RGB")
            self.set_image(img)
            self.status.set(f"Loaded: {path}")
        except Exception as ex:
            messagebox.showerror("Open image", str(ex))

    def set_image(self, pil_img: "Image.Image") -> None:
        self.state.image = pil_img
        try:
            self.viewer.set_image(pil_img)
        except Exception:
            pass

    # ---------------------------- RUN PIPELINE --------------------------------
    def _pil_to_u8_rgb(self, img: "Image.Image"):
        if np is None:
            raise RuntimeError("NumPy is required.")
        arr = np.array(img, dtype=np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.shape[-1] == 4:
            rgb = arr[..., :3]
            a = arr[..., 3:4].astype(np.float32) / 255.0
            arr = (rgb.astype(np.float32) * a + 0.0).clip(0, 255).astype(np.uint8)
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8, copy=False)
        return arr

    def run_step(self, step: Dict[str, Any]) -> None:
        """Uruchamia pojedynczy krok filtra."""
        if self.state.image is None:
            messagebox.showwarning("Run", "No image loaded.")
            return
        if build_ctx is None or apply_pipeline is None:
            messagebox.showwarning("Core", "core.pipeline is not available.")
            return
        try:
            img_u8 = self._pil_to_u8_rgb(self.state.image)
        except Exception as e:
            messagebox.showerror("Image", f"Convert failed: {e}")
            return

        # zbuduj kontekst wg presetu (lub domyślny)
        cfg = self.state.preset_cfg or {"version": 2, "steps": []}
        try:
            ctx = build_ctx(img_u8, seed=self.state.seed, cfg=cfg)
        except Exception as e:
            messagebox.showerror("Ctx", f"build_ctx failed: {e}")
            return

        # dolej maski użytkownika, jeśli są
        try:
            if isinstance(self.state.masks, dict) and self.state.masks:
                ctx.masks.update(self.state.masks)
        except Exception:
            pass

        steps = [dict(step or {})]
        self.status.set("Running…")

        # 1) async przez PipelineRunner jeśli dostępny
        if self.runner is not None:
            try:
                self.runner.run(img_u8, ctx, steps)
                return
            except Exception as e:
                messagebox.showerror("Runner", str(e))
                return

        # 2) fallback: wykonaj synchronicznie i opublikuj run.done samemu
        try:
            out = apply_pipeline(img_u8, ctx, steps, fail_fast=True, metrics=True)
            # publikuj „ręcznie” żeby reszta ścieżki UI była wspólna
            self.bus.publish("run.done", {"output": out, "ctx": ctx})
        except Exception as e:
            self.bus.publish("run.error", {"error": str(e)})

    # ---------------------------- UI HELPERS ----------------------------------
    def _show_output_array(self, out: Any) -> None:
        if out is None or Image is None or np is None:
            return
        try:
            a = np.asarray(out)
            if a.ndim == 2:
                a = np.stack([a, a, a], axis=-1)
            if a.dtype != np.uint8:
                a = np.clip(a, 0, 255).astype(np.uint8)
            pil = Image.fromarray(a, mode="RGB")
            self.viewer.set_image(pil)
        except Exception:
            pass

    def _update_hud_from_cache(self, cache: Dict[str, Any]) -> None:
        try:
            if hasattr(self.hud, "set_cache") and callable(getattr(self.hud, "set_cache")):
                self.hud.set_cache(cache)
            elif hasattr(self.hud, "render_from_cache") and callable(getattr(self.hud, "render_from_cache")):
                self.hud.render_from_cache(cache)
        except Exception:
            pass

    def _propagate_ctx(self, ctx_obj: Any) -> None:
        """Przekaż nowy ctx do tabs (jeśli mają set_ctx)."""
        for tab in (self.tab_general, self.tab_filter):
            try:
                if hasattr(tab, "set_ctx") and callable(getattr(tab, "set_ctx")):
                    tab.set_ctx(ctx_obj)
            except Exception:
                pass


# ----------------------------- ENTRYPOINT ------------------------------------
def main() -> None:
    root = tk.Tk()
    root.title("GlitchLab GUI v4")
    root.geometry("1200x800")
    app = App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
