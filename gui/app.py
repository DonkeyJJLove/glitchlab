# glitchlab/gui/app.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Any, Dict, Optional

from glitchlab.gui.views.bottom_area import init_styles

if __package__ in (None, ""):
    PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if PROJ not in sys.path:
        sys.path.insert(0, PROJ)

# third-party
try:
    from PIL import Image, ImageOps  # pillow
except Exception:
    Image = ImageOps = None  # type: ignore
try:
    import numpy as np  # numpy
except Exception:
    np = None  # type: ignore

# core
try:
    from glitchlab.core.pipeline import build_ctx, apply_pipeline
except Exception:
    build_ctx = apply_pipeline = None  # type: ignore

# ensure filters
try:
    import glitchlab.filters  # noqa: F401
except Exception:
    pass

# GUI
from glitchlab.gui.views.tab_general import GeneralTab, GeneralTabConfig
from glitchlab.gui.views.tab_filter import TabFilter, FilterTabConfig
from glitchlab.gui.views.tab_preset import PresetsTab
from glitchlab.gui.widgets.canvas_container import CanvasContainer
from glitchlab.gui.views.menu import MenuBar
from glitchlab.gui.views.left_dummy import LeftDummy  # NEW: wąski lewy dock „in project”

# services / fallbacks
try:
    from glitchlab.gui.event_bus import EventBus
except Exception:  # minimalny stub
    class EventBus:
        def __init__(self, *_a, **_k): ...
        def publish(self, t, p): print(t, p)
        def subscribe(self, *_a, **_k): ...

try:
    from glitchlab.gui.services.pipeline_runner import PipelineRunner
except Exception:
    PipelineRunner = None  # type: ignore

try:
    from glitchlab.gui.views.bottom_panel import BottomPanel
except Exception:
    BottomPanel = None  # type: ignore

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
    PresetServiceConfig = object  # type: ignore

try:
    from glitchlab.gui.views.bottom_area import BottomArea
except Exception:
    BottomArea = None  # type: ignore

try:
    from glitchlab.gui.services.image_history import ImageHistory
except Exception:
    ImageHistory = None  # type: ignore


class AppState:
    def __init__(self) -> None:
        self.image: Optional["Image.Image"] = None
        self.masks: Dict[str, Any] = {}
        self.cache: Dict[str, Any] = {}
        self.preset_cfg: Optional[Dict[str, Any]] = None
        self.seed: int = 7


class App(tk.Frame):
    """LeftDummy ▸ Viewer (CanvasContainer) ▸ Notebook ▸ BottomArea (+ historia obrazu)."""

    def __init__(self, master: tk.Tk, **kw: Any) -> None:
        super().__init__(master, **kw)

        # flagi widoków
        self._leftdock_visible = True
        self._right_visible = True
        self._hud_visible = True
        self._fullscreen = False

        self.master = master
        self.bus = EventBus(master)
        self.state = AppState()
        self.runner = PipelineRunner(self.bus, master) if PipelineRunner else None
        self.history = ImageHistory(bus=self.bus, max_len=50) if ImageHistory else None

        # referencje
        self.menubar: Optional[MenuBar] = None
        self.bottom: Optional[Any] = None
        self.status: Optional[Any] = None

        # tryb narzędzia dla viewer’a (zsynchronizowany z menu Tools)
        self.tool_var = tk.StringVar(value="pan")

        # UI + BUS
        self._build_ui()
        self._wire_bus()

    # ────────────── UI ──────────────
    def _build_ui(self) -> None:
        self.pack(fill="both", expand=True)

        # MENUBAR
        self.menubar = MenuBar(self.master, bus=self.bus)

        # GŁÓWNY PODZIAŁ: [LeftDummy] | [Viewer] | [RightNotebook]
        self.main = ttk.Panedwindow(self, orient="horizontal")
        self.main.pack(fill="both", expand=True)

        # (1) Wąski lewy dock – tylko placeholder „in project”
        self.left_dummy = LeftDummy(self.main)
        self.main.add(self.left_dummy, weight=0)  # stała, wąska kolumna

        # (2) ŚRODEK: Viewer (CanvasContainer ma własny, wewnętrzny toolbox – ciemny)
        self.center = ttk.Frame(self.main)
        self.viewer = CanvasContainer(self.center, bus=self.bus, tool_var=self.tool_var)
        self.viewer.pack(fill="both", expand=True)
        self.main.add(self.center, weight=65)

        # (3) PRAWO: Notebook
        self.right = ttk.Notebook(self.main)
        self.tab_general = GeneralTab(self.right, ctx_ref=self.state, cfg=GeneralTabConfig(preview_size=320), bus=self.bus)
        self.tab_filter = TabFilter(self.right, bus=self.bus, ctx_ref=self.state, cfg=FilterTabConfig(allow_apply=True))
        self.tab_preset = PresetsTab(self.right, bus=self.bus)
        self.right.add(self.tab_general, text="General")
        self.right.add(self.tab_filter, text="Filters")
        self.right.add(self.tab_preset, text="Presets")
        self.main.add(self.right, weight=35)

        # sash: 65% na viewer, 35% na prawe zakładki
        def _init_sash():
            w = self.main.winfo_width()
            if w < 100:
                self.after(30, _init_sash)
                return
            # sash #0 to granica między left_dummy a center – zostawiamy przy lewym brzegu
            # ustaw pozycję #1 na 65% szerokości (po dummy)
            try:
                total = w
                # pozycja sasha #1: szerokość dummy + 65% (reszty)
                dummy_w = self.left_dummy.winfo_width() or 28
                self.main.sashpos(1, int(dummy_w + (total - dummy_w) * 0.65))
            except Exception:
                pass
        self.after_idle(_init_sash)

        # BOTTOM (panel + status)
        if BottomArea:
            self.bottom_area = BottomArea(self, bus=self.bus, default="hud")
            self.bottom_area.pack(fill="x", side="bottom", anchor="s")
            self.bottom = self.bottom_area.panel
            self.status = self.bottom_area.status
        else:
            self.bottom = None
            self.status = None

        # docking (best-effort)
        try:
            self.dock = DockManager(self.winfo_toplevel(),
                                    {"bottom": self.bottom or ttk.Frame(self), "right": self.right})
        except Exception:
            self.dock = DockManager()  # type: ignore

        # preset service
        try:
            self.svc_presets = PresetService(self.master, self.bus, PresetServiceConfig())
        except Exception:
            self.svc_presets = PresetService(self.master, self.bus)  # type: ignore[arg-type]

        # initial Edit menu state
        self._update_edit_menu_state()

    # ───────────── BUS wiring ──────────────
    def _wire_bus(self) -> None:
        B = self.bus

        # View toggles
        B.subscribe("ui.view.toggle_left", lambda *_: self._toggle_left())   # teraz przełącza LeftDummy
        B.subscribe("ui.view.toggle_right", lambda *_: self._toggle_right())
        B.subscribe("ui.view.toggle_hud", lambda *_: self._toggle_hud())
        B.subscribe("ui.view.fullscreen", lambda *_: self._toggle_fullscreen())

        # File
        B.subscribe("ui.files.open", lambda _t, d: self._load_image_path((d or {}).get("path")))
        B.subscribe("ui.files.save", lambda _t, d: self._save_image_path((d or {}).get("path")))
        B.subscribe("ui.app.quit", lambda *_: self.master.destroy())

        # Presets (menu → PresetService)
        B.subscribe("ui.preset.open", lambda *_t, _d: self.bus.publish("ui.preset.open", {}))
        B.subscribe("ui.preset.save", lambda *_t, _d: self.bus.publish("ui.preset.save", {"cfg": self.state.preset_cfg or {}}))

        # Tools (menu → viewer/toolbox)
        B.subscribe("ui.tools.select", lambda _t, d: self.tool_var.set((d or {}).get("name", "pan")))

        # Run
        B.subscribe("ui.run.apply_filter",
                    lambda _t, p: self.run_step((p or {}).get("step") or self.tab_filter.get_current_step()))
        B.subscribe("ui.run.apply_preset",
                    lambda _t, p: self.run_preset((p or {}).get("cfg")))

        # Edit
        B.subscribe("ui.edit.undo", lambda *_: self._on_undo())
        B.subscribe("ui.edit.redo", lambda *_: self._on_redo())

        # Progress / Done / Error
        B.subscribe("run.progress", lambda _t, d: self._on_progress(d))
        B.subscribe("run.done", self._on_run_done)
        B.subscribe("run.error", self._on_run_error)

        # StatusBar (opcjonalnie)
        if self.status is not None:
            B.subscribe("ui.status.set", lambda _t, d: self.status.set_text(d.get("text", "")))
            B.subscribe("ui.cursor.pos",
                        lambda _t, d: self.status.set_right(f"x={d.get('x', '-')}, y={d.get('y', '-')}" if d else ""))

        # Historia – zewnętrzne sygnały
        B.subscribe("history.changed", lambda *_: self._update_edit_menu_state())

        # preset_cfg – śledź
        B.subscribe("preset.parsed", lambda _t, d: self._on_preset_parsed(d))

    def _on_preset_parsed(self, d: Dict[str, Any]) -> None:
        try:
            cfg = d.get("cfg") or {}
            if isinstance(cfg, dict):
                self.state.preset_cfg = dict(cfg)
        except Exception:
            pass

    # ───────────── File helpers ─────────────
    def _load_image_path(self, path: Optional[str]) -> None:
        if not path:
            return
        if Image is None:
            messagebox.showerror("Pillow", "Pillow is required.")
            return
        try:
            img = ImageOps.exif_transpose(Image.open(path))  # type: ignore[arg-type]
            img = img.convert("RGB") if img.mode != "RGB" else img
            self.state.image = img
            self.viewer.set_image(img)

            if self.history is not None:
                try:
                    self.history.reset(img, cache={}, label="source")
                except Exception:
                    pass
            self._update_edit_menu_state()

            if self.status is not None:
                self.status.set_text(f"Loaded: {os.path.basename(path)}")

            try:
                self.bus.publish("ui.image.loaded", {"path": path})
            except Exception:
                pass
        except Exception as ex:
            messagebox.showerror("Open image", str(ex))

    def _save_image_path(self, path: Optional[str]) -> None:
        if not path:
            return
        if self.state.image is None:
            messagebox.showwarning("Save Image", "No image to save.")
            return
        try:
            self.state.image.save(path)
            if self.status is not None:
                self.status.set_text(f"Saved: {os.path.basename(path)}")
        except Exception as ex:
            messagebox.showerror("Save Image", str(ex))

    # ───────────── helpers ─────────────
    @staticmethod
    def _pil_to_u8(img: "Image.Image"):
        if np is None:
            raise RuntimeError("NumPy required")
        a = np.asarray(img, dtype=np.uint8)  # type: ignore[arg-type]
        if a.ndim == 2:
            a = np.stack([a] * 3, axis=-1)
        if a.ndim == 3 and a.shape[-1] == 4:
            rgb = a[..., :3].astype(np.float32)
            alpha = a[..., 3:4].astype(np.float32) / 255.0
            a = (rgb * alpha).clip(0, 255).astype(np.uint8)
        return a

    @staticmethod
    def _array_to_pil(arr: Any) -> Optional["Image.Image"]:
        if Image is None or np is None:
            return None
        try:
            a = np.asarray(arr)
            if a.ndim == 2:
                a = np.stack([a, a, a], axis=-1)
            if a.dtype != np.uint8:
                a = np.clip(a, 0, 255).astype(np.uint8)
            return Image.fromarray(a, mode="RGB")
        except Exception:
            return None

    # ───────────── RUN: single step ─────────────
    def run_step(self, step: Dict[str, Any]) -> None:
        if not self.state.image:
            messagebox.showwarning("Run", "No image loaded.")
            return
        if not (build_ctx and apply_pipeline):
            messagebox.showwarning("Core", "core.pipeline unavailable.")
            return

        img_u8 = self._pil_to_u8(self.state.image)
        ctx = build_ctx(img_u8, seed=self.state.seed,
                        cfg=self.state.preset_cfg or {"version": 2, "steps": []})  # type: ignore[arg-type]
        if self.state.masks:
            try:
                ctx.masks.update(self.state.masks)  # type: ignore[attr-defined]
            except Exception:
                pass

        self.bus.publish("run.start", {"mode": "single"})
        self.bus.publish("ui.status.set", {"text": "Applying filter…"})
        self.bus.publish("run.progress", {"value": 0.0, "text": "Applying filter…"})

        if self.runner:
            try:
                self.runner.run(img_u8, ctx, [dict(step)])  # type: ignore[arg-type]
                return
            except Exception as e:
                messagebox.showerror("Runner", str(e))
                return

        try:
            out = apply_pipeline(img_u8, ctx, [dict(step)], fail_fast=True, metrics=True)  # type: ignore[arg-type]
            self.bus.publish("run.done", {"output": out, "ctx": ctx})
        except Exception as e:
            self.bus.publish("run.error", {"error": str(e)})

    # ───────────── RUN: preset ─────────────
    def run_preset(self, cfg: Optional[Dict[str, Any]] = None) -> None:
        if not self.state.image:
            messagebox.showwarning("Run", "No image loaded.")
            return
        if not (build_ctx and apply_pipeline):
            messagebox.showwarning("Core", "core.pipeline unavailable.")
            return

        effective_cfg = None
        if isinstance(cfg, dict):
            effective_cfg = dict(cfg)
        elif self.state.preset_cfg:
            effective_cfg = dict(self.state.preset_cfg)
        else:
            try:
                effective_cfg = self.tab_preset.get_cfg()
            except Exception:
                effective_cfg = {"version": 2, "steps": []}

        steps = list(effective_cfg.get("steps") or [])
        self.state.preset_cfg = effective_cfg

        img_u8 = self._pil_to_u8(self.state.image)
        ctx = build_ctx(img_u8, seed=self.state.seed, cfg=effective_cfg)  # type: ignore[arg-type]
        if self.state.masks:
            try:
                ctx.masks.update(self.state.masks)  # type: ignore[attr-defined]
            except Exception:
                pass

        self.bus.publish("run.start", {"mode": "preset"})
        self.bus.publish("ui.status.set", {"text": "Applying preset…"})
        self.bus.publish("run.progress", {"value": 0.0, "text": "Applying preset…"})

        if self.runner:
            try:
                self.runner.run(img_u8, ctx, steps)  # type: ignore[arg-type]
                return
            except Exception as e:
                messagebox.showerror("Runner", str(e))
                return

        try:
            out = apply_pipeline(img_u8, ctx, steps, fail_fast=True, metrics=True)  # type: ignore[arg-type]
            self.bus.publish("run.done", {"output": out, "ctx": ctx})
        except Exception as e:
            self.bus.publish("run.error", {"error": str(e)})

    # ───────────── BUS handlers ─────────────
    def _on_progress(self, d: Dict[str, Any]) -> None:
        val = float(d.get("value", 0.0) or 0.0)
        try:
            self.viewer.set_enabled(not (0.0 < val < 1.0))
        except Exception:
            pass
        try:
            if self.status is not None:
                self.status.set_progress(val, d.get("text", ""))
        except Exception:
            pass

    def _on_run_done(self, _t: str, data: Dict[str, Any]) -> None:
        out = (data or {}).get("output")
        ctx = (data or {}).get("ctx")

        if ctx is not None:
            try:
                self.state.cache = dict(getattr(ctx, "cache", {}) or {})
            except Exception:
                pass

        if self.history is not None and out is not None:
            try:
                self.history.snapshot_from_ctx(out, ctx, label="step")
            except Exception:
                pass

        pil = self._array_to_pil(out)
        if pil is not None:
            self.state.image = pil
            try:
                self.viewer.set_image(pil)
            except Exception:
                pass

        self._update_bottom_from_cache(self.state.cache)
        self._update_edit_menu_state()

        if self.status is not None:
            self.status.set_text("Done")
            try:
                self.status.set_progress(None, None)
            except Exception:
                pass

    def _on_run_error(self, _t: str, data: Dict[str, Any]) -> None:
        messagebox.showerror("Run", f"{(data or {}).get('error')}")
        try:
            self.viewer.set_enabled(True)
        except Exception:
            pass
        if self.status is not None:
            self.status.set_text("Error")
            try:
                self.status.set_progress(None, None)
            except Exception:
                pass

    # ───────────── Bottom helpers ─────────────
    def _update_bottom_from_cache(self, cache: Dict[str, Any]) -> None:
        try:
            if self.bottom is not None and hasattr(self.bottom, "set_ctx"):
                self.bottom.set_ctx(cache)
        except Exception:
            pass

    # ───────────── Undo/Redo ─────────────
    def _update_edit_menu_state(self) -> None:
        if not self.history or not self.menubar:
            return
        try:
            self.menubar.set_edit_enabled(self.history.can_undo(), self.history.can_redo())
        except Exception:
            pass

    def _on_undo(self) -> None:
        if not self.history or not self.history.can_undo():
            return
        img_u8 = self.history.undo()
        if img_u8 is None:
            return
        pil = self._array_to_pil(img_u8)
        if pil is not None:
            self.state.image = pil
            try:
                self.viewer.set_image(pil)
            except Exception:
                pass
        try:
            cache = self.history.get_current_cache()
            self.state.cache = cache
            self._update_bottom_from_cache(cache)
        except Exception:
            pass
        self._update_edit_menu_state()
        if self.status is not None:
            self.status.set_text("Undo")

    def _on_redo(self) -> None:
        if not self.history or not self.history.can_redo():
            return
        img_u8 = self.history.redo()
        if img_u8 is None:
            return
        pil = self._array_to_pil(img_u8)
        if pil is not None:
            self.state.image = pil
            try:
                self.viewer.set_image(pil)
            except Exception:
                pass
        try:
            cache = self.history.get_current_cache()
            self.state.cache = cache
            self._update_bottom_from_cache(cache)
        except Exception:
            pass
        self._update_edit_menu_state()
        if self.status is not None:
            self.status.set_text("Redo")

    # ───────────── View helpers ─────────────
    def _toggle_left(self) -> None:
        """Przełącz wąski lewy dock (dummy). Viewer i prawa strona są nienaruszone."""
        try:
            panes = self.main.panes()
            if self._leftdock_visible:
                # schowaj
                if str(self.left_dummy) in panes:
                    self.main.forget(self.left_dummy)
                self._leftdock_visible = False
            else:
                # pokaż z powrotem jako pierwszy pane
                if str(self.left_dummy) not in panes:
                    self.main.insert(0, self.left_dummy)
                self._leftdock_visible = True
            # po zmianie popraw sash dla #1 (granica viewer/right)
            w = self.main.winfo_width() or 1200
            dummy_w = self.left_dummy.winfo_width() if self._leftdock_visible else 0
            try:
                self.main.sashpos(1, int(dummy_w + max(0, w - dummy_w) * 0.65))
            except Exception:
                pass
        except Exception:
            pass

    def _toggle_right(self) -> None:
        self._right_visible = not self._right_visible
        try:
            panes = self.main.panes()
            if self._right_visible:
                if str(self.right) not in panes:
                    self.main.add(self.right, weight=35)
                    w = self.main.winfo_width() or 1200
                    dummy_w = self.left_dummy.winfo_width() if self._leftdock_visible else 0
                    self.main.sashpos(1, int(dummy_w + (w - dummy_w) * 0.65))
            else:
                if str(self.right) in panes:
                    self.main.forget(self.right)
        except Exception:
            pass

    def _toggle_hud(self) -> None:
        if not hasattr(self, "bottom_area") or self.bottom_area is None:
            return
        self._hud_visible = not self._hud_visible
        try:
            if hasattr(self.bottom_area, "panel") and self.bottom_area.panel is not None:
                if hasattr(self.bottom_area.panel, "set_visible"):
                    self.bottom_area.panel.set_visible(self._hud_visible)
                else:
                    if self._hud_visible:
                        self.bottom_area.panel.pack(fill="both", expand=True)
                    else:
                        self.bottom_area.panel.pack_forget()
        except Exception:
            pass

    def _toggle_fullscreen(self) -> None:
        self._fullscreen = not self._fullscreen
        try:
            self.master.attributes("-fullscreen", self._fullscreen)
        except Exception:
            try:
                if self._fullscreen:
                    self.master.state("zoomed")
                else:
                    self.master.state("normal")
            except Exception:
                pass


# ───────────── entrypoint ─────────────
def main() -> None:
    root = tk.Tk()
    init_styles(root)
    root.title("GlitchLab GUI v4.5")
    try:
        root.state("zoomed")
    except Exception:
        root.geometry("1280x800")
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
