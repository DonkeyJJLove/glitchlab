# glitchlab/gui/app.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Any, Dict, Optional, List

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

# >>> LAYERS <<<
try:
    from glitchlab.gui.services.layer_manager import LayerManager
    from glitchlab.gui.services.compositor import BlendMode
except Exception:
    LayerManager = None  # type: ignore
    BlendMode = str  # type: ignore


class AppState:
    def __init__(self) -> None:
        # single-image fallback (legacy)
        self.image: Optional["Image.Image"] = None

        # layers state (zarządzane przez LayerManager)
        self.layers: List[Any] = []               # list[Layer]
        self.active_layer_id: Optional[str] = None

        self.masks: Dict[str, Any] = {}
        self.cache: Dict[str, Any] = {}
        self.preset_cfg: Optional[Dict[str, Any]] = None
        self.seed: int = 7


class App(tk.Frame):
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

        # LAYERS service
        self.layer_mgr: Optional[LayerManager] = None

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

        # (1) Wąski lewy dock – placeholder
        self.left_dummy = LeftDummy(self.main)
        self.main.add(self.left_dummy, weight=0)

        # (2) Viewer (CanvasContainer ma toolbox + rulers + LayerCanvas)
        self.center = ttk.Frame(self.main)
        self.viewer = CanvasContainer(self.center, bus=self.bus, tool_var=self.tool_var)
        self.viewer.pack(fill="both", expand=True)
        self.main.add(self.center, weight=65)

        # (3) Notebook
        self.right = ttk.Notebook(self.main)
        self.tab_general = GeneralTab(self.right, ctx_ref=self.state, cfg=GeneralTabConfig(preview_size=320), bus=self.bus)
        self.tab_filter = TabFilter(self.right, bus=self.bus, ctx_ref=self.state, cfg=FilterTabConfig(allow_apply=True))
        self.tab_preset = PresetsTab(self.right, bus=self.bus)
        self.right.add(self.tab_general, text="General")
        self.right.add(self.tab_filter, text="Filters")
        self.right.add(self.tab_preset, text="Presets")
        self.main.add(self.right, weight=35)

        # sash init
        def _init_sash():
            w = self.main.winfo_width()
            if w < 100:
                self.after(30, _init_sash)
                return
            try:
                dummy_w = self.left_dummy.winfo_width() or 28
                self.main.sashpos(1, int(dummy_w + (w - dummy_w) * 0.65))
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
        B.subscribe("ui.view.toggle_left", lambda *_: self._toggle_left())
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

        # >>> Layers wiring <<<
        B.subscribe("ui.layer.add", lambda _t, d: self._on_layer_add(d))
        B.subscribe("ui.layer.remove", lambda _t, d: self._on_layer_remove(d))
        B.subscribe("ui.layer.set_active", lambda _t, d: self._on_layer_set_active(d))
        B.subscribe("ui.layer.update", lambda _t, d: self._on_layer_update(d))
        B.subscribe("ui.layers.reorder", lambda _t, d: self._on_layers_reorder(d))
        # „pull” z panelu Layers → odeślij snapshot bieżącego stanu
        B.subscribe("ui.layers.pull", lambda *_: self._publish_layers_snapshot())
        # Reakcja na wewnętrzne publikacje LayerManagera (przeliczenie podglądu)
        B.subscribe("ui.layers.changed", lambda *_: self._refresh_composite())

        # Move Layer – trwałe zapisanie przesunięcia
        B.subscribe("ui.layer.commit_offset", lambda _t, d: self._on_layer_commit_offset(d))

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
            # layers bootstrap
            if LayerManager and self.layer_mgr is None:
                self.layer_mgr = LayerManager(self.state, self.bus.publish)
            if self.layer_mgr:
                self.state.layers.clear()
                self.state.active_layer_id = None
                self.layer_mgr.add_layer(img, name="Background", blend="normal", opacity=1.0, visible=True)  # type: ignore[arg-type]
                self._layers_set_snapshot()
                # composite → viewer
                self._refresh_composite()
            else:
                # fallback single image
                self.state.image = img
                self.viewer.set_image(img)

            if self.history is not None:
                try:
                    base = self._pil_to_u8(img)
                    self.history.reset(base, cache={}, label="source")
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
        # zapisujemy kompozyt
        comp = self._get_composite_np()
        if comp is None:
            messagebox.showwarning("Save Image", "No image to save.")
            return
        try:
            if Image is None or np is None:
                messagebox.showerror("Save", "Pillow/NumPy required.")
                return
            Image.fromarray(comp, "RGB").save(path)
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
        # Źródłem dla filtra jest obraz AKTYWNEJ WARSTWY (jeśli mamy layer_mgr),
        # w przeciwnym razie single-image fallback.
        img_input_pil = self._get_active_layer_pil() or self.state.image
        if not img_input_pil:
            messagebox.showwarning("Run", "No image loaded.")
            return
        if not (build_ctx and apply_pipeline):
            messagebox.showwarning("Core", "core.pipeline unavailable.")
            return

        img_u8 = self._pil_to_u8(img_input_pil)
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
        img_input_pil = self._get_active_layer_pil() or self.state.image
        if not img_input_pil:
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

        img_u8 = self._pil_to_u8(img_input_pil)
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

        # Aktualizacja aktywnej warstwy lub fallback single-image
        updated_pil = self._array_to_pil(out)
        if updated_pil is not None:
            if self.layer_mgr and self.state.active_layer_id:
                try:
                    # zamień obraz aktywnej warstwy
                    self.layer_mgr.update_layer(self.state.active_layer_id, image=self._pil_to_u8(updated_pil))
                except Exception:
                    # awaryjnie: single-image fallback
                    self.state.image = updated_pil
                self._layers_set_snapshot()
                self._refresh_composite()
            else:
                self.state.image = updated_pil
                try:
                    self.viewer.set_image(updated_pil)
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
            if self.layer_mgr and self.state.active_layer_id:
                try:
                    self.layer_mgr.update_layer(self.state.active_layer_id, image=self._pil_to_u8(pil))
                    self._layers_set_snapshot()
                    self._refresh_composite()
                except Exception:
                    self.state.image = pil
                    self.viewer.set_image(pil)
            else:
                self.state.image = pil
                self.viewer.set_image(pil)
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
            if self.layer_mgr and self.state.active_layer_id:
                try:
                    self.layer_mgr.update_layer(self.state.active_layer_id, image=self._pil_to_u8(pil))
                    self._layers_set_snapshot()
                    self._refresh_composite()
                except Exception:
                    self.state.image = pil
                    self.viewer.set_image(pil)
            else:
                self.state.image = pil
                self.viewer.set_image(pil)
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
                if str(self.left_dummy) in panes:
                    self.main.forget(self.left_dummy)
                self._leftdock_visible = False
            else:
                if str(self.left_dummy) not in panes:
                    self.main.insert(0, self.left_dummy)
                self._leftdock_visible = True
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

    # ───────────── Layers helpers ─────────────
    def _get_active_layer_pil(self) -> Optional["Image.Image"]:
        """Zwraca obraz aktywnej warstwy (PIL) jeśli dostępny."""
        if not (self.layer_mgr and self.state.active_layer_id):
            return None
        try:
            lid = self.state.active_layer_id
            for l in self.state.layers:
                if getattr(l, "id", None) == lid:
                    img = getattr(l, "image", None)
                    if img is None:
                        return None
                    if Image is None or np is None:
                        return None
                    if isinstance(img, Image.Image):
                        return img.convert("RGB")
                    a = np.asarray(img)
                    if a.dtype != np.uint8:
                        a = np.clip(a, 0, 255).astype(np.uint8)
                    return Image.fromarray(a, "RGB")
        except Exception:
            return None
        return None

    def _get_composite_np(self) -> Optional["np.ndarray"]:
        """Zwraca skomponowany podgląd sceny (RGB uint8) lub None."""
        if self.layer_mgr:
            try:
                comp = self.layer_mgr.get_composite_for_viewport()
                if comp is not None:
                    return comp
            except Exception:
                pass
        # fallback
        if self.state.image is not None and np is not None:
            try:
                return self._pil_to_u8(self.state.image)
            except Exception:
                return None
        return None

    def _refresh_composite(self) -> None:
        """Przelicza kompozyt i aktualizuje viewer."""
        comp = self._get_composite_np()
        if comp is None:
            return
        pil = self._array_to_pil(comp)
        if pil is None:
            return
        try:
            self.viewer.set_image(pil)
        except Exception:
            pass

    # ───────────── Layers BUS handlers ─────────────
    def _on_layer_add(self, d: Dict[str, Any]) -> None:
        if not LayerManager:
            messagebox.showwarning("Layers", "LayerManager unavailable.")
            return
        if self.layer_mgr is None:
            self.layer_mgr = LayerManager(self.state, self.bus.publish)
        # źródło: path | pil | np | duplicate_active
        src_path = d.get("path")
        name = d.get("name") or "Layer"
        blend = d.get("blend") or "normal"
        opacity = float(d.get("opacity", 1.0))
        visible = bool(d.get("visible", True))
        try:
            if src_path and Image:
                pil = Image.open(src_path).convert("RGB")
                self.layer_mgr.add_layer(pil, name=name, blend=blend, opacity=opacity, visible=visible)
            elif d.get("duplicate_active"):
                pil = self._get_active_layer_pil() or self.state.image
                if pil is not None:
                    self.layer_mgr.add_layer(pil, name=name, blend=blend, opacity=opacity, visible=visible)
            else:
                # pusty layer (czarny)
                base = self._get_active_layer_pil() or self.state.image
                if base is None or Image is None:
                    return
                w, h = base.size
                self.layer_mgr.add_layer(Image.new("RGB", (w, h), (0, 0, 0)), name=name,
                                         blend=blend, opacity=opacity, visible=visible)
        finally:
            self._layers_set_snapshot()
            self._refresh_composite()

    def _on_layer_remove(self, d: Dict[str, Any]) -> None:
        lid = d.get("id")
        if not (lid and self.layer_mgr):
            return
        try:
            self.layer_mgr.remove_layer(str(lid))
        finally:
            self._layers_set_snapshot()
            self._refresh_composite()

    def _on_layer_set_active(self, d: Dict[str, Any]) -> None:
        lid = d.get("id")
        if not lid:
            return
        self.state.active_layer_id = str(lid)
        self._layers_set_snapshot()
        self._refresh_composite()

    def _on_layer_update(self, d: Dict[str, Any]) -> None:
        """Patch layer fields: visible/opacity/blend/mask/image."""
        lid = d.get("id")
        if not (lid and self.layer_mgr):
            return
        patch = {}
        for k in ("visible", "opacity", "blend", "mask", "name", "image"):
            if k in d:
                patch[k] = d[k]
        if patch:
            try:
                self.layer_mgr.update_layer(str(lid), **patch)
            finally:
                self._layers_set_snapshot()
                self._refresh_composite()

    def _on_layers_reorder(self, d: Dict[str, Any]) -> None:
        """Reorder wymaga prostego przepisania listy w state.layers."""
        order = d.get("order")  # list of layer ids top→bottom
        if not (order and isinstance(order, list)):
            return
        try:
            id_to_layer = {getattr(l, "id", None): l for l in self.state.layers}
            new_list = [id_to_layer.get(i) for i in order if i in id_to_layer]
            self.state.layers = [l for l in new_list if l is not None]
            self.bus.publish("ui.layers.changed", {})
        except Exception:
            pass
        finally:
            self._layers_set_snapshot()
            self._refresh_composite()

    # ───────────── Move Layer: commit przesunięcia ─────────────
    def _on_layer_commit_offset(self, d: Dict[str, Any]) -> None:
        """Trwale przesuwa piksele aktywnej warstwy o (dx, dy)."""
        if not (self.layer_mgr and self.state.active_layer_id and np is not None):
            return
        dx = int(d.get("dx", 0) or 0)
        dy = int(d.get("dy", 0) or 0)
        if dx == 0 and dy == 0:
            return

        # znajdź aktywną warstwę
        lid = self.state.active_layer_id
        target = None
        for l in self.state.layers:
            if getattr(l, "id", None) == lid:
                target = l
                break
        if target is None:
            return

        # pobierz obraz jako uint8 RGB ndarray
        src = getattr(target, "image", None)
        if src is None:
            return
        try:
            if Image is not None and isinstance(src, Image.Image):
                src_u8 = self._pil_to_u8(src)
            else:
                src_u8 = np.asarray(src)
                if src_u8.ndim == 2:
                    src_u8 = np.stack([src_u8, src_u8, src_u8], axis=-1)
                if src_u8.dtype != np.uint8:
                    src_u8 = np.clip(src_u8, 0, 255).astype(np.uint8)
        except Exception:
            return

        H, W, C = src_u8.shape
        if abs(dx) >= W or abs(dy) >= H:
            dst = np.zeros_like(src_u8)
        else:
            dst = np.zeros_like(src_u8)
            # oblicz okna kopiowania
            if dx >= 0:
                sx0, sx1 = 0, W - dx
                dx0, dx1 = dx, W
            else:
                sx0, sx1 = -dx, W
                dx0, dx1 = 0, W + dx

            if dy >= 0:
                sy0, sy1 = 0, H - dy
                dy0, dy1 = dy, H
            else:
                sy0, sy1 = -dy, H
                dy0, dy1 = 0, H + dy

            if sx1 > sx0 and sy1 > sy0 and dx1 > dx0 and dy1 > dy0:
                dst[dy0:dy1, dx0:dx1, :] = src_u8[sy0:sy1, sx0:sx1, :]

        # aktualizuj warstwę
        try:
            self.layer_mgr.update_layer(lid, image=dst)
        finally:
            # zaktualizuj snapshot + odśwież podgląd
            self._layers_set_snapshot()
            self._refresh_composite()
            # (CanvasContainer wyczyści overlay po commit przez własną subskrypcję)

    # ───────────── Layers snapshots (dla panelu Layers) ─────────────
    def _layers_snapshot(self) -> Dict[str, Any]:
        """Zwraca {layers:[...], active:<id>} do panelu Layers."""
        layers_desc: List[Dict[str, Any]] = []
        for l in self.state.layers:
            try:
                layers_desc.append({
                    "id": getattr(l, "id", ""),
                    "name": getattr(l, "name", "Layer"),
                    "visible": bool(getattr(l, "visible", True)),
                    "opacity": float(getattr(l, "opacity", 1.0)),
                    "blend": str(getattr(l, "blend", "normal")),
                })
            except Exception:
                continue
        return {"layers": layers_desc, "active": self.state.active_layer_id}

    def _layers_set_snapshot(self) -> None:
        """Aktualizuje cache snapshotu u busa (best-effort; bez emisji)."""
        try:
            setattr(self.bus, "last_layers_snapshot", self._layers_snapshot())
        except Exception:
            pass

    def _publish_layers_snapshot(self) -> None:
        """Publikuje 'ui.layers.changed' z pełnym snapshotem (na żądanie ui.layers.pull)."""
        snap = self._layers_snapshot()
        try:
            setattr(self.bus, "last_layers_snapshot", snap)
        except Exception:
            pass
        try:
            self.bus.publish("ui.layers.changed", snap)
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
