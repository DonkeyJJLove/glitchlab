# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, time
from typing import Any, Dict, Optional
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
from PIL import Image

# Core imports with fallbacks
try:
    from glitchlab.core.registry import available as reg_available, meta as reg_meta
    from glitchlab.core.pipeline import build_ctx, apply_pipeline, normalize_preset
except Exception:  # pragma: no cover
    from core.registry import available as reg_available, meta as reg_meta  # type: ignore
    from core.pipeline import build_ctx, apply_pipeline, normalize_preset  # type: ignore

from .panel_loader import get_panel_class, list_panels
from .panel_base import PanelContext
from .widgets.image_canvas import ImageCanvas
from .widgets.hud import Hud
from .widgets.mosaic_view import MosaicMini
from .widgets.graph_view import GraphView
from .widgets.preset_manager import PresetManager
from .docking import DockManager

try:
    RESAMPLING = Image.Resampling
except Exception:  # Pillow<10
    RESAMPLING = Image

LAYOUT_PATH = os.path.expanduser("~/.glitchlab/layout.json")

# --- ensure filters package is imported so registry is populated ---
def _force_import_filters() -> None:
    try:
        import importlib  # stdlib
        import glitchlab.filters as _flt  # side-effect: registers filters
        # opcjonalnie: przeładuj w dev trybie, gdy istnieją
        # importlib.reload(_flt)
        print("[app] filters imported OK")
    except Exception as e:
        try:
            # fallback na strukturę bez pakietu
            import filters as _flt  # type: ignore
            print("[app] filters imported via local 'filters' package")
        except Exception as e2:
            print(f"[app] WARN: could not import filters: {e} / {e2}")


class App(tk.Frame):
    """Główna rama aplikacji z naprawionymi hotkeys, widokami show/hide i PresetManagerem."""

    def __init__(self, master: Optional[tk.Misc] = None):
        _force_import_filters()
        root = master if isinstance(master, tk.Tk) else None
        if root is None:
            root = tk.Tk()
            root.title("GlitchLab v3 GUI")
            root.geometry("1380x860")
        super().__init__(root)
        self.pack(fill="both", expand=True)

        # state
        self.img_u8: Optional[np.ndarray] = None
        self.result_u8: Optional[np.ndarray] = None
        self.seed: int = 7
        self.cfg: Dict[str, Any] = {"version": 2, "amplitude": {"kind": "none", "strength": 1.0},
                                    "edge_mask": {"thresh": 60, "dilate": 0, "ksize": 3}, "steps": []}
        self.active_filter: Optional[str] = None
        self.active_params: Dict[str, Any] = {}
        self.extra_masks: Dict[str, np.ndarray] = {}
        self.last_ctx = None

        self._build_menu(root)
        self._build_layout(root)
        self._refresh_filter_list()
        self._restore_layout_silent()

    # ---------------- Menu ----------------
    def _build_menu(self, root: tk.Tk) -> None:
        m = tk.Menu(root); root.config(menu=m)

        mf = tk.Menu(m, tearoff=0)
        mf.add_command(label="Open image…", command=self.on_open_image, accelerator="Ctrl+O")
        mf.add_command(label="Load preset…", command=self.on_load_preset)
        mf.add_command(label="Load mask…", command=self.on_load_mask)
        mf.add_separator()
        mf.add_command(label="Save result…", command=self.on_save_result, accelerator="Ctrl+S")
        mf.add_separator()
        mf.add_command(label="Exit", command=root.quit)
        m.add_cascade(label="File", menu=mf)

        mr = tk.Menu(m, tearoff=0)
        mr.add_command(label="Apply filter", command=self.on_apply_filter, accelerator="Ctrl+R")
        mr.add_command(label="Apply preset steps", command=self.on_apply_preset_steps)
        m.add_cascade(label="Run", menu=mr)

        self._menu_view = mv = tk.Menu(m, tearoff=0)
        mv.add_checkbutton(label="Show left panel (F9)", command=lambda: self._toggle_slot('left'), onvalue=True, offvalue=False, variable=tk.BooleanVar())
        mv.add_checkbutton(label="Show right panel (F10)", command=lambda: self._toggle_slot('right'), onvalue=True, offvalue=False, variable=tk.BooleanVar())
        mv.add_checkbutton(label="Show bottom HUD (F11)", command=lambda: self._toggle_slot('bottom'), onvalue=True, offvalue=False, variable=tk.BooleanVar())
        mv.add_separator()
        mv.add_command(label="Save layout", command=self._save_layout)
        mv.add_command(label="Restore layout", command=self._restore_layout)
        mv.add_separator()
        mv.add_command(label="List panels (console)", command=self._dump_panels_console)
        m.add_cascade(label="View", menu=mv)

    # ---------------- Layout ----------------
    def _build_layout(self, root: tk.Tk) -> None:
        # Splitters
        self.vsplit = ttk.Panedwindow(self, orient="vertical")
        self.vsplit.pack(fill="both", expand=True)

        self.top_split = ttk.Panedwindow(self.vsplit, orient="horizontal")
        self.left_host = ttk.Frame(self.top_split, width=280)
        self.center_host = ttk.Frame(self.top_split)
        self.right_host = ttk.Frame(self.top_split, width=380)

        self.top_split.add(self.left_host, weight=0)
        self.top_split.add(self.center_host, weight=1)
        self.top_split.add(self.right_host, weight=0)

        self.vsplit.add(self.top_split, weight=1)

        # Bottom HUD host
        self.bottom_host = ttk.Frame(self.vsplit, height=220)
        self.vsplit.add(self.bottom_host, weight=0)

        # Center: viewer z zoom/pan/fit
        self.viewer = ImageCanvas(self.center_host); self.viewer.pack(fill="both", expand=True)

        # Bottom HUD
        self.hud = Hud(self.bottom_host); self.hud.pack(fill="both", expand=True)

        # Right: notebook
        self.right_nb = ttk.Notebook(self.right_host)
        self.right_nb.pack(fill="both", expand=True)
        self.tab_filter = ttk.Frame(self.right_nb); self.right_nb.add(self.tab_filter, text="Filter")
        self.tab_global = ttk.Frame(self.right_nb); self.right_nb.add(self.tab_global, text="Global")
        self.tab_views = ttk.Frame(self.right_nb); self.right_nb.add(self.tab_views, text="Views")

        self._build_filter_tab(self.tab_filter)
        self._build_global_tab(self.tab_global)
        self._build_views_tab(self.tab_views)

        # Left: notebook — Welcome + Presets
        self.left_nb = ttk.Notebook(self.left_host)
        self.left_nb.pack(fill="both", expand=True)
        self.left_welcome = ttk.Frame(self.left_nb)
        ttk.Label(self.left_welcome, text="Welcome\n\nTips:\n• F9/F10/F11 toggle panels\n• Ctrl+O / Ctrl+S / Ctrl+R\n• Load masks and set mask_key in panel for overlays.",
                  justify="left").pack(padx=8, pady=8, anchor="nw")
        self.left_nb.add(self.left_welcome, text="Welcome")

        self.preset_mgr = PresetManager(self.left_nb, on_load=self._apply_cfg_to_ui, on_apply=self.on_apply_preset_steps, on_save=None, get_current_cfg=lambda: self.cfg, on_add_step=self._add_current_step_to_cfg)
        self.left_nb.add(self.preset_mgr, text="Presets")

        # DockManager manages undock/dock
        self.dock = DockManager(root, {
            "left": self.left_host,
            "right": self.right_host,
            "bottom": self.bottom_host,
        })

        # View state
        self._view_show_left = tk.BooleanVar(value=True)
        self._view_show_right = tk.BooleanVar(value=True)
        self._view_show_bottom = tk.BooleanVar(value=True)

        # Hotkeys
        self._bind_hotkeys(root)

    def _build_filter_tab(self, frm: ttk.Frame) -> None:
        row0 = ttk.Frame(frm, padding=8); row0.pack(fill="x")
        ttk.Label(row0, text="Filter:").pack(side="left")
        self.cmb_filter = ttk.Combobox(row0, state="readonly", values=[])
        self.cmb_filter.pack(side="left", fill="x", expand=True, padx=6)
        self.cmb_filter.bind("<<ComboboxSelected>>", lambda e: self._on_filter_change())
        ttk.Button(row0, text="Apply", command=self.on_apply_filter).pack(side="right")

        self.panel_container = ttk.Frame(frm, padding=8); self.panel_container.pack(fill="both", expand=True)
        self._mount_panel("")

    def _build_global_tab(self, g: ttk.Frame) -> None:
        amp = ttk.LabelFrame(g, text="Amplitude", padding=8); amp.pack(fill="x", padx=8, pady=8)
        kinds = ["none","linear_x","linear_y","radial","perlin","mask"]
        self.var_amp_kind = tk.StringVar(value="none")
        self.var_amp_strength = tk.DoubleVar(value=1.0)
        self.var_amp_mask = tk.StringVar(value="")
        ttk.Label(amp, text="kind").grid(row=0,column=0,sticky="w"); ttk.Combobox(amp, values=kinds, state="readonly", textvariable=self.var_amp_kind, width=12).grid(row=0,column=1,sticky="ew",padx=6,pady=2)
        ttk.Label(amp, text="strength").grid(row=1,column=0,sticky="w"); ttk.Entry(amp, textvariable=self.var_amp_strength, width=10).grid(row=1,column=1,sticky="w",padx=6,pady=2)
        ttk.Label(amp, text="mask_key").grid(row=2,column=0,sticky="w"); ttk.Entry(amp, textvariable=self.var_amp_mask, width=12).grid(row=2,column=1,sticky="w",padx=6,pady=2)

        ed = ttk.LabelFrame(g, text="Edge mask", padding=8); ed.pack(fill="x", padx=8, pady=8)
        self.var_edge_thresh = tk.IntVar(value=60); self.var_edge_dilate = tk.IntVar(value=0); self.var_edge_ksize = tk.IntVar(value=3)
        ttk.Label(ed, text="thresh").grid(row=0,column=0,sticky="w"); ttk.Spinbox(ed,from_=0,to=255,textvariable=self.var_edge_thresh,width=8).grid(row=0,column=1,sticky="w",padx=6,pady=2)
        ttk.Label(ed, text="dilate").grid(row=1,column=0,sticky="w"); ttk.Spinbox(ed,from_=0,to=64,textvariable=self.var_edge_dilate,width=8).grid(row=1,column=1,sticky="w",padx=6,pady=2)
        ttk.Label(ed, text="ksize").grid(row=2,column=0,sticky="w"); ttk.Spinbox(ed,from_=1,to=9,increment=2,textvariable=self.var_edge_ksize,width=8).grid(row=2,column=1,sticky="w",padx=6,pady=2)

        seedf = ttk.LabelFrame(g, text="Seed", padding=8); seedf.pack(fill="x", padx=8, pady=(0,8))
        self.var_seed = tk.IntVar(value=int(self.seed))
        ttk.Spinbox(seedf, from_=0, to=2**31-1, textvariable=self.var_seed, width=14).grid(row=0, column=0, sticky="w", padx=6, pady=2)

        rowb = ttk.Frame(g, padding=8); rowb.pack(fill="x")
        ttk.Button(rowb, text="Apply preset steps", command=self.on_apply_preset_steps).pack(side="left")
        ttk.Button(rowb, text="Refresh HUD", command=self._update_hud).pack(side="right")

    def _build_views_tab(self, v: ttk.Frame) -> None:
        self.mosaic = MosaicMini(v); self.mosaic.pack(fill="x", padx=8, pady=4)
        self.graph = GraphView(v); self.graph.pack(fill="both", expand=True, padx=8, pady=4)

    # ---------------- Panels ----------------
    def _refresh_filter_list(self) -> None:
        names = sorted(reg_available())
        if not names:
            # spróbuj jeszcze raz dograć pakiet filtrów
            _force_import_filters()
            names = sorted(reg_available())
        self.cmb_filter["values"] = names
        if self.active_filter is None and names:
            self.cmb_filter.set(names[0]);
            self._on_filter_change()

    def _on_filter_change(self) -> None:
        name = self.cmb_filter.get().strip()
        if not name: return
        self.active_filter = name
        defaults = dict(reg_meta(name)["defaults"])
        params = {k: self.active_params.get(k, defaults.get(k)) for k in defaults.keys()}
        self.active_params = params
        self._mount_panel(name, defaults, params)

    def _mount_panel(self, filter_name: str, defaults: Optional[Dict[str, Any]] = None,
                     params: Optional[Dict[str, Any]] = None) -> None:
        for w in self.panel_container.winfo_children():
            w.destroy()
        defaults = defaults or {}
        params = params or {}
        panel_cls = get_panel_class(filter_name or "")
        ctx = PanelContext(filter_name=filter_name or "(none)", defaults=defaults, params=params,
                           on_change=self._on_panel_change, cache_ref={})
        try:
            panel = panel_cls(self.panel_container, ctx=ctx)
        except TypeError:
            panel = panel_cls(self.panel_container)
        if hasattr(panel, "load_defaults"):
            try: panel.load_defaults(defaults)
            except Exception: pass
        panel.pack(fill="both", expand=True)

    def _on_panel_change(self, new_params: Dict[str, Any]) -> None:
        self.active_params = dict(new_params)

    # ---------------- Actions ----------------
    def on_open_image(self) -> None:
        path = filedialog.askopenfilename(title="Open image",
            filetypes=[("Images","*.png;*.jpg;*.jpeg;*.webp;*.bmp"),("All files","*.*")])
        if not path: return
        try:
            im = Image.open(path).convert("RGB")
            self.img_u8 = np.asarray(im, dtype=np.uint8)
            self._show_on_canvas(self.img_u8)
            self.result_u8 = None
            self._status(f"Loaded: {os.path.basename(path)} {self.img_u8.shape[1]}x{self.img_u8.shape[0]}")
        except Exception as e:
            messagebox.showerror("Open image", str(e))

    def on_save_result(self) -> None:
        if self.result_u8 is None:
            messagebox.showinfo("Save result", "No result image to save yet."); return
        path = filedialog.asksaveasfilename(title="Save result",
            defaultextension=".png", filetypes=[("PNG","*.png"),("JPEG","*.jpg;*.jpeg"),("All files","*.*")])
        if not path: return
        try:
            Image.fromarray(self.result_u8, "RGB").save(path)
            self._status(f"Saved: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Save result", str(e))

    def on_load_mask(self) -> None:
        if self.img_u8 is None:
            messagebox.showinfo("Load mask", "Open an image first."); return
        path = filedialog.askopenfilename(title="Load mask (grayscale)",
            filetypes=[("Images","*.png;*.jpg;*.jpeg;*.webp;*.bmp"),("All files","*.*")])
        if not path: return
        try:
            im = Image.open(path).convert("L").resize(
                (self.img_u8.shape[1], self.img_u8.shape[0]), RESAMPLING.BICUBIC)
            g = np.asarray(im, dtype=np.float32) / 255.0
            key = os.path.splitext(os.path.basename(path))[0]
            self.extra_masks[key] = np.clip(g, 0.0, 1.0)
            if self.var_amp_kind.get() == "mask" and not self.var_amp_mask.get().strip():
                self.var_amp_mask.set(key)
            self._status(f"Mask loaded: {key}")
        except Exception as e:
            messagebox.showerror("Load mask", str(e))

    def on_load_preset(self) -> None:
        path = filedialog.askopenfilename(title="Load preset (YAML/JSON)",
            filetypes=[("YAML","*.yaml;*.yml"),("JSON","*.json"),("All files","*.*")])
        if not path: return
        try:
            text = open(path, "r", encoding="utf-8").read()
            if path.lower().endswith((".yaml",".yml")):
                try:
                    import yaml  # type: ignore
                    raw = yaml.safe_load(text)
                except Exception:
                    raw = json.loads(text)
            else:
                raw = json.loads(text)
            cfg = normalize_preset(raw)
            self._apply_cfg_to_ui(cfg)
            self._status(f"Preset loaded: {os.path.basename(path)} ({len(cfg.get('steps', []))} steps)")
        except Exception as e:
            messagebox.showerror("Load preset", str(e))

    def _apply_cfg_to_ui(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        amp = cfg.get("amplitude", {}) or {}
        self.var_amp_kind.set(str(amp.get("kind","none"))); self.var_amp_strength.set(float(amp.get("strength",1.0)))
        self.var_amp_mask.set(str(amp.get("mask_key","")))
        edge = cfg.get("edge_mask", {}) or {}
        self.var_edge_thresh.set(int(edge.get("thresh",60))); self.var_edge_dilate.set(int(edge.get("dilate",0)))
        self.var_edge_ksize.set(int(edge.get("ksize",3)))
        steps = cfg.get("steps", []) or []
        if steps:
            first = steps[0]; name = first.get("name","")
            if name and name in reg_available():
                self.cmb_filter.set(name); self._on_filter_change()
                defaults = dict(reg_meta(name)["defaults"]); params = dict(defaults)
                params.update(first.get("params", {}) or {})
                self._mount_panel(name, defaults, params)

    def _collect_cfg_from_ui(self) -> None:
        self.cfg["version"] = 2
        try: self.seed = int(self.var_seed.get())
        except Exception: pass
        amp = {"kind": self.var_amp_kind.get().strip() or "none",
               "strength": float(self.var_amp_strength.get())}
        mask_key = self.var_amp_mask.get().strip()
        if amp["kind"] == "mask" and mask_key: amp["mask_key"] = mask_key
        self.cfg["amplitude"] = amp
        self.cfg["edge_mask"] = {
            "thresh": int(self.var_edge_thresh.get()),
            "dilate": int(self.var_edge_dilate.get()),
            "ksize": int(self.var_edge_ksize.get()),
        }
        if self.active_filter:
            step = {"name": self.active_filter, "params": dict(self.active_params)}
            if self.cfg.get("steps"):
                self.cfg["steps"][0] = step
            else:
                self.cfg["steps"] = [step]

    def _add_current_step_to_cfg(self) -> None:
        """Dokleja aktualny filtr+parametry do cfg['steps'] (na końcu)."""
        self._collect_cfg_from_ui()
        step = {"name": self.active_filter or "(none)",
                "params": dict(self.active_params)}
        lst = list(self.cfg.get("steps", []))
        lst.append(step)
        self.cfg["steps"] = lst
        messagebox.showinfo("Preset", f"Dodano krok: {step['name']}")

    def on_apply_filter(self) -> None:
        if self.img_u8 is None:
            messagebox.showinfo("Apply filter", "Open an image first."); return
        if not self.active_filter:
            messagebox.showinfo("Apply filter", "Select a filter."); return
        self._collect_cfg_from_ui()
        ctx = build_ctx(self.img_u8, seed=self.seed, cfg=self.cfg)
        for k, m in self.extra_masks.items(): ctx.masks[k] = m
        steps = [{"name": self.active_filter, "params": dict(self.active_params)}]
        t0 = time.time()
        out = apply_pipeline(self.img_u8, ctx, steps, fail_fast=True, debug_log=[])
        t_ms = (time.time() - t0) * 1000.0
        self.last_ctx = ctx; self.result_u8 = out
        self._show_on_canvas(out); self._update_hud()
        self._status(f"Applied '{self.active_filter}' in {t_ms:.1f} ms")

    def on_apply_preset_steps(self) -> None:
        if self.img_u8 is None:
            messagebox.showinfo("Apply preset", "Open an image first."); return
        self._collect_cfg_from_ui()
        steps = list(self.cfg.get("steps", []))
        if not steps:
            messagebox.showinfo("Apply preset", "Preset has no steps."); return
        ctx = build_ctx(self.img_u8, seed=self.seed, cfg=self.cfg)
        for k, m in self.extra_masks.items(): ctx.masks[k] = m
        t0 = time.time()
        out = apply_pipeline(self.img_u8, ctx, steps, fail_fast=True, debug_log=[])
        t_ms = (time.time() - t0) * 1000.0
        self.last_ctx = ctx; self.result_u8 = out
        self._show_on_canvas(out); self._update_hud()
        self._status(f"Applied preset ({len(steps)} steps) in {t_ms:.1f} ms")

    # ---------------- HUD & viewer ----------------
    def _update_hud(self) -> None:
        if self.last_ctx is None:
            self.hud.render_from_cache(None); 
            self.mosaic.set_overlay(None)
            self.graph.set_ast_json(None)
            return
        self.hud.render_from_cache(self.last_ctx)
        cache = getattr(self.last_ctx, "cache", {}) or {}
        mosaic = cache.get("stage/0/mosaic")
        if mosaic is None:
            # fallback: wybierz pierwszą diagnostykę 'diag/*'
            for k,v in cache.items():
                if isinstance(k, str) and k.startswith("diag/") and isinstance(v, np.ndarray):
                    mosaic = v; break
        self.mosaic.set_overlay(mosaic)
        self.graph.set_ast_json(cache.get("ast/json"))

    def _show_on_canvas(self, arr_rgb_u8: np.ndarray) -> None:
        self.viewer.set_image(Image.fromarray(arr_rgb_u8, "RGB"))

    # ---------------- View toggles / hotkeys ----------------
    def _toggle_slot(self, which: str) -> None:
        if which == "left":
            self._view_show_left.set(not self._view_show_left.get())
            if self._view_show_left.get():
                if self.left_host not in self.top_split.panes():
                    self.top_split.insert(0, self.left_host)
            else:
                try: self.top_split.forget(self.left_host)
                except Exception: pass
        elif which == "right":
            self._view_show_right.set(not self._view_show_right.get())
            if self._view_show_right.get():
                if self.right_host not in self.top_split.panes():
                    self.top_split.add(self.right_host, weight=0)
            else:
                try: self.top_split.forget(self.right_host)
                except Exception: pass
        elif which == "bottom":
            self._view_show_bottom.set(not self._view_show_bottom.get())
            if self._view_show_bottom.get():
                if self.bottom_host not in self.vsplit.panes():
                    self.vsplit.add(self.bottom_host, weight=0)
            else:
                try: self.vsplit.forget(self.bottom_host)
                except Exception: pass

    def _bind_hotkeys(self, root: tk.Tk) -> None:
        root.bind("<F9>", lambda e: self._toggle_slot("left"))
        root.bind("<F10>", lambda e: self._toggle_slot("right"))
        root.bind("<F11>", lambda e: self._toggle_slot("bottom"))
        root.bind("<Control-r>", lambda e: self.on_apply_filter())
        root.bind("<Control-o>", lambda e: self.on_open_image())
        root.bind("<Control-s>", lambda e: self.on_save_result())

    # ---------------- Layout save/restore ----------------
    def _save_layout(self) -> None:
        data = {
            "show_left": self._view_show_left.get(),
            "show_right": self._view_show_right.get(),
            "show_bottom": self._view_show_bottom.get(),
        }
        os.makedirs(os.path.dirname(LAYOUT_PATH), exist_ok=True)
        try:
            with open(LAYOUT_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            messagebox.showinfo("Layout", f"Saved to {LAYOUT_PATH}")
        except Exception as e:
            messagebox.showerror("Layout", str(e))

    def _restore_layout_silent(self) -> None:
        if not os.path.exists(LAYOUT_PATH): return
        try:
            with open(LAYOUT_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not data.get("show_left", True): 
                if self.left_host in self.top_split.panes():
                    self.top_split.forget(self.left_host)
                self._view_show_left.set(False)
            if not data.get("show_right", True): 
                if self.right_host in self.top_split.panes():
                    self.top_split.forget(self.right_host)
                self._view_show_right.set(False)
            if not data.get("show_bottom", True): 
                if self.bottom_host in self.vsplit.panes():
                    self.vsplit.forget(self.bottom_host)
                self._view_show_bottom.set(False)
        except Exception:
            pass

    def _restore_layout(self) -> None:
        self._restore_layout_silent()
        messagebox.showinfo("Layout", "Layout restored (if saved).")

    # ---------------- misc ----------------
    def _status(self, text: str) -> None:
        self.master.title(f"GlitchLab v3 — {text}")

    def _dump_panels_console(self) -> None:
        try:
            names = list_panels()
            print("Registered panels:", names)
            messagebox.showinfo("Panels", f"Registered panels: {len(names)}")
        except Exception as e:
            messagebox.showinfo("Panels", f"(error) {e}")
