# glitchlab/gui/app.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import importlib
from pathlib import Path
import threading
import time
import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np

# --- core ---
try:
    from glitchlab.core.registry import available as registry_available, get as registry_get
except Exception:
    def registry_available():
        return []
    def registry_get(name):
        raise KeyError(name)

try:
    from glitchlab.core.pipeline import apply_pipeline
except Exception:
    def apply_pipeline(img, ctx, steps, fail_fast=True, metrics=True, debug_log=None):
        return img

# Wymuś import filtrów (żeby rejestr nie był pusty)
try:
    from glitchlab import filters as _filters  # noqa: F401
    print("[app] filters imported OK")
except Exception as e:
    print("[app] filters import failed:", e)

# --- widgets / gui helpers ---
try:
    from glitchlab.gui.widgets.image_canvas import ImageCanvas
except Exception:
    ImageCanvas = None

try:
    from glitchlab.gui.widgets.hud import Hud
except Exception:
    Hud = None

try:
    from glitchlab.gui.widgets.graph_view import GraphView
except Exception:
    GraphView = None

try:
    from glitchlab.gui.widgets.mosaic_view import MosaicMini
except Exception:
    MosaicMini = None

try:
    from glitchlab.gui.widgets.preset_manager import PresetManager
except Exception:
    PresetManager = None

try:
    from glitchlab.gui.widgets.mask_chooser import MaskChooser
except Exception:
    MaskChooser = None

# panele „ręczne”
try:
    from glitchlab.gui.panel_base import PanelContext  # context dla paneli
except Exception:
    class PanelContext:
        def __init__(self, **kw): self.__dict__.update(kw)


# ------- narzędzia -------
def np_to_u8(img: np.ndarray) -> np.ndarray | None:
    if img is None:
        return None
    if isinstance(img, np.ndarray) and img.dtype == np.uint8:
        return img
    return np.clip(img, 0, 255).astype(np.uint8)

def _win_path(p: str | Path) -> str:
    return str(Path(p))

def determine_default_preset_dir() -> Path:
    env = os.environ.get("GLITCHLAB_PRESETS_DIR")
    if env and Path(env).exists():
        return Path(env)
    fixed = Path(r"C:\Users\donke\PycharmProjects\glitchlab_project\glitchlab\presets")
    fixed.mkdir(parents=True, exist_ok=True)
    return fixed


# ------ fallback: dynamiczny formularz parametrów ------
class ParamForm(ttk.Frame):
    def __init__(self, parent, get_filter_callable):
        super().__init__(parent)
        self._get_filter_callable = get_filter_callable
        self._controls: dict[str, tk.Variable] = {}
        self._current_name: str | None = None

    def build_for(self, name: str):
        for w in self.winfo_children(): w.destroy()
        self._controls.clear()
        self._current_name = name

        spec = self._resolve_spec(name)
        if not spec:
            ttk.Label(self, text="(Brak znanego schematu parametrów; użyj presetów)").pack(anchor="w", padx=6, pady=6)
            return

        grid = ttk.Frame(self); grid.pack(fill="x", expand=True, padx=6, pady=6)
        for i, item in enumerate(spec):
            n = item["name"]; typ = item.get("type", "float")
            default = item.get("default"); choices = item.get("choices")
            ttk.Label(grid, text=n).grid(row=i, column=0, sticky="w", padx=(2,6), pady=2)

            if   typ == "bool":
                var = tk.BooleanVar(value=bool(default)); ctrl = ttk.Checkbutton(grid, variable=var)
            elif typ == "enum" and choices:
                var = tk.StringVar(value=str(default if default in choices else (choices[0] if choices else "")))
                ctrl = ttk.Combobox(grid, textvariable=var, state="readonly", values=list(choices))
            elif typ == "int":
                var = tk.StringVar(value=str(int(default if default is not None else 0)))
                ctrl = ttk.Entry(grid, textvariable=var, width=10)
            elif typ == "float":
                var = tk.StringVar(value=str(float(default if default is not None else 0.0)))
                ctrl = ttk.Entry(grid, textvariable=var, width=10)
            else:
                var = tk.StringVar(value=str(default if default is not None else ""))
                ctrl = ttk.Entry(grid, textvariable=var)
            ctrl.grid(row=i, column=1, sticky="ew", padx=(0,6), pady=2)
            grid.columnconfigure(1, weight=1)
            self._controls[n] = var

    def values(self) -> dict:
        spec = self._resolve_spec(self._current_name) if self._current_name else []
        type_map = {i["name"]: i.get("type", "float") for i in spec}
        out = {}
        for k, var in self._controls.items():
            v = var.get(); typ = type_map.get(k, "float")
            try:
                if   typ == "bool":  out[k] = bool(v)
                elif typ == "int":   out[k] = int(v)
                elif typ == "float": out[k] = float(v)
                else:                out[k] = str(v)
            except Exception:
                out[k] = v
        return out

    def _resolve_spec(self, name: str) -> list[dict]:
        try:
            f = self._get_filter_callable(name)
        except Exception:
            return []
        for attr in ("schema", "params_schema", "PARAMS", "SPEC"):
            try:
                obj = getattr(f, attr)
                schema = obj() if callable(obj) else obj
                if isinstance(schema, (list, tuple)):
                    return [dict(x) for x in schema]
                if isinstance(schema, dict) and "params" in schema:
                    return [dict(x) for x in (schema["params"] or [])]
            except Exception:
                pass
        try:
            import inspect
            sig = inspect.signature(f); spec: list[dict] = []
            for p in sig.parameters.values():
                if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD): continue
                if p.name in ("img", "ctx"): continue
                default = p.default if p.default is not inspect._empty else None
                typ = "float"
                if isinstance(default, bool):  typ = "bool"
                elif isinstance(default, int): typ = "int"
                elif isinstance(default, str): typ = "str"
                spec.append({"name": p.name, "type": typ, "default": default})
            return spec
        except Exception:
            return []


# ----------------- RULERS -----------------
class _Ruler(ttk.Frame):
    def __init__(self, master, orientation: str = "x", height=26, width=28):
        super().__init__(master)
        self.orientation = orientation
        if orientation == "x":
            self.canvas = tk.Canvas(self, height=height, bg="#2b2b2b", highlightthickness=0)
            self.canvas.pack(fill="x", expand=True)
        else:
            self.canvas = tk.Canvas(self, width=width, bg="#2b2b2b", highlightthickness=0)
            self.canvas.pack(fill="y", expand=True)
        self.zoom = 1.0
        self.length_px = 100
        self._marker_line_id = None
        self._marker_text_id = None
        self.canvas.bind("<Configure>", lambda _e: self._redraw())

    def set_zoom_and_size(self, zoom: float, length_px: int):
        self.zoom = max(zoom, 1e-6)
        self.length_px = max(1, int(length_px))
        self._redraw()

    def update_marker_view_px(self, view_px: int):
        c = self.canvas
        if self._marker_line_id is not None:
            try: c.delete(self._marker_line_id)
            except Exception: pass
            self._marker_line_id = None
        if self._marker_text_id is not None:
            try: c.delete(self._marker_text_id)
            except Exception: pass
            self._marker_text_id = None

        if self.orientation == "x":
            x = int(view_px)
            h = int(c.winfo_height() or 26)
            self._marker_line_id = c.create_line(x, 0, x, h, fill="#e0e0e0", dash=(4, 3))
            img_coord = int(round(x / max(self.zoom, 1e-6)))
            self._marker_text_id = c.create_text(x+4, h-2, anchor="se",
                                                 text=str(img_coord), fill="#d0d0d0", font=("", 8))
        else:
            y = int(view_px)
            w = int(c.winfo_width() or 28)
            self._marker_line_id = c.create_line(0, y, w, y, fill="#e0e0e0", dash=(4, 3))
            img_coord = int(round(y / max(self.zoom, 1e-6)))
            self._marker_text_id = c.create_text(w-2, y+2, anchor="se",
                                                 text=str(img_coord), fill="#d0d0d0", font=("", 8))

    def _redraw(self):
        c = self.canvas
        c.delete("all")
        try:
            if self.orientation == "x":
                c.create_text(6, 2, anchor="nw", text="X →", fill="#9fbfff", font=("", 9, "bold"))
            else:
                c.create_text(2, 2, anchor="nw", text="Y ↓", fill="#9fbfff", font=("", 9, "bold"))
        except Exception:
            pass
        if self.orientation == "x":
            w = int(c.winfo_width() or (self.length_px * self.zoom))
            step = max(10, int(50 * self.zoom))
            for x in range(0, w, step):
                c.create_line(x, 0, x, 12, fill="#aaaaaa")
                c.create_text(x+2, 14, anchor="nw", text=str(int(x/self.zoom)), fill="#cccccc", font=("", 8))
        else:
            h = int(c.winfo_height() or (self.length_px * self.zoom))
            step = max(10, int(50 * self.zoom))
            for y in range(0, h, step):
                c.create_line(0, y, 12, y, fill="#aaaaaa")
                c.create_text(14, y+2, anchor="nw", text=str(int(y/self.zoom)), fill="#cccccc", font=("", 8))


# ----------------- TOOLBOX -----------------
class _Toolbox(ttk.Frame):
    """Mini-widget na tryby + toggles (crosshair/rulers)."""
    def __init__(self, master, on_mode_change, on_toggle):
        super().__init__(master)
        self.on_mode_change = on_mode_change
        self.on_toggle = on_toggle

        self.var_mode = tk.StringVar(value="hand")
        self.var_cross = tk.BooleanVar(value=True)
        self.var_rulers = tk.BooleanVar(value=True)

        self._mode_btn("🖐", "hand",  "Pan / drag").pack(side="left", padx=(4,0))
        self._mode_btn("🔍", "zoom",  "Zoom tool").pack(side="left", padx=2)
        self._mode_btn("🎯", "pick",  "Color picker").pack(side="left", padx=2)
        self._mode_btn("📏", "measure","Measure distance").pack(side="left", padx=2)

        ttk.Separator(self, orient="vertical").pack(side="left", fill="y", padx=6)

        self._toggle_btn("⊕", self.var_cross, "cross", "Crosshair").pack(side="left", padx=2)
        self._toggle_btn("📐", self.var_rulers,"rulers","Rulers").pack(side="left", padx=2)

    def _mode_btn(self, text, value, tip):
        b = ttk.Radiobutton(self, text=text, value=value, variable=self.var_mode, style="Toolbutton",
                            command=lambda: self.on_mode_change(self.var_mode.get()))
        b.tooltip = tip
        return b

    def _toggle_btn(self, text, var, key, tip):
        b = ttk.Checkbutton(self, text=text, variable=var, style="Toolbutton",
                            command=lambda: self.on_toggle(key, bool(var.get())))
        b.tooltip = tip
        return b

    def set_cross(self, v: bool):   self.var_cross.set(bool(v))
    def set_rulers(self, v: bool):  self.var_rulers.set(bool(v))


# ------- główna aplikacja -------
class App(ttk.Frame):
    def __init__(self, root: tk.Tk):
        super().__init__(root)
        self.root = root
        self.root.title("GlitchLab")
        self.pack(fill="both", expand=True)

        # stan obrazu
        self.img_u8: np.ndarray | None = None
        self._display_u8: np.ndarray | None = None

        # kontekst
        self.last_ctx = type("Ctx", (), {})()
        self.last_ctx.rng = np.random.default_rng(7)
        self.last_ctx.amplitude = None
        self.last_ctx.masks = {}
        self.last_ctx.cache = {}
        self.last_ctx.meta = {}

        # busy / operacje
        self.is_busy = False
        self._op_thread: threading.Thread | None = None
        self._cancel_requested = False

        # dir presetów
        self.preset_dir: Path = determine_default_preset_dir()

        # preset w pamięci
        self.preset_cfg: dict = {
            "version": 2,
            "seed": 7,
            "amplitude": {"kind": "none", "strength": 1.0},
            "edge_mask": {"thresh": 60, "dilate": 0, "ksize": 3},
            "steps": [],
            "__preset_dir": str(self.preset_dir),
        }

        # stan zakładki Filter
        self._filter_panel: tk.Widget | None = None
        self._filter_params: dict | None = None
        self._fallback_form: ParamForm | None = None

        # GUI refs
        self.btn_apply_filter: ttk.Button | None = None
        self.cmb_filter: ttk.Combobox | None = None
        self.menu_objects: dict[str, tk.Menu] = {}

        # rulers / crosshair (sterowane przez toolbox)
        self.ruler_top: _Ruler | None = None
        self.ruler_left: _Ruler | None = None
        self.var_rulers = tk.BooleanVar(value=True)
        self.var_cross = tk.BooleanVar(value=True)

        # dynamiczny krzyż – dwa cienkie canvasy (1px) nakładane na obraz
        self.cross_h: tk.Canvas | None = None  # wys. 1px
        self.cross_v: tk.Canvas | None = None  # szer. 1px

        # overlay z osiami i współrzędnymi
        self.axes: tk.Canvas | None = None
        self.axes_coord_id = None

        # toolbox
        self.toolbox: _Toolbox | None = None

        # bieżąca pozycja kursora (w przestrzeni widoku)
        self._mouse_view_x = 0
        self._mouse_view_y = 0

        # transformacje widoku (2.5D)
        self.var_roll = tk.DoubleVar(value=0.0)
        self.var_pitch = tk.DoubleVar(value=0.0)
        self.var_yaw = tk.DoubleVar(value=0.0)

        self._build_menu()
        self._build_layout(root)

        # skróty
        self.root.bind("<Control-o>", lambda e: self.on_open())
        self.root.bind("<Control-s>", lambda e: self.on_save())
        self.root.bind("<F10>", lambda e: self._toggle_slot("bottom"))
        self.root.bind("<F9>",  lambda e: self._toggle_slot("right"))
        self.root.bind("<F8>",  lambda e: self._toggle_left_tools())

        # globalny tracking kursora
        self.root.bind_all("<Motion>", self._on_any_motion)

    # ---------- UI ----------
    def _build_menu(self):
        mbar = tk.Menu(self.root); self.root.config(menu=mbar)
        self.menu_objects["menubar"] = mbar

        m_file = tk.Menu(mbar, tearoff=False); mbar.add_cascade(label="File", menu=m_file)
        m_file.add_command(label="Open image…", command=self.on_open, accelerator="Ctrl+O")
        m_file.add_command(label="Save result as…", command=self.on_save, accelerator="Ctrl+S")
        m_file.add_separator()
        m_file.add_command(label="Exit", command=self.root.destroy)
        self.menu_objects["file"] = m_file

        m_view = tk.Menu(mbar, tearoff=False); mbar.add_cascade(label="View", menu=m_view)
        m_view.add_command(label="Toggle bottom (F10)", command=lambda: self._toggle_slot("bottom"))
        m_view.add_command(label="Toggle right (F9)",   command=lambda: self._toggle_slot("right"))
        m_view.add_command(label="Toggle left tools (F8)", command=self._toggle_left_tools)
        m_view.add_separator()
        # (usunięto duplikaty checkboxów Crosshair/Rulers – steruje Toolbox)
        m_view.add_command(label="Zoom 100%", command=self.on_zoom_100)
        m_view.add_command(label="Fit to window", command=self.on_zoom_fit)
        m_view.add_command(label="Center", command=self.on_center)
        self.menu_objects["view"] = m_view

        m_settings = tk.Menu(mbar, tearoff=False); mbar.add_cascade(label="Settings", menu=m_settings)
        m_settings.add_command(label="Preset Folder…", command=self.on_choose_preset_dir)
        self.menu_objects["settings"] = m_settings

        m_help = tk.Menu(mbar, tearoff=False); mbar.add_cascade(label="Help", menu=m_help)
        m_help.add_command(label="About…", command=self.on_about)
        self.menu_objects["help"] = m_help

    def _build_layout(self, root: tk.Tk):
        # pasek techniczny
        self.statusbar = ttk.Frame(self); self.statusbar.pack(side="bottom", fill="x")
        self.prog = ttk.Progressbar(self.statusbar, mode="indeterminate", length=140)
        self.btn_show_log = ttk.Button(self.statusbar, text="Show log", command=self._show_tech_log)
        self.lbl_status = ttk.Label(self.statusbar, text="Ready")
        self.lbl_coord = ttk.Label(self.statusbar, text="(x: -, y: -)")
        self.btn_hide_bottom = ttk.Button(self.statusbar, text="✖", width=3, command=lambda: self._toggle_slot("bottom"))
        self.prog.pack(side="left", padx=(6,2), pady=2)
        self.lbl_status.pack(side="left", padx=(6,2))
        self.lbl_coord.pack(side="left", padx=(12,2))
        self.btn_show_log.pack(side="right", padx=4)
        self.btn_hide_bottom.pack(side="right", padx=(2,6))

        # split: viewer | tools | right
        self.hsplit = ttk.Panedwindow(self, orient="horizontal"); self.hsplit.pack(fill="both", expand=True)

        # --- viewer + bottom
        self.left_view = ttk.Frame(self.hsplit); self.hsplit.add(self.left_view, weight=1)
        self.vsplit = ttk.Panedwindow(self.left_view, orient="vertical"); self.vsplit.pack(fill="both", expand=True)

        # viewer host (grid)
        self.viewer_host = ttk.Frame(self.vsplit); self.vsplit.add(self.viewer_host, weight=10)

        # toolbar
        toolbar = ttk.Frame(self.viewer_host); toolbar.grid(row=0, column=0, columnspan=3, sticky="ew")
        ttk.Button(toolbar, text="Fit", command=self.on_zoom_fit).pack(side="left", padx=(6,0), pady=3)
        ttk.Button(toolbar, text="100%", command=self.on_zoom_100).pack(side="left", padx=(6,0))
        ttk.Button(toolbar, text="Center", command=self.on_center).pack(side="left", padx=(6,0))

        # TOOLBOX
        self.toolbox = _Toolbox(
            toolbar,
            on_mode_change=self._on_tool_mode_change,
            on_toggle=self._on_tool_toggle
        )
        self.toolbox.pack(side="left", padx=(12,0))

        ttk.Label(toolbar, text="Zoom").pack(side="left", padx=(12,4))
        self.zoom_scale = ttk.Scale(toolbar, from_=0.1, to=8.0, value=1.0, command=self._on_zoom_slider)
        self.zoom_scale.pack(side="left", fill="x", expand=True, padx=(0,6))

        # rulers
        self.ruler_top = _Ruler(self.viewer_host, orientation="x", height=26); self.ruler_top.grid(row=1, column=1, sticky="ew")
        self.ruler_left = _Ruler(self.viewer_host, orientation="y", width=28);  self.ruler_left.grid(row=2, column=0, sticky="ns")

        # właściwy canvas
        if ImageCanvas is not None:
            self.canvas = ImageCanvas(self.viewer_host)
        else:
            self.canvas = tk.Canvas(self.viewer_host, bg="#101010", highlightthickness=0)
        self.canvas.grid(row=2, column=1, sticky="nsew")

        # dynamiczny krzyż – wąskie canvasy nad obrazem (1 px)
        self._init_crosshair_overlays()

        # overlay z osiami i współrzędnymi (pseudo-alpha + odsunięcie)
        self._init_axes_overlay()

        # reaguj na rozmiar
        self.canvas.bind("<Configure>", lambda _e: self._on_canvas_configure())

        # grid weights
        self.viewer_host.columnconfigure(1, weight=1)
        self.viewer_host.rowconfigure(2, weight=1)

        # bottom tabs
        self.bottom_host = ttk.Frame(self.vsplit); self.vsplit.add(self.bottom_host, weight=0)
        self.bottom_tabs = ttk.Notebook(self.bottom_host); self.bottom_tabs.pack(fill="both", expand=True)

        hud_frame = ttk.Frame(self.bottom_tabs); self.bottom_tabs.add(hud_frame, text="HUD")
        self.hud = Hud(hud_frame) if Hud is not None else None
        if self.hud and hasattr(self.hud, "pack"): self.hud.pack(fill="both", expand=True)

        if GraphView is not None:
            gv_frame = ttk.Frame(self.bottom_tabs); self.bottom_tabs.add(gv_frame, text="Graph")
            self.graph = GraphView(gv_frame); self.graph.pack(fill="both", expand=True)

        if MosaicMini is not None:
            mv_frame = ttk.Frame(self.bottom_tabs); self.bottom_tabs.add(mv_frame, text="Mosaic")
            self.mosaic = MosaicMini(mv_frame); self.mosaic.pack(fill="both", expand=True)

        tech_frame = ttk.Frame(self.bottom_tabs); self.bottom_tabs.add(tech_frame, text="Tech")
        self.tech_log = tk.Text(tech_frame, height=8, bg="#141414", fg="#e6e6e6", insertbackground="#e6e6e6")
        self.tech_log.pack(fill="both", expand=True)
        self._log("--- app started ---")

        # left tools (transformy widoku)
        self.left_tools = ttk.Frame(self.hsplit)
        self._build_left_tools(self.left_tools)
        self.hsplit.add(self.left_tools, weight=0)

        # right side panels
        self.right = ttk.Frame(self.hsplit); self.hsplit.add(self.right, weight=0)
        self.tabs = ttk.Notebook(self.right); self.tabs.pack(fill="both", expand=True)

        self.tab_global = ttk.Frame(self.tabs); self.tabs.add(self.tab_global, text="Global")
        self._build_global_tab(self.tab_global)

        self.tab_filter = ttk.Frame(self.tabs); self.tabs.add(self.tab_filter, text="Filter")
        self._build_filter_tab(self.tab_filter)

        self.tab_presets = ttk.Frame(self.tabs); self.tabs.add(self.tab_presets, text="Presets")
        self._build_presets_tab(self.tab_presets)

        self._apply_view_overlays()

    def _build_left_tools(self, p: ttk.Frame):
        p.columnconfigure(0, weight=1)
        ttk.Label(p, text="Tools", font=("", 10, "bold")).grid(row=0, column=0, sticky="ew", padx=6, pady=(6,4))
        ttk.Separator(p).grid(row=1, column=0, sticky="ew", padx=6, pady=6)
        ttk.Label(p, text="View Transform (preview):").grid(row=2, column=0, sticky="w", padx=8)
        self._mk_slider(p, "Roll (Z°)", self.var_roll, -180, 180, row=3)
        self._mk_slider(p, "Pitch (°)", self.var_pitch, -45, 45, row=4)
        self._mk_slider(p, "Yaw (°)",   self.var_yaw,   -45, 45, row=5)
        btns = ttk.Frame(p); btns.grid(row=6, column=0, sticky="ew", padx=8, pady=(2,8))
        ttk.Button(btns, text="Reset view", command=self._reset_view_transform).pack(side="left", padx=(0,6))
        ttk.Button(btns, text="Show Tech log", command=self._show_tech_log).pack(side="left")
        for v in (self.var_roll, self.var_pitch, self.var_yaw):
            v.trace_add("write", lambda *_: self._refresh_display())
        ttk.Separator(p).grid(row=7, column=0, sticky="ew", padx=6, pady=6)
        ttk.Button(p, text="Hide Tools (F8)", command=self._toggle_left_tools).grid(row=8, column=0, sticky="ew", padx=8, pady=(0,6))

    def _mk_slider(self, parent, label, var, vmin, vmax, row):
        fr = ttk.Frame(parent); fr.grid(row=row, column=0, sticky="ew", padx=8, pady=2)
        ttk.Label(fr, text=label).pack(side="left")
        ttk.Scale(fr, from_=vmin, to=vmax, variable=var).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Entry(fr, textvariable=var, width=6).pack(side="left")

    def _toggle_left_tools(self):
        try:
            panes = self.hsplit.panes()
            if str(self.left_tools) in panes:
                self.hsplit.forget(self.left_tools)
            else:
                self.hsplit.insert(1, self.left_tools, weight=0)
        except Exception:
            pass

    def _select_bottom_tab(self, name: str):
        try:
            for i in range(self.bottom_tabs.index("end")):
                if self.bottom_tabs.tab(i, "text") == name:
                    self.bottom_tabs.select(i)
                    break
        except Exception:
            pass

    def _show_tech_log(self):
        self._select_bottom_tab("Tech")

    # ---------- GLOBAL TAB ----------
    def _build_global_tab(self, p: ttk.Frame):
        row = 0
        ttk.Label(p, text="Masks").grid(row=row, column=0, sticky="w", padx=6, pady=(6, 0)); row += 1
        mask_box = ttk.Frame(p); mask_box.grid(row=row, column=0, sticky="ew", padx=6, pady=4)
        p.columnconfigure(0, weight=1)
        if MaskChooser is not None:
            self.mask_chooser = MaskChooser(
                mask_box,
                get_mask_keys=lambda: list(self.last_ctx.masks.keys()),
                on_add_mask=self._on_add_mask_from_file,
                on_select=lambda key: None,
            )
            self.mask_chooser.pack(fill="x")
        else:
            ttk.Label(mask_box, text="(MaskChooser unavailable)").pack()

    # ---------- FILTER TAB ----------
    def _build_filter_tab(self, p: ttk.Frame):
        top = ttk.Frame(p); top.pack(fill="x", pady=(6, 4))
        ttk.Label(top, text="Filter:").pack(side="left", padx=(6, 4))
        self.cmb_filter = ttk.Combobox(top, values=sorted(registry_available()), state="readonly")
        self.cmb_filter.pack(side="left", fill="x", expand=True)
        if self.cmb_filter["values"]:
            self.cmb_filter.current(0)
        self.btn_apply_filter = ttk.Button(top, text="Apply filter", command=self.on_apply_filter)
        self.btn_apply_filter.pack(side="left", padx=(6, 6))
        self.filter_panel_host = ttk.Frame(p)
        self.filter_panel_host.pack(fill="x", padx=6, pady=(0, 6))
        self._mount_panel_for(self.cmb_filter.get())
        self.cmb_filter.bind("<<ComboboxSelected>>", lambda _e=None: self._mount_panel_for(self.cmb_filter.get()))

    def _mount_panel_for(self, filter_name: str):
        for w in self.filter_panel_host.winfo_children():
            w.destroy()
        self._filter_panel = None
        self._filter_params = None
        self._fallback_form = None

        panel = None
        try:
            mod_name = f"glitchlab.gui.panels.panel_{filter_name}"
            mod = importlib.import_module(mod_name)
            pick = None
            for attr in dir(mod):
                obj = getattr(mod, attr)
                if isinstance(obj, type) and issubclass(obj, ttk.Frame) and attr.lower().endswith("panel"):
                    pick = obj; break
            if pick:
                ctx = PanelContext(
                    filter_name=filter_name,
                    defaults={}, params={},
                    on_change=lambda params: self._on_filter_params_changed(params),
                    cache_ref=self.last_ctx.cache,
                    get_mask_keys=lambda: list(self.last_ctx.masks.keys()),
                )
                panel = pick(self.filter_panel_host, ctx=ctx)
        except Exception as e:
            print(f"[filter-panel] load failed for {filter_name}: {e}")

        if panel is None:
            form = ParamForm(self.filter_panel_host, get_filter_callable=lambda nm: registry_get(nm))
            form.build_for(filter_name)
            panel = form
            self._fallback_form = form

        panel.pack(fill="x")
        self._filter_panel = panel

    def _on_filter_params_changed(self, params: dict):
        self._filter_params = dict(params or {})

    # ---------- PRESETS TAB ----------
    def _build_presets_tab(self, p: ttk.Frame):
        if PresetManager is None:
            ttk.Label(p, text="(PresetManager unavailable)").pack(fill="both", expand=True)
            return

        # GÓRNY 'Edit JSON…' usunięty – dolny edytor z PresetManagera wystarcza
        bar_top = ttk.Frame(p); bar_top.pack(fill="x", pady=(6, 2))
        ttk.Button(bar_top, text="Change…", command=self.on_choose_preset_dir).pack(side="left", padx=(6, 0))
        self.lbl_preset_dir = ttk.Label(bar_top, text=f"Preset folder: {_win_path(self.preset_dir)}")
        self.lbl_preset_dir.pack(side="left", padx=(6, 0))

        def _get_cfg():
            self.preset_cfg["__preset_dir"] = str(self.preset_dir)
            return self.preset_cfg

        def _set_cfg(cfg: dict):
            self.preset_cfg = dict(cfg or {})
            try:
                d = self.preset_cfg.get("__preset_dir")
                if d: self.preset_dir = Path(d)
            except Exception:
                pass

        def _apply():
            self.on_apply_preset_steps()

        def _get_filters():
            return sorted(registry_available())

        def _get_current_step():
            name = (self.cmb_filter.get() or "").strip()
            if not name:
                return {}
            params: dict = {}
            if self._filter_params is not None:
                params = dict(self._filter_params)
            elif self._fallback_form is not None:
                try: params = dict(self._fallback_form.values())
                except Exception: params = {}
            return {"name": name, "params": params}

        try:
            self.preset_mgr = PresetManager(p, _get_cfg, _set_cfg, _apply, _get_filters, _get_current_step)
        except TypeError:
            self.preset_mgr = PresetManager(p, _get_cfg, _set_cfg, _apply, _get_filters)

        self.preset_mgr.pack(fill="both", expand=True)
        self._sync_preset_dir_to_mgr()

    def _sync_preset_dir_to_mgr(self):
        if not getattr(self, "preset_mgr", None):
            return
        d = str(self.preset_dir)
        ok = False
        for name in ("set_preset_dir", "set_dir", "set_folder", "set_root", "set_root_dir", "set_base_dir"):
            fn = getattr(self.preset_mgr, name, None)
            if callable(fn):
                try:
                    fn(d); ok = True; break
                except Exception:
                    pass
        if not ok:
            try:
                if hasattr(self.preset_mgr, "preset_dir"):
                    self.preset_mgr.preset_dir = Path(d)
            except Exception:
                pass
        for refresh_name in ("refresh", "reload", "_refresh_from_cfg", "rebuild"):
            fn = getattr(self.preset_mgr, refresh_name, None)
            if callable(fn):
                try: fn()
                except Exception: pass

    # ---------- akcje / I/O ----------
    def on_open(self):
        if self.is_busy: return
        path = filedialog.askopenfilename(
            title="Open image",
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.webp;*.bmp"), ("All files", "*.*")]
        )
        if not path: return
        try:
            from PIL import Image
            im = Image.open(path).convert("RGB")
            a = np.array(im, dtype=np.uint8)
            self.img_u8 = a
            self.last_ctx.cache.clear()
            self._refresh_display()
            self._update_hud()
            self._log(f"Opened: {path}")
        except Exception as e:
            messagebox.showerror("Open image", str(e))

    def on_save(self):
        if self.is_busy: return
        if self._display_u8 is None:
            messagebox.showinfo("Save", "Brak obrazu."); return
        path = filedialog.asksaveasfilename(
            title="Save image as",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg;*.jpeg"), ("All files", "*.*")]
        )
        if not path: return
        try:
            from PIL import Image
            Image.fromarray(self._display_u8, "RGB").save(path)
            self._log(f"Saved: {path}")
        except Exception as e:
            messagebox.showerror("Save image", str(e))

    def on_choose_preset_dir(self):
        if self.is_busy: return
        d = filedialog.askdirectory(title="Choose preset folder", initialdir=str(self.preset_dir))
        if not d: return
        self.preset_dir = Path(d)
        self.preset_cfg["__preset_dir"] = str(self.preset_dir)
        try:
            self.lbl_preset_dir.config(text=f"Preset folder: {_win_path(self.preset_dir)}")
        except Exception:
            pass
        self._sync_preset_dir_to_mgr()
        self._log(f"Preset dir: {self.preset_dir}")

    def on_zoom_100(self):
        if hasattr(self.canvas, "set_zoom"):
            try:
                self.canvas.set_zoom(1.0)
                self.zoom_scale.set(1.0)
                self._update_rulers()
                self._redraw_crosshair_lines()
                self._position_axes_overlay()
            except Exception:
                pass

    def on_zoom_fit(self):
        if hasattr(self.canvas, "fit"):
            try:
                self.canvas.fit()
                if hasattr(self.canvas, "get_zoom"):
                    self.zoom_scale.set(float(self.canvas.get_zoom()))
                self._update_rulers()
                self._redraw_crosshair_lines()
                self._position_axes_overlay()
            except Exception:
                pass

    def on_center(self):
        if hasattr(self.canvas, "center"):
            try: self.canvas.center()
            except Exception: pass

    def _on_zoom_slider(self, val):
        try:
            z = float(val)
            if hasattr(self.canvas, "set_zoom"):
                self.canvas.set_zoom(z)
            self._update_rulers()
            self._redraw_crosshair_lines()
            self._position_axes_overlay()
        except Exception:
            pass

    def on_about(self):
        messagebox.showinfo(
            "About GlitchLab",
            "GlitchLab — controlled glitch for analysis\nGUI prototype\n© D2J3 aka Cha0s"
        )

    def _toggle_slot(self, which: str):
        try:
            if which == "bottom":
                panes = self.vsplit.panes()
                if str(self.bottom_host) in panes:
                    self.vsplit.forget(self.bottom_host)
                else:
                    self.vsplit.add(self.bottom_host, weight=0)
            elif which == "right":
                panes = self.hsplit.panes()
                if str(self.right) in panes:
                    self.hsplit.forget(self.right)
                else:
                    self.hsplit.add(self.right, weight=0)
        except Exception as e:
            print("toggle error:", e)

    # ---------- maski ----------
    def _on_add_mask_from_file(self, key: str | None, arr: np.ndarray | None):
        if key and isinstance(arr, np.ndarray):
            self.last_ctx.masks[key] = arr.astype(np.float32)
            try:
                self.last_ctx.cache["cfg/masks/keys"] = list(self.last_ctx.masks.keys())
            except Exception:
                pass
            self._update_hud()
            self._log(f"Mask added: {key}")

    # ---------- pipeline / kontekst ----------
    def _build_ctx_for_run(self):
        ctx = type("Ctx", (), {})()
        H, W = (self.img_u8.shape[0], self.img_u8.shape[1]) if isinstance(self.img_u8, np.ndarray) else (0, 0)
        ctx.rng = self.last_ctx.rng
        if self.last_ctx.amplitude is None and H > 0 and W > 0:
            self.last_ctx.amplitude = np.ones((H, W), dtype=np.float32)
        ctx.amplitude = self.last_ctx.amplitude
        ctx.masks = self.last_ctx.masks
        ctx.cache = {}
        try:
            ctx.cache["cfg/masks/keys"] = list(self.last_ctx.masks.keys())
        except Exception:
            ctx.cache["cfg/masks/keys"] = []
        ctx.meta = {}
        return ctx

    def _collect_current_filter_params(self) -> dict:
        if self._filter_params is not None:
            return dict(self._filter_params)
        if self._fallback_form is not None:
            try:
                return dict(self._fallback_form.values())
            except Exception:
                return {}
        return {}

    # ---------- async uruchamianie ----------
    def _set_busy(self, busy: bool, label: str = ""):
        self.is_busy = busy
        try:
            if self.btn_apply_filter: self.btn_apply_filter.config(state=("disabled" if busy else "normal"))
            if self.cmb_filter: self.cmb_filter.config(state=("disabled" if busy else "readonly"))
            for m in ("file", "view", "settings", "help"):
                menu = self.menu_objects.get(m)
                if not menu: continue
                end = menu.index("end")
                if end is None:
                    continue
                for i in range(end + 1):
                    try:
                        menu.entryconfig(i, state=("disabled" if busy else "normal"))
                    except Exception:
                        pass
            self.lbl_status.config(text=(label if label else ("Working…" if busy else "Ready")))
            if busy: self.prog.start(80)
            else:    self.prog.stop()
        except Exception:
            pass

    def _run_async(self, label: str, func, on_done):
        if self.is_busy: return
        self._cancel_requested = False
        self._set_busy(True, label)
        self._log(f"[start] {label}")
        t0 = time.time()

        def worker():
            err = None
            result = None
            try:
                result = func()
            except Exception as e:
                err = e
            self.root.after(0, lambda: self._on_worker_done(label, t0, result, err, on_done))

        self._op_thread = threading.Thread(target=worker, daemon=True)
        self._op_thread.start()

    def _on_worker_done(self, label: str, t0: float, result, err: Exception | None, on_done):
        dt = time.time() - t0
        if err is not None:
            self._log(f"[fail] {label} — {err} ({dt:.2f}s)")
            messagebox.showerror("Operation failed", str(err))
        else:
            self._log(f"[ok]   {label} ({dt:.2f}s)")
        try:
            if callable(on_done):
                on_done(result, err)
        finally:
            self._set_busy(False)

    # ---------- działania ----------
    def on_apply_filter(self):
        if self.img_u8 is None:
            messagebox.showinfo("Filter", "Najpierw otwórz obraz."); return
        if self.is_busy:
            return
        name = (self.cmb_filter.get() or "").strip()
        if not name:
            messagebox.showinfo("Filter", "Wybierz filtr."); return

        params = self._collect_current_filter_params()
        steps = [{"name": name, "params": params}]
        ctx = self._build_ctx_for_run()

        def job():
            return apply_pipeline(self.img_u8, ctx, steps, fail_fast=True, metrics=True, debug_log=[])

        def done(out, err):
            if err is not None:
                return
            self.img_u8 = np_to_u8(out)
            self.last_ctx = ctx
            self._refresh_display()
            self._update_hud()
            self._push_history_if_possible({"name": name, "params": params})

        self._run_async(f"Apply filter: {name}", job, done)

    def on_apply_preset_steps(self):
        if self.img_u8 is None:
            messagebox.showinfo("Preset", "Najpierw otwórz obraz."); return
        if self.is_busy:
            return
        steps = list(self.preset_cfg.get("steps") or [])
        if not steps:
            messagebox.showinfo("Preset", "Preset nie ma kroków."); return
        ctx = self._build_ctx_for_run()

        def job():
            return apply_pipeline(self.img_u8, ctx, steps, fail_fast=True, metrics=True, debug_log=[])

        def done(out, err):
            if err is not None:
                return
            self.img_u8 = np_to_u8(out)
            self.last_ctx = ctx
            self._refresh_display()
            self._update_hud()

        self._run_async("Apply preset steps", job, done)

    def _push_history_if_possible(self, step: dict):
        pm = getattr(self, "preset_mgr", None)
        if not pm:
            return
        for name in ("push_history", "add_history", "remember_step", "history_add"):
            fn = getattr(pm, name, None)
            if callable(fn):
                try:
                    fn(dict(step))
                except Exception:
                    pass
                break

    # ---------- wyświetlanie / transform widoku ----------
    def _refresh_display(self):
        base = self.img_u8
        if base is None:
            return
        out = base
        roll = float(self.var_roll.get())
        pitch = float(self.var_pitch.get())
        yaw = float(self.var_yaw.get())
        if any(abs(v) > 1e-6 for v in (roll, pitch, yaw)):
            out = self._apply_view_transform(out, roll, pitch, yaw)
        self._display_u8 = np_to_u8(out)
        self._show_on_canvas(self._display_u8)

    def _apply_view_transform(self, img_u8: np.ndarray, roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
        try:
            from PIL import Image
        except Exception:
            return img_u8
        im = Image.fromarray(img_u8, "RGB")
        kx = math.tan(math.radians(yaw_deg)) * 0.5
        ky = math.tan(math.radians(pitch_deg)) * 0.5
        a, b, c = (1.0, -kx, 0.0)
        d, e, f = (-ky, 1.0, 0.0)
        im = im.transform(im.size, Image.AFFINE, (a, b, c, d, e, f), resample=Image.BILINEAR)
        if abs(roll_deg) > 1e-6:
            im = im.rotate(roll_deg, resample=Image.BILINEAR, expand=True, fillcolor=(16,16,16))
        return np.array(im, dtype=np.uint8)

    def _get_canvas_bg(self) -> str:
        try:
            return self.canvas.cget("bg")
        except Exception:
            return "#101010"

    def _show_on_canvas(self, img_u8: np.ndarray):
        if hasattr(self.canvas, "set_image"):
            try:
                self.canvas.set_image(np_to_u8(img_u8))
                if hasattr(self.canvas, "get_zoom"):
                    self.zoom_scale.set(float(self.canvas.get_zoom()))
            except Exception as e:
                print("canvas.set_image error:", e)
        else:
            try:
                from PIL import Image, ImageTk
                self._tk_img = ImageTk.PhotoImage(Image.fromarray(np_to_u8(img_u8)))
                self.canvas.delete("all")
                self.canvas.create_image(0, 0, image=self._tk_img, anchor="nw")
            except Exception:
                pass
        self._update_rulers()
        self._redraw_crosshair_lines()
        self._position_axes_overlay()

    def _update_rulers(self):
        if not (self.ruler_top and self.ruler_left):
            return
        try:
            H = int(self._display_u8.shape[0]) if isinstance(self._display_u8, np.ndarray) else (
                int(self.img_u8.shape[0]) if isinstance(self.img_u8, np.ndarray) else 0
            )
            W = int(self._display_u8.shape[1]) if isinstance(self._display_u8, np.ndarray) else (
                int(self.img_u8.shape[1]) if isinstance(self.img_u8, np.ndarray) else 0
            )
            zoom = float(self.zoom_scale.get())
            self.ruler_top.set_zoom_and_size(zoom, W)
            self.ruler_left.set_zoom_and_size(zoom, H)
            self.ruler_top.update_marker_view_px(self._mouse_view_x)
            self.ruler_left.update_marker_view_px(self._mouse_view_y)
        except Exception:
            pass

    # ---------- dynamiczny krzyż ----------
    def _init_crosshair_overlays(self):
        bg = self._get_canvas_bg()
        self.cross_h = tk.Canvas(self.viewer_host, height=1, bg=bg, highlightthickness=0, bd=0, takefocus=0)
        self.cross_h.place(in_=self.canvas, x=0, y=0, relwidth=1, height=1)

        self.cross_v = tk.Canvas(self.viewer_host, width=1, bg=bg, highlightthickness=0, bd=0, takefocus=0)
        self.cross_v.place(in_=self.canvas, x=0, y=0, width=1, relheight=1)

        self._redraw_crosshair_lines()
        self._update_crosshair_visibility()

    def _redraw_crosshair_lines(self):
        try:
            ch = self.cross_h
            cw = max(1, int(ch.winfo_width() or self.canvas.winfo_width() or 1))
            ch.delete("all")
            ch.create_line(0, 0, cw, 0, fill="#e8e8e8", dash=(6, 4))

            cv = self.cross_v
            chh = max(1, int(cv.winfo_height() or self.canvas.winfo_height() or 1))
            cv.delete("all")
            cv.create_line(0, 0, 0, chh, fill="#e8e8e8", dash=(6, 4))
        except Exception:
            pass

    def _update_crosshair_visibility(self):
        show = bool(self.var_cross.get())
        try:
            if show:
                self.cross_h.place(in_=self.canvas, x=0, y=self._mouse_view_y, relwidth=1, height=1)
                self.cross_v.place(in_=self.canvas, x=self._mouse_view_x, y=0, width=1, relheight=1)
                self.root.after_idle(self._redraw_crosshair_lines)
            else:
                self.cross_h.place_forget()
                self.cross_v.place_forget()
        except Exception:
            pass

    # ---------- overlay osi + współrzędne ----------
    def _init_axes_overlay(self):
        # tło overlay = tło canvas (brak crasha, brak białego kwadratu)
        self.axes = tk.Canvas(self.viewer_host, width=200, height=110,
                              bg=self._get_canvas_bg(), highlightthickness=0, bd=0)
        self._draw_axes_overlay()
        self._position_axes_overlay()

    def _draw_axes_overlay(self):
        c = self.axes
        c.delete("all")
        pad = 8
        w = int(c.cget("width")); h = int(c.cget("height"))
        # „półprzezroczysta” płytka (stipple jako pseudo-alpha)
        c.create_rectangle(0, 0, w, h, fill="#000000", outline="", stipple="gray50")

        # osie z marginesami i czytelnym odstępem od krawędzi
        c.create_line(pad+18, h-pad-28, w-pad-30, h-pad-28, fill="#9fbfff", width=3, arrow="last")
        c.create_text(w-pad-20, h-pad-28, anchor="w", text="X→", fill="#d0e0ff", font=("", 12, "bold"))

        c.create_line(pad+18, h-pad-22, pad+18, pad+22, fill="#9fbfff", width=3, arrow="last")
        c.create_text(pad+18, pad+14, anchor="s", text="Y↑", fill="#d0e0ff", font=("", 12, "bold"))

        self.axes_coord_id = c.create_text(w//2, h-pad-8, anchor="s",
                                           text="(x: -, y: -)", fill="#ffffff", font=("", 11, "bold"))

    def _position_axes_overlay(self):
        try:
            # prawa-dolna ćwiartka, minimalny offset od krawędzi obrazu
            self.axes.place(in_=self.canvas, relx=1.0, rely=1.0, x=-12, y=-12, anchor="se")
        except Exception:
            pass

    def _update_axes_coords(self, ix: int, iy: int):
        try:
            if self.axes and self.axes_coord_id:
                self.axes.itemconfigure(self.axes_coord_id, text=f"(x: {ix}, y: {iy})")
        except Exception:
            pass

    # ---------- toolbox callbacks ----------
    def _on_tool_mode_change(self, mode: str):
        if mode == "hand":
            try: self.canvas.configure(cursor="fleur")
            except Exception: pass
        elif mode == "zoom":
            try: self.canvas.configure(cursor="tcross")
            except Exception: pass
        elif mode == "pick":
            try: self.canvas.configure(cursor="circle")
            except Exception: pass
        elif mode == "measure":
            try: self.canvas.configure(cursor="crosshair")
            except Exception: pass

    def _on_tool_toggle(self, key: str, state: bool):
        if key == "cross":
            self.var_cross.set(bool(state))
            self._update_crosshair_visibility()
        elif key == "rulers":
            self.var_rulers.set(bool(state))
            self._apply_view_overlays()

    def _apply_view_overlays(self):
        show_r = bool(self.var_rulers.get())
        try:
            self.ruler_top.grid() if show_r else self.ruler_top.grid_remove()
            self.ruler_left.grid() if show_r else self.ruler_left.grid_remove()
        except Exception:
            pass
        self._update_crosshair_visibility()
        if self.toolbox:
            self.toolbox.set_rulers(self.var_rulers.get())
            self.toolbox.set_cross(self.var_cross.get())

    # ---------- obsługa rozmiaru / myszy ----------
    def _on_canvas_configure(self):
        self._redraw_crosshair_lines()
        self._update_rulers()
        self._position_axes_overlay()

    def _on_any_motion(self, _e):
        try:
            rx, ry = self.canvas.winfo_rootx(), self.canvas.winfo_rooty()
            cx, cy = self.root.winfo_pointerx() - rx, self.root.winfo_pointery() - ry
            inside = (0 <= cx < self.canvas.winfo_width() and 0 <= cy < self.canvas.winfo_height())
        except Exception:
            inside = False
        if inside:
            self._on_canvas_motion_xy(int(cx), int(cy))
        else:
            self._on_canvas_leave()

    def _on_canvas_leave(self):
        try:
            self.cross_h.place_forget()
            self.cross_v.place_forget()
        except Exception:
            pass
        self.lbl_coord.config(text="(x: -, y: -)")
        self._update_axes_coords(-1, -1)

    def _on_canvas_motion_xy(self, x: int, y: int):
        self._mouse_view_x = int(x)
        self._mouse_view_y = int(y)
        if bool(self.var_cross.get()):
            try:
                self.cross_h.place_configure(y=self._mouse_view_y)
                self.cross_v.place_configure(x=self._mouse_view_x)
            except Exception:
                pass
        self._update_rulers()
        ix, iy, _ = self._view_to_image_coords(x, y)
        self.lbl_coord.config(text=f"(x: {ix}, y: {iy})")
        self._update_axes_coords(ix, iy)

    def _view_to_image_coords(self, x: int, y: int) -> tuple[int,int,str]:
        try:
            if hasattr(self.canvas, "screen_to_image") and callable(self.canvas.screen_to_image):
                ix, iy = self.canvas.screen_to_image(x, y); return int(ix), int(iy), "[img]"
            if hasattr(self.canvas, "to_image") and callable(self.canvas.to_image):
                ix, iy = self.canvas.to_image(x, y); return int(ix), int(iy), "[img]"
            if hasattr(self.canvas, "view_to_image") and callable(self.canvas.view_to_image):
                ix, iy = self.canvas.view_to_image(x, y); return int(ix), int(iy), "[img]"
        except Exception:
            pass
        try:
            zoom = float(self.zoom_scale.get())
            ix = int(round(x / max(zoom, 1e-6)))
            iy = int(round(y / max(zoom, 1e-6)))
            return ix, iy, "[view≈]"
        except Exception:
            return 0, 0, "[?]"

    # ---------- HUD ----------
    def _update_hud(self):
        if not self.hud: return
        try:
            self.hud.render_from_cache(self.last_ctx)
        except Exception as e:
            print("HUD render error:", e)

    # ---------- log ----------
    def _log(self, msg: str):
        try:
            self.tech_log.insert("end", msg + "\n")
            self.tech_log.see("end")
        except Exception:
            pass

    # ---------- reset / tools ----------
    def _reset_view_transform(self):
        self.var_roll.set(0.0)
        self.var_pitch.set(0.0)
        self.var_yaw.set(0.0)
        self._refresh_display()


# --- bootstrap ---
def main():
    root = tk.Tk()
    app = App(root)
    root.geometry("1380x860")
    root.mainloop()

if __name__ == "__main__":
    main()
