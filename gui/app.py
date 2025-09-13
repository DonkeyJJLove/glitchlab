# glitchlab/gui/app.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import importlib
from pathlib import Path
import threading
import time
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
        # fallback no-op
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
    """
    Domyślny katalog presetów:
    1) ENV: GLITCHLAB_PRESETS_DIR
    2) Twardo wskazane przez Ciebie:
       C:\\Users\\donke\\PycharmProjects\\glitchlab_project\\glitchlab\\presets
    (jeśli nie istnieje – utworzymy)
    """
    env = os.environ.get("GLITCHLAB_PRESETS_DIR")
    if env and Path(env).exists():
        return Path(env)

    fixed = Path(r"C:\Users\donke\PycharmProjects\glitchlab_project\glitchlab\presets")
    fixed.mkdir(parents=True, exist_ok=True)
    return fixed


# ------ fallback: dynamiczny formularz parametrów ------
class ParamForm(ttk.Frame):
    """
    Prostolinijny formularz parametrów budowany na podstawie schematu
    (jeśli filtr go udostępnia) albo sygnatury funkcji.
    """
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
        # 1) struktury konwencyjne
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
        # 2) sygnatura funkcji
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
    """Prosta linijka – górna (orientation='x') lub lewa ('y')."""
    def __init__(self, master, orientation: str = "x", height=20, width=24):
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

    def set_zoom_and_size(self, zoom: float, length_px: int):
        self.zoom = max(zoom, 1e-6)
        self.length_px = max(1, int(length_px))
        self._redraw()

    def _redraw(self):
        c = self.canvas
        c.delete("all")
        if self.orientation == "x":
            w = int(c.winfo_width() or 0)
            if w <= 0: w = self.length_px
            step = max(10, int(50 * self.zoom))
            for x in range(0, w, step):
                c.create_line(x, 0, x, 10, fill="#aaaaaa")
                c.create_text(x+2, 12, anchor="nw", text=str(int(x/self.zoom)), fill="#cccccc", font=("", 7))
        else:
            h = int(c.winfo_height() or 0)
            if h <= 0: h = self.length_px
            step = max(10, int(50 * self.zoom))
            for y in range(0, h, step):
                c.create_line(0, y, 10, y, fill="#aaaaaa")
                c.create_text(12, y+2, anchor="nw", text=str(int(y/self.zoom)), fill="#cccccc", font=("", 7))


# ------- główna aplikacja -------
class App(ttk.Frame):
    def __init__(self, root: tk.Tk):
        super().__init__(root)
        self.root = root
        self.root.title("GlitchLab")
        self.pack(fill="both", expand=True)

        # stan
        self.img_u8: np.ndarray | None = None
        self.last_ctx = type("Ctx", (), {})()
        self.last_ctx.rng = np.random.default_rng(7)
        self.last_ctx.amplitude = None
        self.last_ctx.masks = {}
        self.last_ctx.cache = {}
        self.last_ctx.meta = {}

        # busy / operacje
        self.is_busy = False
        self._op_thread: threading.Thread | None = None
        self._cancel_requested = False  # (placeholder; pipeline nie wspiera cancel)

        # katalog presetów
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

        # referencje do elementów GUI
        self.btn_apply_filter: ttk.Button | None = None
        self.cmb_filter: ttk.Combobox | None = None
        self.menu_objects: dict[str, tk.Menu] = {}

        # viewer i rulers
        self.ruler_top: _Ruler | None = None
        self.ruler_left: _Ruler | None = None
        self.var_rulers = tk.BooleanVar(value=True)
        self.var_cross = tk.BooleanVar(value=True)

        # lewy panel Tools
        self.left_tools: ttk.Frame | None = None
        self.left_tools_visible = tk.BooleanVar(value=True)

        self._build_menu()
        self._build_layout(root)

        # skróty
        self.root.bind("<Control-o>", lambda e: self.on_open())
        self.root.bind("<Control-s>", lambda e: self.on_save())
        self.root.bind("<F10>", lambda e: self._toggle_slot("bottom"))
        self.root.bind("<F9>",  lambda e: self._toggle_slot("right"))
        self.root.bind("<F8>",  lambda e: self._toggle_left_tools())

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
        m_view.add_checkbutton(label="Rulers", variable=self.var_rulers, command=self._apply_view_overlays)
        m_view.add_checkbutton(label="Crosshair", variable=self.var_cross, command=self._apply_view_overlays)
        m_view.add_separator()
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
        # pasek techniczny na samym dole
        self.statusbar = ttk.Frame(self); self.statusbar.pack(side="bottom", fill="x")
        self.prog = ttk.Progressbar(self.statusbar, mode="indeterminate", length=140)
        self.btn_show_log = ttk.Button(self.statusbar, text="Show log", command=self._show_tech_log)
        self.lbl_status = ttk.Label(self.statusbar, text="Ready")
        self.btn_hide_bottom = ttk.Button(self.statusbar, text="✖", width=3, command=lambda: self._toggle_slot("bottom"))
        self.prog.pack(side="left", padx=(6,2), pady=2)
        self.lbl_status.pack(side="left", padx=(6,2))
        self.btn_show_log.pack(side="right", padx=4)
        self.btn_hide_bottom.pack(side="right", padx=(2,6))

        # split: lewe (viewer) | lewy panel (tools) | prawe (panele)
        self.hsplit = ttk.Panedwindow(self, orient="horizontal"); self.hsplit.pack(fill="both", expand=True)

        # --- viewer + dół
        self.left_view = ttk.Frame(self.hsplit); self.hsplit.add(self.left_view, weight=1)

        self.vsplit = ttk.Panedwindow(self.left_view, orient="vertical"); self.vsplit.pack(fill="both", expand=True)

        # viewer host (z gridem pod rulers + canvas)
        self.viewer_host = ttk.Frame(self.vsplit); self.vsplit.add(self.viewer_host, weight=10)

        # górny toolbar nad viewerem
        toolbar = ttk.Frame(self.viewer_host); toolbar.grid(row=0, column=0, columnspan=3, sticky="ew")
        ttk.Button(toolbar, text="Fit", command=self.on_zoom_fit).pack(side="left", padx=(6,0), pady=3)
        ttk.Button(toolbar, text="100%", command=self.on_zoom_100).pack(side="left", padx=(6,0))
        ttk.Button(toolbar, text="Center", command=self.on_center).pack(side="left", padx=(6,0))
        ttk.Checkbutton(toolbar, text="rulers", variable=self.var_rulers,
                        command=self._apply_view_overlays).pack(side="left", padx=(12,0))
        ttk.Checkbutton(toolbar, text="crosshair", variable=self.var_cross,
                        command=self._apply_view_overlays).pack(side="left", padx=(6,0))
        ttk.Label(toolbar, text="Zoom").pack(side="left", padx=(12,4))
        self.zoom_scale = ttk.Scale(toolbar, from_=0.1, to=8.0, value=1.0, command=self._on_zoom_slider)
        self.zoom_scale.pack(side="left", fill="x", expand=True, padx=(0,6))

        # rogi dla rulers
        corner = tk.Canvas(self.viewer_host, width=24, height=20, bg="#2b2b2b", highlightthickness=0)
        corner.grid(row=1, column=0, sticky="nw")
        # rulers
        self.ruler_top = _Ruler(self.viewer_host, orientation="x", height=20); self.ruler_top.grid(row=1, column=1, sticky="ew")
        self.ruler_left = _Ruler(self.viewer_host, orientation="y", width=24);  self.ruler_left.grid(row=2, column=0, sticky="ns")

        # właściwy canvas
        if ImageCanvas is not None:
            self.canvas = ImageCanvas(self.viewer_host)
        else:
            self.canvas = ttk.Label(self.viewer_host, text="(no ImageCanvas)")
        self.canvas.grid(row=2, column=1, sticky="nsew")

        # placeholder dla prawego-dolnego rogu (scroll area etc.)
        spacer = ttk.Frame(self.viewer_host); spacer.grid(row=3, column=0, columnspan=3, sticky="ew")

        # siatka dla viewer_host
        self.viewer_host.columnconfigure(1, weight=1)
        self.viewer_host.rowconfigure(2, weight=1)

        # slot dolny (HUD/diag/log)
        self.bottom_host = ttk.Frame(self.vsplit); self.vsplit.add(self.bottom_host, weight=0)
        self.bottom_tabs = ttk.Notebook(self.bottom_host); self.bottom_tabs.pack(fill="both", expand=True)

        # HUD
        hud_frame = ttk.Frame(self.bottom_tabs); self.bottom_tabs.add(hud_frame, text="HUD")
        self.hud = Hud(hud_frame) if Hud is not None else None
        if self.hud and hasattr(self.hud, "pack"): self.hud.pack(fill="both", expand=True)

        # graf (opcjonalnie)
        if GraphView is not None:
            gv_frame = ttk.Frame(self.bottom_tabs); self.bottom_tabs.add(gv_frame, text="Graph")
            self.graph = GraphView(gv_frame); self.graph.pack(fill="both", expand=True)

        # mozaika (opcjonalnie)
        if MosaicMini is not None:
            mv_frame = ttk.Frame(self.bottom_tabs); self.bottom_tabs.add(mv_frame, text="Mosaic")
            self.mosaic = MosaicMini(mv_frame); self.mosaic.pack(fill="both", expand=True)

        # TECH / LOG tab
        tech_frame = ttk.Frame(self.bottom_tabs); self.bottom_tabs.add(tech_frame, text="Tech")
        self.tech_log = tk.Text(tech_frame, height=8, bg="#141414", fg="#e6e6e6", insertbackground="#e6e6e6")
        self.tech_log.pack(fill="both", expand=True)
        self._log("--- app started ---")

        # --- środkowy: lewy panel Tools (zwijalny)
        self.left_tools = ttk.Frame(self.hsplit)
        self._build_left_tools(self.left_tools)
        self.hsplit.add(self.left_tools, weight=0)

        # --- prawy: panele
        self.right = ttk.Frame(self.hsplit); self.hsplit.add(self.right, weight=0)

        self.tabs = ttk.Notebook(self.right); self.tabs.pack(fill="both", expand=True)

        # Global
        self.tab_global = ttk.Frame(self.tabs); self.tabs.add(self.tab_global, text="Global")
        self._build_global_tab(self.tab_global)

        # Filter
        self.tab_filter = ttk.Frame(self.tabs); self.tabs.add(self.tab_filter, text="Filter")
        self._build_filter_tab(self.tab_filter)

        # Presets
        self.tab_presets = ttk.Frame(self.tabs); self.tabs.add(self.tab_presets, text="Presets")
        self._build_presets_tab(self.tab_presets)

        # na start – ustaw overlays
        self._apply_view_overlays()

    def _build_left_tools(self, p: ttk.Frame):
        """Prosty lewy panel z toggle'ami i przyciskami pomocniczymi."""
        p.columnconfigure(0, weight=1)
        head = ttk.Label(p, text="Tools", font=("", 10, "bold"))
        head.grid(row=0, column=0, sticky="ew", padx=6, pady=(6,4))

        ttk.Checkbutton(p, text="Rulers", variable=self.var_rulers, command=self._apply_view_overlays).grid(row=1, column=0, sticky="w", padx=8)
        ttk.Checkbutton(p, text="Crosshair", variable=self.var_cross, command=self._apply_view_overlays).grid(row=2, column=0, sticky="w", padx=8)

        ttk.Separator(p, orient="horizontal").grid(row=3, column=0, sticky="ew", padx=6, pady=6)
        ttk.Button(p, text="Show HUD", command=lambda: self._select_bottom_tab("HUD")).grid(row=4, column=0, sticky="ew", padx=8, pady=(0,6))
        ttk.Button(p, text="Show Graph", command=lambda: self._select_bottom_tab("Graph")).grid(row=5, column=0, sticky="ew", padx=8, pady=(0,6))
        ttk.Button(p, text="Show Tech log", command=self._show_tech_log).grid(row=6, column=0, sticky="ew", padx=8, pady=(0,6))

        ttk.Separator(p, orient="horizontal").grid(row=7, column=0, sticky="ew", padx=6, pady=6)
        ttk.Button(p, text="Hide Tools (F8)", command=self._toggle_left_tools).grid(row=8, column=0, sticky="ew", padx=8, pady=(0,6))

    def _toggle_left_tools(self):
        try:
            panes = self.hsplit.panes()
            if str(self.left_tools) in panes:
                self.hsplit.forget(self.left_tools)
                self.left_tools_visible.set(False)
            else:
                self.hsplit.insert(1, self.left_tools, weight=0)  # po viewerze
                self.left_tools_visible.set(True)
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

    def _apply_view_overlays(self):
        # rulers
        show_r = bool(self.var_rulers.get())
        try:
            self.ruler_top.grid() if show_r else self.ruler_top.grid_remove()
            self.ruler_left.grid() if show_r else self.ruler_left.grid_remove()
        except Exception:
            pass
        # crosshair
        if hasattr(self.canvas, "set_crosshair"):
            try: self.canvas.set_crosshair(bool(self.var_cross.get()))
            except Exception: pass

    # ---------------- FILTER TAB ----------------
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

    def _build_filter_tab(self, p: ttk.Frame):
        top = ttk.Frame(p); top.pack(fill="x", pady=(6, 4))
        ttk.Label(top, text="Filter:").pack(side="left", padx=(6, 4))

        self.cmb_filter = ttk.Combobox(top, values=sorted(registry_available()), state="readonly")
        self.cmb_filter.pack(side="left", fill="x", expand=True)
        if self.cmb_filter["values"]:
            self.cmb_filter.current(0)

        self.btn_apply_filter = ttk.Button(top, text="Apply filter", command=self.on_apply_filter)
        self.btn_apply_filter.pack(side="left", padx=(6, 6))

        # kontener na panel parametrów
        self.filter_panel_host = ttk.Frame(p)
        self.filter_panel_host.pack(fill="x", padx=6, pady=(0, 6))

        # Pierwszy panel
        self._mount_panel_for(self.cmb_filter.get())

        # zmiana wyboru → montuj panel
        def on_sel(_e=None):
            self._mount_panel_for(self.cmb_filter.get())
        self.cmb_filter.bind("<<ComboboxSelected>>", on_sel)

    def _mount_panel_for(self, filter_name: str):
        # posprzątaj poprzedni
        for w in self.filter_panel_host.winfo_children():
            w.destroy()
        self._filter_panel = None
        self._filter_params = None
        self._fallback_form = None

        panel = None
        try:
            mod_name = f"glitchlab.gui.panels.panel_{filter_name}"
            mod = importlib.import_module(mod_name)
            # wybierz pierwszą klasę Frame zakończoną na "Panel"
            pick = None
            for attr in dir(mod):
                obj = getattr(mod, attr)
                if isinstance(obj, type) and issubclass(obj, ttk.Frame) and attr.lower().endswith("panel"):
                    pick = obj; break
            if pick:
                ctx = PanelContext(
                    filter_name=filter_name,
                    defaults={},
                    params={},
                    on_change=lambda params: self._on_filter_params_changed(params),
                    cache_ref=self.last_ctx.cache,
                    get_mask_keys=lambda: list(self.last_ctx.masks.keys()),  # żywa lista
                )
                panel = pick(self.filter_panel_host, ctx=ctx)
        except Exception as e:
            print(f"[filter-panel] load failed for {filter_name}: {e}")

        if panel is None:
            # fallback – formularz automatyczny
            form = ParamForm(self.filter_panel_host, get_filter_callable=lambda nm: registry_get(nm))
            form.build_for(filter_name)
            panel = form
            self._fallback_form = form

        panel.pack(fill="x")
        self._filter_panel = panel

    def _on_filter_params_changed(self, params: dict):
        self._filter_params = dict(params or {})

    # ---------------- PRESETS TAB ----------------
    def _build_presets_tab(self, p: ttk.Frame):
        if PresetManager is None:
            ttk.Label(p, text="(PresetManager unavailable)").pack(fill="both", expand=True)
            return

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
            """Dla przycisku 'Add current step' w PresetManager (jeśli obsługuje)."""
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

    # --- pomocnicze: synchronizacja z PresetManagerem ---
    def _sync_preset_dir_to_mgr(self):
        if not getattr(self, "preset_mgr", None):
            return
        d = str(self.preset_dir)
        ok = False
        for name in ("set_preset_dir", "set_dir", "set_folder", "set_root", "set_root_dir", "set_base_dir"):
            fn = getattr(self.preset_mgr, name, None)
            if callable(fn):
                try:
                    fn(d)
                    ok = True
                    break
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

    # ---------- akcje ----------
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
            self._show_on_canvas(a)
            self.last_ctx.cache.clear()
            self._update_hud()
            self._log(f"Opened: {path}")
        except Exception as e:
            messagebox.showerror("Open image", str(e))

    def on_save(self):
        if self.is_busy: return
        if self.img_u8 is None:
            messagebox.showinfo("Save", "Brak obrazu."); return
        path = filedialog.asksaveasfilename(
            title="Save image as",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg;*.jpeg"), ("All files", "*.*")]
        )
        if not path: return
        try:
            from PIL import Image
            Image.fromarray(self.img_u8, "RGB").save(path)
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
            except Exception:
                pass

    def on_zoom_fit(self):
        if hasattr(self.canvas, "fit"):
            try:
                self.canvas.fit()
                if hasattr(self.canvas, "get_zoom"):
                    self.zoom_scale.set(float(self.canvas.get_zoom()))
                self._update_rulers()
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

    # ---------- asynchroniczne uruchamianie ----------
    def _set_busy(self, busy: bool, label: str = ""):
        self.is_busy = busy
        try:
            # przyciski i kontrolki
            if self.btn_apply_filter: self.btn_apply_filter.config(state=("disabled" if busy else "normal"))
            if self.cmb_filter: self.cmb_filter.config(state=("disabled" if busy else "readonly"))
            # menu – zablokuj wybrane pozycje
            for m in ("file", "view", "settings", "help"):
                menu = self.menu_objects.get(m)
                if not menu: continue
                # disable wszystkie pozycje
                for i in range(menu.index("end") or -1):
                    try:
                        menu.entryconfig(i, state=("disabled" if busy else "normal"))
                    except Exception:
                        pass
            # progress + status
            self.lbl_status.config(text=(label if label else ("Working…" if busy else "Ready")))
            if busy:
                self.prog.start(80)
            else:
                self.prog.stop()
        except Exception:
            pass

    def _run_async(self, label: str, func, on_done):
        if self.is_busy:
            return
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

    # ---------- działania użytkownika ----------
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
            self._show_on_canvas(self.img_u8)
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
            self._show_on_canvas(self.img_u8)
            self._update_hud()

        self._run_async("Apply preset steps", job, done)

    # ---------- historia presetów ----------
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

    # ---------- rysowanie ----------
    def _show_on_canvas(self, img_u8: np.ndarray):
        if hasattr(self.canvas, "set_image"):
            try:
                self.canvas.set_image(np_to_u8(img_u8))
                if hasattr(self.canvas, "get_zoom"):
                    self.zoom_scale.set(float(self.canvas.get_zoom()))
                self._update_rulers()
            except Exception as e:
                print("canvas.set_image error:", e)

    def _update_rulers(self):
        if not (self.ruler_top and self.ruler_left):
            return
        try:
            # zgadujemy rozmiar – jeśli ImageCanvas ma API, można to podmienić
            H = int(self.img_u8.shape[0]) if isinstance(self.img_u8, np.ndarray) else 0
            W = int(self.img_u8.shape[1]) if isinstance(self.img_u8, np.ndarray) else 0
            zoom = float(self.zoom_scale.get())
            self.ruler_top.set_zoom_and_size(zoom, W)
            self.ruler_left.set_zoom_and_size(zoom, H)
        except Exception:
            pass

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


# --- bootstrap ---
def main():
    root = tk.Tk()
    app = App(root)
    root.geometry("1380x860")
    root.mainloop()

if __name__ == "__main__":
    main()
