# glitchlab/gui/app.py
# -*- coding: utf-8 -*-
"""
GlitchLab — aplikacja GUI (Tkinter) spajająca core, filtry i panele mini-UI.

FUNKCJE:
- Wczytywanie/zapis obrazu
- Presety (YAML) + Apply preset
- Tryb „Single filter” (z panelami mini-UI) + Apply single filter
- Amplitude & Edge Mask z GUI (działa zarówno dla presetów jak i single)
- Ładowanie masek symbolicznych (bitmapa → ctx.masks[key])
- Podglądy diagnostyczne: ctx.cache[*], amplitude, maska, logi
- Analiza format-aware (entropy, edge density, JPEG blockiness, PNG alpha)
- AST pipeline’u (drzewo) + tryb mozaiki (tile masks → mask_key)
"""

from __future__ import annotations

import os
import traceback
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

from glitchlab.core import registry as reg
from glitchlab.core.pipeline import (
    load_image, save_image, load_config, build_ctx, apply_pipeline,
)
from glitchlab.core.utils import normalize_image
from glitchlab.core.symbols import load_mask_image, register_mask

# metryki analizy
try:
    from glitchlab.analysis.metrics import compute_entropy, edge_density
except Exception:
    # fallback, gdy moduł jeszcze nie podpięty
    def compute_entropy(a: np.ndarray) -> float:
        g = a
        if g.ndim == 3:
            g = 0.299 * g[..., 0] + 0.587 * g[..., 1] + 0.114 * g[..., 2]
        g = g.astype(np.float64)
        if g.max() > 1.0:
            g = g / 255.0
        hist, _ = np.histogram(g, bins=256, range=(0, 1), density=True)
        p = hist + 1e-12
        import numpy as _np
        return float(-_np.sum(p * _np.log2(p)))


    def edge_density(a: np.ndarray) -> float:
        if a.ndim == 3:
            g = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
        else:
            g = a
        g = g.astype(np.float64)
        if g.max() > 1.0:
            g = g / 255.0
        gx = np.abs(np.diff(g, axis=1))
        gy = np.abs(np.diff(g, axis=0))
        gx = np.pad(gx, ((0, 0), (0, 1)), constant_values=0)
        gy = np.pad(gy, ((0, 1), (0, 0)), constant_values=0)
        return float(np.mean((gx + gy) / 2.0))

from glitchlab.gui.panel_loader import get_panel_for_filter
from glitchlab.gui.panel_base import PanelContext

# Wymuszenie rejestracji filtrów (efekt uboczny importu):
import glitchlab.filters  # noqa: F401

# Kolorystyka
BG = "#101216"
PANEL_BG = "#16181c"
FG = "#e6e6e6"
ACCENT = "#9aa5ff"

APP_TITLE = "GlitchLab — Panels Edition"


# -------------------------- Helpery obrazów --------------------------

def _to_rgb_uint8(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 2:
        a = np.stack([a, a, a], axis=-1)
    if a.ndim == 3 and a.shape[2] == 4:
        alpha = a[..., 3:4].astype(np.float32) / 255.0
        rgb = a[..., :3].astype(np.float32)
        a = (rgb * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    if a.dtype != np.uint8:
        if np.issubdtype(a.dtype, np.floating):
            a = np.clip(a * (255.0 if a.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
        else:
            a = np.clip(a, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(a)


def _np_to_tk(arr: np.ndarray, max_side: int = 1200) -> ImageTk.PhotoImage:
    """Skalowanie tylko w dół — główny podgląd."""
    arr = _to_rgb_uint8(arr)
    pil = Image.fromarray(arr, "RGB")
    w, h = pil.size
    s = max_side / max(w, h)
    if s < 1.0:
        pil = pil.resize((int(w * s), int(h * s)), Image.BICUBIC)
    return ImageTk.PhotoImage(pil)


def _np_to_tk_fit(arr: np.ndarray, side: int = 256) -> ImageTk.PhotoImage:
    """Dopasuj dłuższy bok do 'side' (może powiększać)."""
    arr = _to_rgb_uint8(arr)
    pil = Image.fromarray(arr, "RGB")
    w, h = pil.size
    s = side / max(w, h)
    pil = pil.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BICUBIC)
    return ImageTk.PhotoImage(pil)


def _gray_vis(f: np.ndarray) -> np.ndarray:
    f = np.asarray(f).astype(np.float32)
    f = f - f.min()
    d = f.max() + 1e-8
    if d > 0:
        f = f / d
    return (f * 255.0).astype(np.uint8)


# -------------------------- Analiza format-aware --------------------------

def _estimate_jpeg_blockiness(arr: np.ndarray) -> float:
    """Prosta miara „kratki JPEG”: kontrast na liniach 8 px – tło."""
    g = arr
    if g.ndim == 3:
        g = 0.299 * g[..., 0] + 0.587 * g[..., 1] + 0.114 * g[..., 2]
    g = g.astype(np.float64)
    if g.max() > 1.0:
        g = g / 255.0
    H, W = g.shape
    vgrid = np.abs(g[:, 8::8] - g[:, 7:-1:8]).mean() if W > 8 else 0.0
    hgrid = np.abs(g[8::8, :] - g[7:-1:8, :]).mean() if H > 8 else 0.0
    vbg = np.abs(g[:, 9::8] - g[:, 8:-1:8]).mean() if W > 9 else 0.0
    hbg = np.abs(g[9::8, :] - g[8:-1:8, :]).mean() if H > 9 else 0.0
    return float(max(0.0, (vgrid + hgrid) - (vbg + hbg)))


def _alpha_coverage(pil_img: Image.Image) -> Optional[float]:
    if "A" not in pil_img.getbands():
        return None
    a = np.asarray(pil_img.getchannel("A"), dtype=np.uint8)
    return float((a > 0).mean())


# -------------------------- Listy presetów/filtrów --------------------------

def _presets_dir() -> str:
    here = os.path.dirname(os.path.dirname(__file__))  # .../glitchlab/gui -> .../glitchlab
    return os.path.join(here, "presets")


def list_presets() -> List[str]:
    pdir = _presets_dir()
    if not os.path.isdir(pdir):
        return []
    return sorted(os.path.splitext(fn)[0] for fn in os.listdir(pdir) if fn.lower().endswith(".yaml"))


def normalize_preset(cfg, preset_name: str | None = None):
    """
    Akceptuje oba formaty:
      A) root ma 'steps' / 'edge_mask' / 'amplitude'
      B) pojedyncza nazwana sekcja (np. {my_preset: {steps: ...}})
    Zwraca samą sekcję z krokami/parametrami.
    """
    if not isinstance(cfg, dict):
        raise ValueError("Preset YAML must be a dict.")

    # Format A (już w root)
    if ("steps" in cfg) or ("edge_mask" in cfg) or ("amplitude" in cfg):
        return cfg

    # Format B (sekcja o nazwie presetu)
    if preset_name and preset_name in cfg and isinstance(cfg[preset_name], dict):
        return cfg[preset_name]

    # Format B (jedyna sekcja bez znajomości nazwy)
    if len(cfg) == 1:
        only = next(iter(cfg.values()))
        if isinstance(only, dict):
            return only

    raise ValueError("Could not find steps/edge_mask/amplitude in preset YAML.")


def list_filter_names_canonical() -> List[str]:
    try:
        names = reg.available()
    except Exception:
        names = []
    seen, uniq = set(), []
    for nm in sorted(names):
        try:
            can = reg.canonical(nm)
        except Exception:
            can = nm
        if can not in seen:
            seen.add(can)
            uniq.append(can)
    return uniq


# -------------------------- Aplikacja --------------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.configure(bg=BG)
        self.geometry("1280x880")

        # dane
        self.arr: np.ndarray | None = None
        self.result: np.ndarray | None = None
        self.ctx = None
        self.image_path: str | None = None
        self.seed: int = 7

        # info o pliku (format-aware metryki)
        self.file_info: Dict[str, Any] = {}

        # maski użytkownika (utrzymujemy między wywołaniami)
        self.user_masks: Dict[str, np.ndarray] = {}

        # obrazy tk
        self._tkimage_main = None
        self._tkpreview_amp = None
        self._tkpreview_edge = None
        self._tkpreview_diag1 = None
        self._tkpreview_diag2 = None

        # log z apply_pipeline
        self.debug_log: List[str] = []

        # AST Tree
        self.tree_ast: Optional[ttk.Treeview] = None

        self._build_layout()
        self._populate_initial_lists()

        # po starcie przesuń pierwsze odświeżenie (żeby mieć wymiary ramek)
        self.after(120, self._update_bottom_previews)

        # aliasy awaryjne (opcjonalnie)
        self._ensure_gui_aliases({
            "anisotropic_contour_warp": ["anisotropic_contour_flow", "contour_flow"],
            "nosh_perlin_grid": ["noise_perlin_grid", "perlin_grid"],
            "spectral_shaper": ["spectral_shape", "spectral_shaper_lab"],
        })

    # -------------------------- Layout --------------------------

    def _build_layout(self):
        # top bar (stały — przyciski nie znikają przy zmianie panelu)
        top = tk.Frame(self, bg=BG)
        top.pack(side=tk.TOP, fill="x", padx=8, pady=6)

        ttk.Button(top, text="Open image…", command=self._cmd_open).pack(side=tk.LEFT)
        ttk.Button(top, text="Save result…", command=self._cmd_save).pack(side=tk.LEFT, padx=(8, 0))

        tk.Label(top, text="Seed:", bg=BG, fg=FG).pack(side=tk.LEFT, padx=(16, 4))
        self.seed_var = tk.IntVar(self, value=7)
        ttk.Entry(top, textvariable=self.seed_var, width=8).pack(side=tk.LEFT)

        tk.Label(top, text="Preset:", bg=BG, fg=FG).pack(side=tk.LEFT, padx=(16, 4))
        self.preset_var = tk.StringVar(self, value="")
        self.preset_box = ttk.Combobox(top, textvariable=self.preset_var, values=[], state="readonly", width=28)
        self.preset_box.pack(side=tk.LEFT)
        ttk.Button(top, text="Apply preset", command=self._cmd_apply_preset).pack(side=tk.LEFT, padx=(8, 0))

        tk.Label(top, text="Filter:", bg=BG, fg=FG).pack(side=tk.LEFT, padx=(16, 4))
        self.filter_var = tk.StringVar(self, value="")
        self.filter_box = ttk.Combobox(top, textvariable=self.filter_var, values=[], state="readonly", width=28)
        self.filter_box.pack(side=tk.LEFT)
        ttk.Button(top, text="Apply single", command=self._cmd_apply_single).pack(side=tk.LEFT, padx=(8, 0))

        ttk.Button(top, text="Load mask…", command=self._cmd_load_mask).pack(side=tk.LEFT, padx=(16, 0))

        # main area
        main = tk.Frame(self, bg=BG)
        main.pack(fill="both", expand=True, padx=8, pady=6)

        left = tk.Frame(main, bg=BG)
        left.pack(side=tk.LEFT, fill="both", expand=True, padx=(0, 6))
        self.canvas_label = tk.Label(left, text="(no image)", bg=BG, fg="#888")
        self.canvas_label.pack(fill="both", expand=True)

        right = tk.Frame(main, bg=PANEL_BG, bd=1, relief="solid")
        right.pack(side=tk.LEFT, fill="y", padx=(6, 0))

        info = tk.Frame(right, bg=PANEL_BG)
        info.pack(fill="x", padx=8, pady=6)
        tk.Label(info, text="Filter info", bg=PANEL_BG, fg=ACCENT,
                 font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
        self.filter_info = tk.Text(info, height=6, width=52, bg="#0f1114", fg=FG, wrap="word")
        self.filter_info.pack(fill="x")
        self.filter_info.configure(state="disabled")

        self.panel_holder = tk.LabelFrame(right, text="Parameters", bg=PANEL_BG, fg=FG)
        self.panel_holder.pack(fill="x", padx=8, pady=6)
        self.active_panel = None

        self.aux_holder = tk.LabelFrame(right, text="Amplitude / Edge / Mosaic", bg=PANEL_BG, fg=FG)
        self.aux_holder.pack(fill="x", padx=8, pady=6)
        self._build_aux_controls(self.aux_holder)

        btns = tk.Frame(right, bg=PANEL_BG)
        btns.pack(fill="x", padx=8, pady=6)
        ttk.Button(btns, text="Reload panels", command=self.on_reload_panels).pack(side=tk.LEFT)
        ttk.Button(btns, text="Reset params", command=self.on_reset_params).pack(side=tk.LEFT, padx=(6, 0))

        # bottom diagnostics (3 kolumny: Masks/Amp, Diagnostics, Logs+AST)
        bottom_h = 320
        self.bottom = tk.Frame(self, bg=BG, height=bottom_h)
        self.bottom.pack(side=tk.BOTTOM, fill="x", padx=8, pady=(0, 8))
        self.bottom.pack_propagate(False)

        self.bot_left = tk.LabelFrame(self.bottom, text="Masks & Amplitude", bg=PANEL_BG, fg=FG)
        self.bot_mid = tk.LabelFrame(self.bottom, text="Filter Diagnostics", bg=PANEL_BG, fg=FG)
        self.bot_right = tk.LabelFrame(self.bottom, text="Inspect", bg=PANEL_BG, fg=FG)

        for lf, pad in ((self.bot_left, (0, 4)), (self.bot_mid, (4, 4)), (self.bot_right, (4, 0))):
            lf.pack(side=tk.LEFT, fill="both", expand=True, padx=pad)
            lf.pack_propagate(False)

        self.lbl_amp = tk.Label(self.bot_left, text="(amplitude)", bg=PANEL_BG, fg="#888", anchor="center")
        self.lbl_edge = tk.Label(self.bot_left, text="(mask)", bg=PANEL_BG, fg="#888", anchor="center")
        self.lbl_amp.pack(fill="both", expand=True, padx=6, pady=6)
        self.lbl_edge.pack(fill="both", expand=True, padx=6, pady=6)

        self.lbl_diag1 = tk.Label(self.bot_mid, text="(diag #1)", bg=PANEL_BG, fg="#888", anchor="center")
        self.lbl_diag2 = tk.Label(self.bot_mid, text="(diag #2)", bg=PANEL_BG, fg="#888", anchor="center")
        self.lbl_diag1.pack(fill="both", expand=True, padx=6, pady=6)
        self.lbl_diag2.pack(fill="both", expand=True, padx=6, pady=6)

        # zakładki: Logs / AST
        self.nb = ttk.Notebook(self.bot_right)
        self.nb.pack(fill="both", expand=True, padx=6, pady=6)

        self.tab_logs = tk.Frame(self.nb, bg=PANEL_BG)
        self.tab_ast = tk.Frame(self.nb, bg=PANEL_BG)
        self.nb.add(self.tab_logs, text="Logs")
        self.nb.add(self.tab_ast, text="AST")

        self.log_text = tk.Text(self.tab_logs, bg="#0f1114", fg=FG, wrap="word")
        self.log_text.pack(fill="both", expand=True, padx=6, pady=6)

        self.tree_ast = ttk.Treeview(self.tab_ast, columns=("value",), show="tree")
        self.tree_ast.pack(fill="both", expand=True, padx=6, pady=6)

        # zdarzenia comboboxów
        self.filter_box.bind("<<ComboboxSelected>>", lambda e: self._cmd_filter_changed())
        self.preset_box.bind("<<ComboboxSelected>>", lambda e: None)
        self.amp_mask_box.bind("<<ComboboxSelected>>", lambda e: self._update_bottom_previews())

        # status
        self.status = tk.Label(self, text="Ready", anchor="w", bg=BG, fg=FG)
        self.status.pack(side=tk.BOTTOM, fill="x")

    def _build_aux_controls(self, parent: tk.Widget):
        # Amplitude
        tk.Label(parent, text="Amplitude", bg=PANEL_BG, fg=ACCENT,
                 font=("TkDefaultFont", 9, "bold")).grid(row=0, column=0, sticky="w", padx=6, pady=(6, 2), columnspan=6)

        tk.Label(parent, text="kind", bg=PANEL_BG, fg=FG).grid(row=1, column=0, sticky="w", padx=6)
        self.amp_kind = tk.StringVar(self, value="none")
        ttk.Combobox(parent, textvariable=self.amp_kind,
                     values=["none", "linear_x", "linear_y", "radial", "perlin", "mask"],
                     state="readonly", width=12).grid(row=1, column=1, sticky="w")

        tk.Label(parent, text="strength", bg=PANEL_BG, fg=FG).grid(row=1, column=2, sticky="w", padx=6)
        self.amp_strength = tk.DoubleVar(self, value=1.0)
        ttk.Entry(parent, textvariable=self.amp_strength, width=8).grid(row=1, column=3, sticky="w")

        tk.Label(parent, text="scale", bg=PANEL_BG, fg=FG).grid(row=2, column=0, sticky="w", padx=6)
        self.amp_scale = tk.DoubleVar(self, value=96.0)
        ttk.Entry(parent, textvariable=self.amp_scale, width=8).grid(row=2, column=1, sticky="w")

        tk.Label(parent, text="octaves", bg=PANEL_BG, fg=FG).grid(row=2, column=2, sticky="w", padx=6)
        self.amp_octaves = tk.IntVar(self, value=4)
        ttk.Entry(parent, textvariable=self.amp_octaves, width=6).grid(row=2, column=3, sticky="w")

        tk.Label(parent, text="mask_key", bg=PANEL_BG, fg=FG).grid(row=3, column=0, sticky="w", padx=6)
        self.amp_mask_key = tk.StringVar(self, value="")
        self.amp_mask_box = ttk.Combobox(parent, textvariable=self.amp_mask_key, values=[], state="normal", width=12)
        self.amp_mask_box.grid(row=3, column=1, sticky="w")

        # Edge
        tk.Label(parent, text="Edge Mask", bg=PANEL_BG, fg=ACCENT,
                 font=("TkDefaultFont", 9, "bold")).grid(row=4, column=0, sticky="w", padx=6, pady=(8, 2), columnspan=6)

        tk.Label(parent, text="thresh", bg=PANEL_BG, fg=FG).grid(row=5, column=0, sticky="w", padx=6)
        self.edge_thresh = tk.IntVar(self, value=60)
        ttk.Entry(parent, textvariable=self.edge_thresh, width=6).grid(row=5, column=1, sticky="w")

        tk.Label(parent, text="dilate", bg=PANEL_BG, fg=FG).grid(row=5, column=2, sticky="w", padx=6)
        self.edge_dilate = tk.IntVar(self, value=0)
        ttk.Entry(parent, textvariable=self.edge_dilate, width=6).grid(row=5, column=3, sticky="w")

        tk.Label(parent, text="ksize", bg=PANEL_BG, fg=FG).grid(row=6, column=0, sticky="w", padx=6)
        self.edge_ksize = tk.IntVar(self, value=3)
        ttk.Entry(parent, textvariable=self.edge_ksize, width=6).grid(row=6, column=1, sticky="w")

        # Mosaic
        tk.Label(parent, text="Mosaic (tile mask → auto mask_key)", bg=PANEL_BG, fg=ACCENT,
                 font=("TkDefaultFont", 9, "bold")).grid(row=7, column=0, sticky="w", padx=6, pady=(8, 2), columnspan=6)
        self.mosaic_enable = tk.BooleanVar(self, False)
        ttk.Checkbutton(parent, text="Enable", variable=self.mosaic_enable).grid(row=8, column=0, sticky="w", padx=6)
        tk.Label(parent, text="Tiles X", bg=PANEL_BG, fg=FG).grid(row=8, column=1, sticky="w", padx=6)
        self.mosaic_tx = tk.IntVar(self, 2);
        ttk.Entry(parent, textvariable=self.mosaic_tx, width=6).grid(row=8, column=2, sticky="w")
        tk.Label(parent, text="Tiles Y", bg=PANEL_BG, fg=FG).grid(row=8, column=3, sticky="w", padx=6)
        self.mosaic_ty = tk.IntVar(self, 2);
        ttk.Entry(parent, textvariable=self.mosaic_ty, width=6).grid(row=8, column=4, sticky="w")

    def _populate_initial_lists(self):
        self.preset_box.configure(values=list_presets())
        if self.preset_box["values"]:
            self.preset_var.set(self.preset_box["values"][0])

        filters = list_filter_names_canonical()
        self.filter_box.configure(values=filters)
        if filters:
            self.filter_var.set(filters[0])
            self._build_filter_panel(filters[0])
            self._update_filter_info(filters[0])

    # -------------------------- Pomocnicze --------------------------

    def set_status(self, txt: str):
        self.title(f"{APP_TITLE} — {txt}")

    def show_image(self, arr: np.ndarray):
        self._tkimage_main = _np_to_tk(arr, max_side=1200)
        self.canvas_label.configure(image=self._tkimage_main, text="")
        self.canvas_label.image = self._tkimage_main
        # analityka do logów (format-aware)
        self._update_metrics_log(arr)

    def _append_log(self, line: str):
        self.debug_log.append(line)
        self.log_text.delete("1.0", "end")
        self.log_text.insert("1.0", "\n".join(self.debug_log))

    def _panel_context(self) -> PanelContext:
        # łączymy maski kontekstu z maskami użytkownika (priorytet user)
        keys = set(self.user_masks.keys())
        if self.ctx is not None and getattr(self.ctx, "masks", None):
            keys.update(self.ctx.masks.keys())
        return PanelContext(mask_keys=sorted(keys))

    def _update_amp_mask_dropdown(self):
        self.amp_mask_box.configure(values=[""] + self._panel_context().mask_keys)

    def _build_filter_panel(self, filter_name: str):
        for w in self.panel_holder.winfo_children():
            w.destroy()
        self.active_panel = get_panel_for_filter(filter_name)
        self.active_panel.set_context_provider(self._panel_context)
        self.active_panel.on_change = lambda: None
        root = self.active_panel.build(self.panel_holder)
        root.pack(fill="both", expand=True)

    def _update_filter_info(self, filter_name: str):
        try:
            m = reg.meta(filter_name)
            doc = (m.get("doc") or "").strip()
            defaults = m.get("defaults") or {}
            lines = [f"[{filter_name}] module={m.get('module', '')}", ""]
            if doc:
                lines += [doc, ""]
            if defaults:
                dd = ", ".join(f"{k}={v!r}" for k, v in defaults.items())
                lines.append("defaults: " + dd)
            txt = "\n".join(lines)
        except Exception:
            txt = f"[{filter_name}]"
        self.filter_info.configure(state="normal")
        self.filter_info.delete("1.0", "end")
        self.filter_info.insert("1.0", txt)
        self.filter_info.configure(state="disabled")

    def _inject_user_masks(self):
        """Po build_ctx wstrzykuj maski usera do ctx, by żyły między wywołaniami."""
        if self.ctx is None or not self.user_masks:
            return
        for k, m in self.user_masks.items():
            try:
                register_mask(self.ctx.masks, k, m, merge="replace")
            except Exception:
                self.ctx.masks[k] = m

    def _calc_side(self, frame: tk.Widget, slots: int) -> int:
        """Dopasuj rozmiar miniatur do ramki (slots=2 — dwa obrazy pionowo)."""
        try:
            w = max(1, frame.winfo_width())
            h = max(1, frame.winfo_height())
        except Exception:
            w, h = 300, 280
        pad = 24
        side_h = (h - pad) // max(1, slots)
        side_w = w - pad
        return max(96, min(side_h, side_w, 512))

    def _get_preview_mask(self):
        """
        Zwraca (mask_array, label) do pokazania w Masks & Amplitude.
        Priorytety:
          1) jeśli amplitude.kind == 'mask' i mask_key ustawione -> ta maska
          2) 'edge' jeśli jest
          3) pierwsza dostępna maska z ctx.masks lub self.user_masks
        """
        candidates = []

        if self.ctx is not None and getattr(self.ctx, "masks", None):
            for k, v in self.ctx.masks.items():
                candidates.append((k, v))

        if getattr(self, "user_masks", None):
            for k, v in self.user_masks.items():
                if not any(k == kk for kk, _ in candidates):
                    candidates.append((k, v))

        try:
            if self.amp_kind.get() == "mask":
                mk = (self.amp_mask_key.get() or "").strip()
                if mk:
                    for k, v in candidates:
                        if k == mk:
                            return v, f"mask:{k}"
        except Exception:
            pass

        for k, v in candidates:
            if k == "edge":
                return v, "mask:edge"

        if candidates:
            k, v = candidates[0]
            return v, f"mask:{k}"

        return None, None

    def _update_bottom_previews(self):
        # amplitude
        try:
            amp = getattr(self.ctx, "amplitude", None)
            side = self._calc_side(self.bot_left, slots=2)
            if amp is not None:
                vis = _gray_vis(amp)
                self._tkpreview_amp = _np_to_tk_fit(vis, side=side)
                self.lbl_amp.configure(image=self._tkpreview_amp, text="")
                self.lbl_amp.image = self._tkpreview_amp
            else:
                self.lbl_amp.configure(image="", text="(amplitude)")
        except Exception:
            self.lbl_amp.configure(image="", text="(amplitude)")

        # mask preview
        try:
            m, label = self._get_preview_mask()
            if m is not None:
                m = np.asarray(m).astype(np.float32)
                if m.ndim == 3:
                    m = m[..., 0]
                vis = (np.clip(m, 0, 1) * 255).astype(np.uint8)
                self._tkpreview_edge = _np_to_tk_fit(vis, side=self._calc_side(self.bot_left, 2))
                self.lbl_edge.configure(image=self._tkpreview_edge, text=label or "mask")
                self.lbl_edge.image = self._tkpreview_edge
            else:
                self.lbl_edge.configure(image="", text="(no mask)")
        except Exception:
            self.lbl_edge.configure(image="", text="(no mask)")

        # diagnostyka (dwie pierwsze pozycje z ctx.cache)
        self.lbl_diag1.configure(image="", text="(diag #1)")
        self.lbl_diag2.configure(image="", text="(diag #2)")
        self._tkpreview_diag1 = None
        self._tkpreview_diag2 = None
        try:
            if self.ctx is not None and getattr(self.ctx, "cache", None):
                imgs = []
                for key, val in self.ctx.cache.items():
                    arr = np.asarray(val)
                    if arr.ndim == 2:
                        arr = _to_rgb_uint8(arr)
                    elif arr.ndim == 3 and arr.shape[2] in (1, 3, 4):
                        arr = _to_rgb_uint8(arr[..., :3])
                    else:
                        continue
                    imgs.append((arr, key))
                    if len(imgs) >= 2:
                        break
                if imgs:
                    side_mid = self._calc_side(self.bot_mid, slots=2)
                    self._tkpreview_diag1 = _np_to_tk_fit(imgs[0][0], side=side_mid)
                    self.lbl_diag1.configure(image=self._tkpreview_diag1, text=imgs[0][1])
                    self.lbl_diag1.image = self._tkpreview_diag1
                if len(imgs) >= 2:
                    side_mid = self._calc_side(self.bot_mid, slots=2)
                    self._tkpreview_diag2 = _np_to_tk_fit(imgs[1][0], side=side_mid)
                    self.lbl_diag2.configure(image=self._tkpreview_diag2, text=imgs[1][1])
                    self.lbl_diag2.image = self._tkpreview_diag2
        except Exception:
            pass

        # logi odśwież
        self.log_text.delete("1.0", "end")
        if self.debug_log:
            self.log_text.insert("1.0", "\n".join(self.debug_log))

    # -------------------------- Analiza / AST --------------------------

    def _update_metrics_log(self, arr: np.ndarray):
        try:
            ent = compute_entropy(arr)
            edg = edge_density(arr)
            lines = [
                f"[Metrics] Entropy: {ent:.4f}",
                f"[Metrics] Edge density: {edg:.4f}",
            ]
            # format aware
            if self.file_info:
                fmt = self.file_info.get("format")
                mode = self.file_info.get("mode")
                size = self.file_info.get("size")
                if fmt:
                    lines.append(f"[File] {fmt} ({mode}) {size[0]}×{size[1]}")
                if fmt == "JPEG":
                    blk = _estimate_jpeg_blockiness(arr)
                    lines.append(f"[JPEG] blockiness: {blk:.6f}")
                if fmt == "PNG" and self.file_info.get("alpha_coverage") is not None:
                    ac = self.file_info["alpha_coverage"]
                    lines.append(f"[PNG] alpha coverage: {ac * 100:.2f}%")
            self._append_log("\n".join(lines))
        except Exception:
            pass

    def _update_ast_view(self, steps: List[Dict[str, Any]]):
        try:
            self.tree_ast.delete(*self.tree_ast.get_children())
        except Exception:
            return
        root = self.tree_ast.insert("", "end", text="pipeline", open=True)
        for i, step in enumerate(steps):
            nm = step.get("name", f"step_{i}")
            node = self.tree_ast.insert(root, "end", text=f"{i}: {nm}", open=False)
            prm = step.get("params", {})
            for k, v in prm.items():
                self.tree_ast.insert(node, "end", text=f"{k} = {v}")

    # -------------------------- Zdarzenia / Komendy --------------------------

    def on_reload_panels(self):
        self._cmd_filter_changed()

    def on_reset_params(self):
        self._cmd_filter_changed()

    def _collect_aux_cfg(self) -> Dict[str, Any]:
        amp_kind = self.amp_kind.get()
        amp = {"kind": amp_kind, "strength": float(self.amp_strength.get() or 1.0)}
        if amp_kind == "perlin":
            amp.update({
                "scale": float(self.amp_scale.get() or 96.0),
                "octaves": int(self.amp_octaves.get() or 4),
                "persistence": 0.5,
                "lacunarity": 2.0,
                "base": int(self.seed_var.get() or 7),
            })
        if amp_kind == "mask":
            mk = self.amp_mask_key.get().strip()
            if mk:
                amp["mask_key"] = mk

        edge = {
            "thresh": int(self.edge_thresh.get() or 60),
            "dilate": int(self.edge_dilate.get() or 0),
            "ksize": int(self.edge_ksize.get() or 3),
        }
        # mozaika tylko dla single (sterujemy poza cfg)
        return {"amplitude": amp, "edge_mask": edge}

    def _cmd_open(self):
        path = filedialog.askopenfilename(
            title="Open image",
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.webp"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            # format-aware info z PIL
            pil = Image.open(path)
            fmt = pil.format or "UNKNOWN"
            mode = pil.mode
            size = pil.size
            self.file_info = {
                "path": path, "format": fmt, "mode": mode, "size": size,
                "alpha_coverage": _alpha_coverage(pil)
            }

            a = normalize_image(load_image(path))
            self.arr = a
            self.image_path = path
            self.result = None
            self.ctx = None
            self.show_image(self.arr)
            self.set_status(
                f"Loaded: {os.path.basename(path)}  |  {self.arr.shape[1]}×{self.arr.shape[0]}  |  {fmt}/{mode}")
            self._update_amp_mask_dropdown()
            self._update_bottom_previews()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _cmd_save(self):
        if self.result is None:
            messagebox.showinfo("Info", "No result to save.")
            return
        out_path = filedialog.asksaveasfilename(
            title="Save result", defaultextension=".png", filetypes=[("PNG", "*.png")]
        )
        if out_path:
            try:
                save_image(_to_rgb_uint8(self.result), out_path)
                self.set_status(f"Saved: {os.path.basename(out_path)}")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def _cmd_filter_changed(self):
        fname = self.filter_var.get()
        if not fname:
            return
        self._build_filter_panel(fname)
        self._update_filter_info(fname)

    def _cmd_apply_preset(self):
        if self.arr is None:
            messagebox.showwarning("Warning", "Load an image first.")
            return

        name = self.preset_var.get()
        if not name:
            messagebox.showwarning("Warning", "Choose a preset.")
            return

        path = os.path.join(_presets_dir(), f"{name}.yaml")
        if not os.path.exists(path):
            messagebox.showerror("Error", f"Preset not found:\n{path}")
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = load_config(f.read())
            cfg = normalize_preset(raw, name)

            # wstrzykiwanie ustawień z panelu Amplitude/Edge:
            aux = self._collect_aux_cfg()
            for k in ("amplitude", "edge_mask"):
                if k in aux and aux[k]:
                    cfg[k] = aux[k]

            self.seed = int(self.seed_var.get() or 7)
            self.ctx = build_ctx(self.arr, seed=self.seed, cfg=cfg)
            self._inject_user_masks()

            steps = cfg.get("steps", [])
            self.set_status(f"Applying preset '{name}'…")

            out = apply_pipeline(self.arr.copy(), self.ctx, steps)  # <— BEZ fail_fast/debug_log
            self.result = _to_rgb_uint8(out)
            self.show_image(self.result)
            self._update_amp_mask_dropdown()
            self._update_bottom_previews()
            self.set_status(f"Applied preset '{name}'")
        except Exception as e:
            messagebox.showerror("Preset error", str(e))

    def _cmd_apply_single(self):
        if self.arr is None:
            messagebox.showwarning("Warning", "Load an image first.")
            return

        fname = self.filter_var.get()
        if not fname:
            messagebox.showwarning("Warning", "Choose a filter.")
            return

        try:
            params = {}
            if self.active_panel is not None:
                self.active_panel.validate()
                params = self.active_panel.get_params()

            steps = [{"name": fname, "params": params}]
            aux = self._collect_aux_cfg()
            cfg = {"steps": steps}
            cfg.update(aux)

            self.seed = int(self.seed_var.get() or 7)
            self.ctx = build_ctx(self.arr, seed=self.seed, cfg=cfg)
            self._inject_user_masks()

            self.set_status(f"Applying filter '{fname}'…")
            out = apply_pipeline(self.arr.copy(), self.ctx, steps)  # <— BEZ fail_fast/debug_log
            self.result = _to_rgb_uint8(out)
            self.show_image(self.result)
            self._update_amp_mask_dropdown()
            self._update_bottom_previews()
            self.set_status(f"Applied filter '{fname}'")
        except Exception as e:
            messagebox.showerror("Filter error", str(e))

    def _cmd_load_mask(self):
        if self.arr is None:
            messagebox.showwarning("Warning", "Load an image first.")
            return
        path = filedialog.askopenfilename(
            title="Load mask (grayscale image)",
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.webp"), ("All files", "*.*")]
        )
        if not path:
            return

        dlg = tk.Toplevel(self)
        dlg.title("Mask options")
        dlg.configure(bg=PANEL_BG)

        tk.Label(dlg, text="mask_key:", bg=PANEL_BG, fg=FG).grid(row=0, column=0, sticky="w", padx=6, pady=4)
        key_var = tk.StringVar(dlg, value=os.path.splitext(os.path.basename(path))[0])
        ttk.Entry(dlg, textvariable=key_var, width=24).grid(row=0, column=1, sticky="w", padx=6)

        tk.Label(dlg, text="threshold (0..255):", bg=PANEL_BG, fg=FG).grid(row=1, column=0, sticky="w", padx=6, pady=4)
        thr_var = tk.IntVar(dlg, value=128)
        ttk.Entry(dlg, textvariable=thr_var, width=10).grid(row=1, column=1, sticky="w", padx=6)

        inv_var = tk.BooleanVar(dlg, value=False)
        ttk.Checkbutton(dlg, text="invert", variable=inv_var).grid(row=2, column=1, sticky="w", padx=6)

        def _ok():
            try:
                key = key_var.get().strip()
                if not key:
                    raise ValueError("mask_key cannot be empty")
                H, W = self.arr.shape[:2]
                m = load_mask_image(
                    path, shape_hw=(H, W),
                    threshold=int(thr_var.get() or 128),
                    invert=bool(inv_var.get())
                )
                # zapamiętaj w pamięci użytkownika
                self.user_masks[key] = m.copy()

                # jeśli ctx nie istnieje — zbuduj, by podglądy zadziałały natychmiast
                if self.ctx is None:
                    self.ctx = build_ctx(self.arr, seed=int(self.seed_var.get() or 7), cfg={})

                # zarejestruj w ctx
                register_mask(self.ctx.masks, key, m, merge="replace")

                # odśwież dropdown i podglądy
                self._update_amp_mask_dropdown()
                self._update_bottom_previews()

                messagebox.showinfo("OK", f"Loaded mask '{key}'")
            except Exception as e:
                messagebox.showerror("Mask error", str(e))
            finally:
                dlg.destroy()

        ttk.Button(dlg, text="OK", command=_ok).grid(row=3, column=1, sticky="e", padx=6, pady=6)
        dlg.transient(self)
        dlg.grab_set()
        self.wait_window(dlg)

    # -------------------------- Alias helper (GUI) --------------------------

    def _ensure_gui_aliases(self, mapping: Dict[str, List[str]]):
        """
        Jeśli 'wanted' nie istnieje, a któryś alias istnieje — doregistruj
        alias w locie po stronie GUI (bez grzebania w core).
        mapping: {wanted: [alias1, alias2, ...]}
        """
        try:
            available = set(reg.available())
        except Exception:
            return
        for wanted, candidates in mapping.items():
            if wanted in available:
                continue
            found = None
            for alt in candidates:
                if alt in available:
                    found = alt
                    break
            if found is None:
                continue
            try:
                fn = reg.get(found)  # type: ignore[attr-defined]
                register = getattr(reg, "register", None)
                if callable(register):
                    register(wanted)(fn)  # type: ignore[misc]
                elif hasattr(reg, "_REGISTRY"):
                    reg._REGISTRY[wanted] = reg._REGISTRY[found]  # type: ignore[attr-defined]
            except Exception:
                pass
