"""
---
version: 4
kind: module
id: "view-general-tab"
name: "glitchlab.gui.views.tab_general"
author: "GlitchLab v4"
role: "Zakładka General/Global – zarządzanie maskami + mini-podgląd"
description: >
  Pokazuje listę masek w pamięci, mini-podgląd zaznaczonej maski oraz
  akcje: Add/Remove/Normalize/Refresh. Uaktualnia cache kluczy masek
  (cfg/masks/keys) i emituje zdarzenia przez Bus.
inputs:
  master: {type: "tk.Misc"}
  ctx_ref: {type: "Ctx", optional: true, desc: "masks/cache"}
  bus: {type: "EventBus-like", optional: true}
  masks_service: {type: "MasksService", optional: true}
  cfg: {type: "GeneralTabConfig", optional: true}
outputs:
  events:
    - "ui.masks.refresh"   # {}
    - "ui.masks.add_file"  # {path}
    - "ui.masks.remove"    # {name}
    - "ui.masks.normalize" # {name}
    - "ui.masks.select"    # {name}
interfaces:
  exports:
    - "GeneralTab(master, ctx_ref=None, cfg=None, bus=None, masks_service=None)"
    - "GeneralTab.set_ctx(ctx_ref) -> None"
    - "GeneralTab.refresh() -> None"
depends_on: ["tkinter", "tkinter.ttk", "typing", "numpy(optional)", "Pillow(optional)"]
used_by: ["glitchlab.gui.app", "glitchlab.gui.views.notebook"]
policy:
  deterministic: true
  ui_thread_only: true
constraints:
  - "Brak twardych zależności na core.*"
  - "Podgląd działa bez PIL/NumPy (tryb tekstowy)"
license: "Proprietary"
---
"""
# glitchlab/gui/views/tab_general.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# NumPy/Pillow są opcjonalne – gdy ich brak, ograniczamy się do tekstu
try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    from PIL import Image, ImageTk  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore
    ImageTk = None  # type: ignore


@dataclass
class GeneralTabConfig:
    """Ustawienia zakładki „Global/General”."""
    preview_size: int = 220
    allow_add_from_file: bool = True
    allow_remove: bool = True
    allow_normalize: bool = True


class GeneralTab(ttk.Frame):
    """
    Zakładka „General”: zarządzanie maskami + mini-podgląd.

    Stan/modele:
      • ctx_ref.masks : dict[str, np.ndarray]  (jeśli brak – utworzymy)
      • ctx_ref.cache : dict (uaktualniamy "cfg/masks/keys")

    Integracje (opcjonalne):
      • bus.publish("ui.masks.*", payload) – zdarzenia UI:
          - ui.masks.refresh            {}
          - ui.masks.add_file           {path}
          - ui.masks.remove             {name}
          - ui.masks.normalize          {name}
          - ui.masks.select             {name}
      • masks_service  (jeśli podasz): oczekiwane nazwy metod, jeżeli istnieją:
          - list() -> list[str]
          - get(name) -> np.ndarray
          - add_from_file(path) -> str (key)
          - remove(name) -> None
          - normalize(name) -> np.ndarray | None
        (brak metody => fallback do ctx_ref.masks)
      • callbacki:
          - on_masks_changed(dict[str,Any])
          - on_mask_selected(str)
    """

    def __init__(
        self,
        parent: tk.Misc,
        *,
        ctx_ref: Optional[Any] = None,
        cfg: Optional[GeneralTabConfig] = None,
        bus: Optional[Any] = None,
        masks_service: Optional[Any] = None,
        on_masks_changed: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_mask_selected: Optional[Callable[[str], None]] = None,
    ) -> None:
        super().__init__(parent)
        self.ctx_ref = ctx_ref
        self.cfg = cfg or GeneralTabConfig()
        self.bus = bus
        self.svc = masks_service
        self.on_masks_changed = on_masks_changed
        self.on_mask_selected = on_mask_selected

        self._tk_preview: Optional[Any] = None  # Pillow PhotoImage
        self._building_ui = False

        self._build_ui()
        self.refresh()

    # ---------------------------- Public API -----------------------------

    def set_ctx(self, ctx_ref: Any) -> None:
        self.ctx_ref = ctx_ref
        self.refresh()

    def refresh(self) -> None:
        """Przeładuj listę masek i odśwież podgląd/metryki."""
        self._ensure_ctx()
        names = self._names()
        self.lst.delete(0, "end")
        for n in names:
            self.lst.insert("end", n)
        if names:
            self.lst.selection_clear(0, "end")
            self.lst.selection_set(0)
            self.lst.event_generate("<<ListboxSelect>>")
        else:
            self._clear_preview()
        self._sync_cache_keys()
        self._publish("ui.masks.refresh", {})

    # ------------------------------ UI ---------------------------------

    def _build_ui(self) -> None:
        self._building_ui = True
        pad = dict(padx=6, pady=6)

        # Pasek akcji
        bar = ttk.Frame(self); bar.pack(fill="x", **pad)
        ttk.Label(bar, text="Masks in memory", font=("TkDefaultFont", 10, "bold")).pack(side="left")

        ttk.Button(bar, text="Refresh", command=self.refresh).pack(side="right")
        if self.cfg.allow_normalize:
            ttk.Button(bar, text="Normalize 0..1→u8", command=self._on_normalize).pack(side="right", padx=(0, 6))
        if self.cfg.allow_remove:
            ttk.Button(bar, text="Remove", command=self._on_remove).pack(side="right", padx=(0, 6))
        if self.cfg.allow_add_from_file:
            ttk.Button(bar, text="Add mask…", command=self._on_add_from_file).pack(side="right", padx=(0, 6))

        # Split: lista | podgląd
        split = ttk.Panedwindow(self, orient="horizontal"); split.pack(fill="both", expand=True, **pad)
        left = ttk.Frame(split); right = ttk.Frame(split)
        split.add(left, weight=0); split.add(right, weight=1)

        # Lista (lewa)
        lst_box = ttk.Frame(left); lst_box.pack(fill="both", expand=True)
        self.lst = tk.Listbox(lst_box, exportselection=False, height=12)
        self.lst.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(lst_box, command=self.lst.yview)
        sb.pack(side="right", fill="y"); self.lst.configure(yscrollcommand=sb.set)
        self.lst.bind("<<ListboxSelect>>", self._on_select)

        # Podgląd + metryki (prawa)
        prev_box = ttk.Frame(right); prev_box.pack(fill="both", expand=True)
        self.preview_host = ttk.LabelFrame(prev_box, text="Preview")
        self.preview_host.pack(fill="both", expand=True)
        self.preview = ttk.Label(self.preview_host, anchor="center")
        self.preview.pack(fill="both", expand=True, padx=6, pady=6)
        self.info = ttk.Label(prev_box, text="—", justify="left")
        self.info.pack(fill="x", pady=(6, 0))

        self._building_ui = False

    # ---------------------------- Events --------------------------------

    def _on_select(self, _evt=None) -> None:
        name = self._current_name()
        if not name:
            self._clear_preview(); return
        arr = self._get(name)
        self._render_preview(arr)
        self._publish("ui.masks.select", {"name": name})
        if callable(self.on_mask_selected):
            try:
                self.on_mask_selected(name)
            except Exception:
                pass

    def _on_add_from_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Add mask (image)",
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.webp;*.bmp;*.tif;*.tiff"), ("All files", "*.*")],
            initialdir=str(Path.cwd()),
        )
        if not path:
            return
        self._publish("ui.masks.add_file", {"path": path})

        # Preferuj serwis, w razie braku – lokalny fallback
        key: Optional[str] = None
        if self._has(self.svc, "add_from_file"):
            try:
                key = self.svc.add_from_file(path)  # type: ignore[attr-defined]
            except Exception:
                key = None
        if key is None:
            key = self._local_add_from_file(path)

        if key:
            self.refresh()
            self._select_by_name(key)

    def _on_remove(self) -> None:
        name = self._current_name()
        if not name:
            return
        self._publish("ui.masks.remove", {"name": name})
        if self._has(self.svc, "remove"):
            try:
                self.svc.remove(name)  # type: ignore[attr-defined]
            except Exception:
                pass
        else:
            self._ensure_ctx()
            self.ctx_ref.masks.pop(name, None)
        self.refresh()

    def _on_normalize(self) -> None:
        name = self._current_name()
        if not name or np is None:
            return
        self._publish("ui.masks.normalize", {"name": name})
        if self._has(self.svc, "normalize"):
            try:
                arr = self.svc.normalize(name)  # type: ignore[attr-defined]
                if arr is not None:
                    self._ensure_ctx(); self.ctx_ref.masks[name] = arr
            except Exception:
                pass
        else:
            arr = self._get(name)
            if arr is None:
                return
            a = np.asarray(arr, dtype=np.float32)
            mn, mx = float(a.min()), float(a.max())
            rng = max(1e-9, mx - mn)
            u8 = np.clip((a - mn) * (255.0 / rng), 0, 255).astype(np.uint8)
            self._ensure_ctx(); self.ctx_ref.masks[name] = u8
        self._notify_masks_changed()
        self._render_preview(self._get(name))

    # --------------------------- Rendering -------------------------------

    def _render_preview(self, arr: Optional["np.ndarray"]) -> None:
        if arr is None:
            self._clear_preview(); return
        h, w = int(arr.shape[0]), int(arr.shape[1])

        # metryki
        if np is not None:
            a = np.asarray(arr)
            mn, mx = float(a.min()), float(a.max())
            mean = float(a.mean())
            txt = f"{w}×{h}  dtype={a.dtype}  min={mn:.3f}  max={mx:.3f}  mean={mean:.3f}"
        else:
            txt = f"{w}×{h}  (NumPy unavailable)"
        self.info.configure(text=txt)

        # podgląd
        if Image is None or ImageTk is None or np is None:
            self.preview.configure(text="(Preview requires Pillow & NumPy)", image=""); self._tk_preview = None; return

        img = Image.fromarray(self._to_u8_gray(arr)).convert("L")
        box = int(self.cfg.preview_size)
        scale = min(box / max(1, w), box / max(1, h))
        sw, sh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        img = img.resize((sw, sh), Image.NEAREST)
        full = Image.new("L", (box, box), color=24)
        full.paste(img, ((box - sw) // 2, (box - sh) // 2))
        self._tk_preview = ImageTk.PhotoImage(full)
        self.preview.configure(image=self._tk_preview, text="")

    def _clear_preview(self) -> None:
        self.preview.configure(image="", text="(no mask selected)")
        self.info.configure(text="—")
        self._tk_preview = None

    # --------------------------- Data helpers ----------------------------

    def _ensure_ctx(self) -> None:
        if self.ctx_ref is None:
            self.ctx_ref = type("Ctx", (), {})()
        if not hasattr(self.ctx_ref, "masks") or self.ctx_ref.masks is None:
            self.ctx_ref.masks = {}
        if not hasattr(self.ctx_ref, "cache") or self.ctx_ref.cache is None:
            self.ctx_ref.cache = {}

    def _names(self) -> List[str]:
        # serwis > ctx
        if self._has(self.svc, "list"):
            try:
                names = list(self.svc.list())  # type: ignore[attr-defined]
                return sorted([n for n in names if isinstance(n, str)])
            except Exception:
                pass
        try:
            names = list(getattr(self.ctx_ref, "masks", {}).keys())
            return sorted([n for n in names if isinstance(n, str)])
        except Exception:
            return []

    def _get(self, name: str) -> Optional["np.ndarray"]:
        if self._has(self.svc, "get"):
            try:
                return self.svc.get(name)  # type: ignore[attr-defined]
            except Exception:
                pass
        try:
            return getattr(self.ctx_ref, "masks", {}).get(name)
        except Exception:
            return None

    def _local_add_from_file(self, path: str) -> Optional[str]:
        if np is None:
            messagebox.showerror("Masks", "NumPy is required."); return None
        if Image is None:
            messagebox.showerror("Masks", "Pillow is required to load images."); return None
        try:
            img = Image.open(path)
            if img.mode not in ("L", "I;16"):
                img = img.convert("L")
            arr = np.array(img, dtype=np.uint8)
            key = self._unique_key_from_path(path)
            self._ensure_ctx(); self.ctx_ref.masks[key] = arr
            self._sync_cache_keys(); self._notify_masks_changed()
            return key
        except Exception as e:
            messagebox.showerror("Add mask", str(e)); return None

    def _sync_cache_keys(self) -> None:
        try:
            self.ctx_ref.cache["cfg/masks/keys"] = self._names()
        except Exception:
            pass

    # ------------------------------ Utils --------------------------------

    def _current_name(self) -> Optional[str]:
        sel = self.lst.curselection()
        return None if not sel else self.lst.get(sel[0])

    def _select_by_name(self, name: str) -> None:
        for i in range(self.lst.size()):
            if self.lst.get(i) == name:
                self.lst.selection_clear(0, "end")
                self.lst.selection_set(i)
                self.lst.see(i)
                self.lst.event_generate("<<ListboxSelect>>")
                break

    def _unique_key_from_path(self, path: str) -> str:
        base = os.path.splitext(os.path.basename(path))[0] or "mask"
        name, i, existing = base, 1, set(self._names())
        while name in existing:
            i += 1; name = f"{base}_{i}"
        return name

    def _to_u8_gray(self, arr: "np.ndarray") -> "np.ndarray":
        a = np.asarray(arr)
        if a.ndim == 3:
            a = a[..., 0]
        if a.dtype != np.uint8:
            a = np.clip(a, 0, 255).astype(np.uint8)
        return a

    def _notify_masks_changed(self) -> None:
        if callable(self.on_masks_changed):
            try:
                self.on_masks_changed(getattr(self.ctx_ref, "masks", {}) or {})
            except Exception:
                pass

    def _publish(self, topic: str, payload: Dict[str, Any]) -> None:
        if self.bus is not None and hasattr(self.bus, "publish"):
            try:
                self.bus.publish(topic, dict(payload))
            except Exception:
                pass

    @staticmethod
    def _has(obj: Any, name: str) -> bool:
        return obj is not None and hasattr(obj, name) and callable(getattr(obj, name))
