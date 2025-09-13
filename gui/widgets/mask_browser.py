# -*- coding: utf-8 -*-
from __future__ import annotations
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Dict, Callable, Optional
import numpy as np
from PIL import Image
import os

def _load_mask_file(path: str) -> np.ndarray:
    im = Image.open(path).convert("L")
    arr = (np.asarray(im).astype(np.float32) / 255.0).clip(0.0, 1.0)
    return arr

class MaskBrowser(ttk.Frame):
    """
    Lightweight mask browser with add/remove.
    Communicates via callbacks:
      - on_add_mask(key:str, arr:np.ndarray) -> None
      - on_remove_mask(key:str) -> None (optional)
    Shows coarse preview info (HxW and mean coverage).
    """
    def __init__(self, parent,
                 get_masks: Callable[[], Dict[str, np.ndarray]],
                 on_add_mask: Callable[[str, np.ndarray], None],
                 on_remove_mask: Optional[Callable[[str], None]] = None):
        super().__init__(parent)
        self.get_masks = get_masks
        self.on_add_mask = on_add_mask
        self.on_remove_mask = on_remove_mask

        tb = ttk.Frame(self); tb.pack(fill="x", pady=(2, 4))
        ttk.Button(tb, text="Add…", command=self._add).pack(side="left")
        ttk.Button(tb, text="Remove", command=self._remove).pack(side="left", padx=(6,0))
        ttk.Button(tb, text="Refresh", command=self._refresh).pack(side="left", padx=(6,0))

        body = ttk.Frame(self); body.pack(fill="both", expand=True)
        self.list = tk.Listbox(body, height=8, exportselection=False)
        self.list.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(body, command=self.list.yview)
        self.list.configure(yscrollcommand=sb.set); sb.pack(side="left", fill="y")
        self.info = tk.StringVar(value="")
        ttk.Label(self, textvariable=self.info).pack(anchor="w", padx=2, pady=2)

        self.list.bind("<<ListboxSelect>>", self._update_info)
        self._refresh()

    def _refresh(self):
        masks = self.get_masks() or {}
        sel = self.list.curselection()
        sel_name = self.list.get(sel[0]) if sel else None
        self.list.delete(0, "end")
        for name in sorted(masks.keys()):
            self.list.insert("end", name)
        if sel_name:
            idx = None
            for i in range(self.list.size()):
                if self.list.get(i) == sel_name:
                    idx = i; break
            if idx is not None:
                self.list.selection_set(idx)
        self._update_info()

    def _update_info(self, _evt=None):
        sel = self.list.curselection()
        if not sel:
            self.info.set("No mask selected"); return
        name = self.list.get(sel[0])
        arr = (self.get_masks() or {}).get(name)
        if arr is None:
            self.info.set("No mask selected"); return
        h, w = arr.shape[:2]
        coverage = float(arr.mean()) * 100.0
        self.info.set(f"{name}: {w}x{h}, coverage ~{coverage:.1f}%")

    def _add(self):
        path = filedialog.askopenfilename(
            title="Load mask",
            filetypes=[("Images","*.png;*.jpg;*.jpeg;*.bmp;*.webp"), ("All","*.*")]
        )
        if not path: return
        try:
            arr = _load_mask_file(path)
        except Exception as e:
            messagebox.showerror("Mask", f"Cannot load: {e}"); return
        base = os.path.splitext(os.path.basename(path))[0]
        name = base
        masks = self.get_masks() or {}
        i = 1
        while name in masks:
            name = f"{base}_{i}"; i += 1
        self.on_add_mask(name, arr)
        self._refresh()

    def _remove(self):
        if self.on_remove_mask is None: return
        sel = self.list.curselection()
        if not sel: return
        name = self.list.get(sel[0])
        if messagebox.askyesno("Remove mask", f"Delete mask '{name}'?"):
            self.on_remove_mask(name)
            self._refresh()
