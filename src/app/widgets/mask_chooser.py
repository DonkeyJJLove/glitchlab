# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Callable, List, Optional
import os
import tkinter as tk
from tkinter import ttk, filedialog, simpledialog, messagebox

import numpy as np

try:
    from PIL import Image
except Exception:
    Image = None  # łagodny fallback – przy Load pokażemy błąd


class MaskChooser(ttk.Frame):
    """
    Prosta biblioteka masek:
      - Lista przewijalna z aktualnymi kluczami (pobierana z get_mask_keys()).
      - Przyciski: Load… (wczytaj maskę z pliku i dodaj przez on_add_mask), Refresh.
      - Zmiana zaznaczenia wywołuje on_select(key).

    Oczekiwane typy:
      - on_add_mask(key: str, arr: np.ndarray) – arr float32 w [0,1], shape (H,W).
      - get_mask_keys() -> list[str]
    """
    def __init__(
        self,
        parent: tk.Misc,
        get_mask_keys: Callable[[], List[str]],
        on_add_mask: Callable[[str, np.ndarray], None],
        on_select: Callable[[Optional[str]], None] | None = None,
    ):
        super().__init__(parent)
        self.get_mask_keys = get_mask_keys
        self.on_add_mask = on_add_mask
        self.on_select = on_select or (lambda key: None)

        self._build_ui()
        self.refresh()

    # -------- UI --------
    def _build_ui(self) -> None:
        # Pasek narzędzi
        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=4, pady=(4, 2))

        ttk.Label(bar, text="Masks").pack(side="left")
        ttk.Button(bar, text="Load…", command=self._on_load).pack(side="right")
        ttk.Button(bar, text="Refresh", command=self.refresh).pack(side="right", padx=(0, 6))

        # Lista przewijalna
        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, padx=4, pady=(0, 4))

        self.listbox = tk.Listbox(body, height=6, exportselection=False)
        self.listbox.pack(side="left", fill="both", expand=True)

        sb = ttk.Scrollbar(body, orient="vertical", command=self.listbox.yview)
        sb.pack(side="right", fill="y")
        self.listbox.configure(yscrollcommand=sb.set)

        self.listbox.bind("<<ListboxSelect>>", self._on_list_select)
        self.listbox.bind("<Double-1>", self._on_list_activate)

        # Stopka z aktywnym kluczem
        foot = ttk.Frame(self)
        foot.pack(fill="x", padx=4, pady=(0, 4))
        ttk.Label(foot, text="Selected:").pack(side="left")
        self.var_sel = tk.StringVar(value="<none>")
        ttk.Label(foot, textvariable=self.var_sel).pack(side="left", padx=(6, 0))

    # -------- actions --------
    def refresh(self) -> None:
        """Przeładuj listę z get_mask_keys()."""
        try:
            keys = list(self.get_mask_keys() or [])
        except Exception:
            keys = []
        self.listbox.delete(0, "end")
        for k in keys:
            self.listbox.insert("end", k)
        # spróbuj zachować wybór
        if keys:
            self.listbox.selection_set(0)
            self._emit_selection(keys[0])
        else:
            self._emit_selection(None)

    def _on_list_select(self, _evt=None) -> None:
        key = self._current_key()
        self._emit_selection(key)

    def _on_list_activate(self, _evt=None) -> None:
        # double click = emit select jeszcze raz (np. podgląd)
        key = self._current_key()
        self._emit_selection(key)

    def _emit_selection(self, key: Optional[str]) -> None:
        self.var_sel.set(key or "<none>")
        try:
            self.on_select(key)
        except Exception:
            pass

    def _current_key(self) -> Optional[str]:
        try:
            sel = self.listbox.curselection()
            if not sel:
                return None
            return self.listbox.get(sel[0])
        except Exception:
            return None

    def _on_load(self) -> None:
        """Wczytaj maskę z pliku i dodaj przez on_add_mask()."""
        if Image is None:
            messagebox.showerror("Load mask", "Pillow (PIL) nie jest zainstalowany.")
            return

        path = filedialog.askopenfilename(
            parent=self,
            title="Load mask image",
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.webp"), ("All files", "*.*")]
        )
        if not path:
            return

        try:
            pil = Image.open(path).convert("L")  # grayscale
            arr = np.asarray(pil, dtype=np.float32) / 255.0
            # zapytaj o nazwę (domyślnie nazwa pliku bez rozszerzenia)
            default_key = os.path.splitext(os.path.basename(path))[0]
            key = simpledialog.askstring("Mask name", "Name for the mask:", initialvalue=default_key, parent=self)
            if not key:
                return

            # dodaj
            self.on_add_mask(str(key), arr)
            self.refresh()
            # ustaw wybór na nowo dodany
            for i in range(self.listbox.size()):
                if self.listbox.get(i) == key:
                    self.listbox.selection_clear(0, "end")
                    self.listbox.selection_set(i)
                    self.listbox.see(i)
                    break
            self._emit_selection(key)
        except Exception as e:
            messagebox.showerror("Load mask", str(e))
