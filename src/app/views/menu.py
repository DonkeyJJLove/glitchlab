# glitchlab/app/views/menu.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Any, Dict, List, Optional


class MenuBar:
    """
    Główne menu aplikacji (File / Edit / Presets / Tools / View / Run / Help),
    odciążające app.py. Publikuje zdarzenia na EventBus.

    Zdarzenia (publish):
      • ui.files.open            {path}
      • ui.files.save            {path}
      • ui.app.quit              {}
      • ui.preset.open           {}
      • ui.preset.save           {}            # payload: {"cfg": ...} – app może uzupełnić
      • ui.run.apply_filter      {}
      • ui.run.apply_preset      {}
      • ui.run.cancel            {}
      • ui.view.toggle_hud       {}
      • ui.view.toggle_left      {}
      • ui.view.toggle_right     {}
      • ui.view.fullscreen       {}
      • ui.tools.select          {name: pan|zoom|ruler|probe|pick}
      • ui.edit.undo             {}
      • ui.edit.redo             {}

    API:
      • set_bus(services)
      • set_recent_files(paths)
      • set_edit_enabled(undo: bool, redo: bool)
      • set_tools_value(name)    # zsynchronizuj radiobuttony Tools
    """

    TOOLS = (("pan",   "Pan"),
             ("zoom",  "Zoom"),
             ("ruler", "Ruler"),
             ("probe", "Probe"),
             ("pick",  "Pick Color"))

    def __init__(self, master: tk.Misc, *, bus: Optional[Any] = None) -> None:
        self.master = master
        self.bus = bus

        self._menubar = tk.Menu(master)

        # sekcje
        self._file   = tk.Menu(self._menubar, tearoff=False)
        self._edit   = tk.Menu(self._menubar, tearoff=False)
        self._presets= tk.Menu(self._menubar, tearoff=False)
        self._tools  = tk.Menu(self._menubar, tearoff=False)
        self._view   = tk.Menu(self._menubar, tearoff=False)
        self._run    = tk.Menu(self._menubar, tearoff=False)
        self._help   = tk.Menu(self._menubar, tearoff=False)

        # wskaźniki pozycji Undo/Redo (do enable/disable)
        self._idx_undo: Optional[int] = None
        self._idx_redo: Optional[int] = None

        # Tools – wspólna zmienna
        self._tools_var = tk.StringVar(value="pan")

        # pamięć „recent files” (opcjonalnie, slot do rozbudowy)
        self._recent_files: List[str] = []

        self._build()
        self.master.configure(menu=self._menubar)
        self._bind_accels()

    # ───────────────────────── public API ─────────────────────────

    def set_bus(self, bus: Any) -> None:
        self.bus = bus

    def set_recent_files(self, paths: List[str]) -> None:
        self._recent_files = list(paths or [])
        # (Miejsce na przebudowę podręcznego menu "Open Recent" jeśli kiedyś dodamy)

    def set_edit_enabled(self, undo: bool, redo: bool) -> None:
        try:
            if self._idx_undo is not None:
                self._edit.entryconfig(self._idx_undo, state=("normal" if undo else "disabled"))
            if self._idx_redo is not None:
                self._edit.entryconfig(self._idx_redo, state=("normal" if redo else "disabled"))
        except Exception:
            pass

    def set_tools_value(self, name: str) -> None:
        if name in {k for k, _ in self.TOOLS}:
            try:
                self._tools_var.set(name)
            except Exception:
                pass

    # ───────────────────────── build ──────────────────────────────

    def _build(self) -> None:
        # File
        self._menubar.add_cascade(label="File", menu=self._file)
        self._file.add_command(label=self._lab("Open Image…", "Ctrl+O"), command=self._open_image)
        self._file.add_command(label=self._lab("Save Image As…", "Ctrl+S"), command=self._save_image)
        self._file.add_separator()
        self._file.add_command(label=self._lab("Quit", "Ctrl+Q"), command=self._quit)

        # Edit (Undo/Redo)
        self._menubar.add_cascade(label="Edit", menu=self._edit)
        # zapisz pozycje (indeksy) – ułatwi później enable/disable
        self._idx_undo = self._edit.index("end") + 1 if self._edit.index("end") is not None else 0
        self._edit.add_command(label=self._lab("Undo", "Ctrl+Z"),
                               command=lambda: self._emit("ui.edit.undo", {}),
                               state="disabled")
        self._idx_redo = self._edit.index("end") + 1 if self._edit.index("end") is not None else 1
        self._edit.add_command(label=self._lab("Redo", "Ctrl+Y"),
                               command=lambda: self._emit("ui.edit.redo", {}),
                               state="disabled")

        # Presets (zgodne tematy dla PresetService)
        self._menubar.add_cascade(label="Presets", menu=self._presets)
        self._presets.add_command(label=self._lab("Open Preset…", "Ctrl+P"),
                                  command=lambda: self._emit("ui.preset.open", {}))
        self._presets.add_command(label="Save Preset As…",
                                  command=lambda: self._emit("ui.preset.save", {}))

        # Tools – radiobuttony publikujące ui.tools.select
        self._menubar.add_cascade(label="Tools", menu=self._tools)
        for key, lbl in self.TOOLS:
            self._tools.add_radiobutton(
                label=lbl, variable=self._tools_var, value=key,
                command=lambda n=key: self._emit("ui.tools.select", {"name": n})
            )

        # View
        self._menubar.add_cascade(label="View", menu=self._view)
        self._view.add_command(label="Toggle HUD (F10)", command=lambda: self._emit("ui.view.toggle_hud", {}))
        self._view.add_command(label="Toggle Left Tools (F8)", command=lambda: self._emit("ui.view.toggle_left", {}))
        self._view.add_command(label="Toggle Right Panel (F9)", command=lambda: self._emit("ui.view.toggle_right", {}))
        self._view.add_separator()
        self._view.add_command(label="Fullscreen", command=lambda: self._emit("ui.view.fullscreen", {}))

        # Run
        self._menubar.add_cascade(label="Run", menu=self._run)
        self._run.add_command(label=self._lab("Apply Filter", "Enter"),
                              command=lambda: self._emit("ui.run.apply_filter", {}))
        self._run.add_command(label="Apply Preset", command=lambda: self._emit("ui.run.apply_preset", {}))
        self._run.add_separator()
        self._run.add_command(label="Cancel", command=lambda: self._emit("ui.run.cancel", {}))

        # Help
        self._menubar.add_cascade(label="Help", menu=self._help)
        self._help.add_command(label="About…", command=self._about)

    # ───────────────────────── helpers ────────────────────────────

    def _about(self) -> None:
        try:
            messagebox.showinfo("About GlitchLab", "GlitchLab GTX")
        except Exception:
            self._emit("ui.about", {})

    # ───────────────────────── commands ───────────────────────────

    def _open_image(self) -> None:
        fn = filedialog.askopenfilename(
            title="Open Image",
            filetypes=[
                ("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.webp;*.tif;*.tiff"),
                ("All files", "*.*"),
            ],
        )
        if fn:
            self._emit("ui.files.open", {"path": fn})

    def _save_image(self) -> None:
        fn = filedialog.asksaveasfilename(
            title="Save Image As",
            defaultextension=".png",
            filetypes=[
                ("PNG", "*.png"),
                ("JPEG", "*.jpg;*.jpeg"),
                ("WEBP", "*.webp"),
                ("BMP", "*.bmp"),
                ("TIFF", "*.tif;*.tiff"),
            ],
        )
        if fn:
            self._emit("ui.files.save", {"path": fn})

    def _quit(self) -> None:
        self._emit("ui.app.quit", {})
        try:
            self.master.quit()
        except Exception:
            pass

    # ───────────────────────── low-level ──────────────────────────

    def _emit(self, topic: str, payload: Dict[str, Any]) -> None:
        if self.bus is not None and hasattr(self.bus, "publish"):
            try:
                self.bus.publish(topic, dict(payload))
                return
            except Exception:
                pass
        # fallback (bez busa) – cicho

    def _bind_accels(self) -> None:
        is_mac = (sys.platform == "darwin")
        mod = "Command" if is_mac else "Control"
        b = self.master.bind_all

        b(f"<{mod}-o>", lambda e: (self._prevent(e), self._open_image()))
        b(f"<{mod}-s>", lambda e: (self._prevent(e), self._save_image()))
        b(f"<{mod}-p>", lambda e: (self._prevent(e), self._emit("ui.preset.open", {})))
        b(f"<{mod}-q>", lambda e: (self._prevent(e), self._quit()))
        b("<F8>",  lambda e: (self._prevent(e), self._emit("ui.view.toggle_left", {})))
        b("<F9>",  lambda e: (self._prevent(e), self._emit("ui.view.toggle_right", {})))
        b("<F10>", lambda e: (self._prevent(e), self._emit("ui.view.toggle_hud", {})))
        b("<Return>", lambda e: (self._prevent(e), self._emit("ui.run.apply_filter", {})))
        # Edit
        b(f"<{mod}-z>", lambda e: (self._prevent(e), self._emit("ui.edit.undo", {})))
        b(f"<{mod}-y>", lambda e: (self._prevent(e), self._emit("ui.edit.redo", {})))

    # ───────────────────────── util ───────────────────────────────

    @staticmethod
    def _lab(label: str, accel: str) -> str:
        return f"{label}\t{accel}"

    @staticmethod
    def _prevent(event: tk.Event) -> None:
        try:
            event.widget.focus_set()
        except Exception:
            pass
