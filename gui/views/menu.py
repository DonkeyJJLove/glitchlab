"""
---
version: 3
kind: module
id: "gui-view-menu"
created_at: "2025-09-13"
name: "glitchlab.gui.views.menu"
author: "GlitchLab v3"
role: "Menubar (File/View/Run/Help) – publikacja zdarzeń na EventBus"
description: >
  Tkinter menubar bez logiki domenowej. Dostarcza akcje plikowe (Open/Save, Recent),
  presetowe (Open/Save), sterowanie widokiem (HUD/lef/right/fullscreen) oraz
  uruchamianie pipeline (apply/cancel). Wszystko emitowane jako zdarzenia busa.
inputs:
  master: {type: "tk.Misc", desc: "root/okno z .configure(menu=...)"}
  bus: {type: "EventBus-like", optional: true, desc: "publish(topic, payload)"}
  recent_files: {type: "list[str]", via: "set_recent_files(paths)"}
outputs:
  events:
    - "ui.files.open"         # {path}
    - "ui.files.save"         # {path}
    - "ui.files.clear_recent" # {}
    - "ui.presets.open"       # {path}
    - "ui.presets.save"       # {path}
    - "ui.app.quit"           # {}
    - "ui.view.toggle_hud"    # {}
    - "ui.view.toggle_left"   # {}
    - "ui.view.toggle_right"  # {}
    - "ui.view.fullscreen"    # {}
    - "ui.run.apply_filter"   # {}
    - "ui.run.apply_preset"   # {}
    - "ui.run.cancel"         # {}
interfaces:
  exports:
    - "MenuBar(master, bus=None, show_algorithms: bool=False)"
    - "MenuBar.set_recent_files(paths: list[str]) -> None"
    - "MenuBar.set_bus(bus) -> None"
depends_on: ["tkinter", "tkinter.filedialog", "tkinter.messagebox", "typing", "sys", "os"]
used_by: ["glitchlab.gui.app_shell", "glitchlab.gui.app (legacy)"]
policy:
  deterministic: true
  ui_thread_only: true
constraints:
  - "brak twardych zależności na core"
  - "dialogi plikowe synchroniczne; akcje domenowe przez EventBus"
accelerators:
  - "Ctrl+O → ui.files.open"
  - "Ctrl+S → ui.files.save"
  - "Ctrl+P → ui.presets.open"
  - "Ctrl+Q → ui.app.quit"
  - "F8/F9/F10 → toggle left/right/HUD"
  - "Enter → ui.run.apply_filter"
license: "Proprietary"
---
"""
# glitchlab/gui/views/menu.py
from __future__ import annotations

import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Any, Callable, Dict, List, Optional


EventCb = Callable[[str, Dict[str, Any]], None]


class MenuBar:
    """
    Tkinter menubar dla GlitchLab GUI.
    Nie wykonuje logiki domenowej – tylko publikuje zdarzenia na EventBus.
    """

    def __init__(
        self,
        master: tk.Misc,
        *,
        bus: Optional[Any] = None,  # EventBus-like: publish(topic, payload)
        show_algorithms: bool = False,
    ) -> None:
        self.master = master
        self.bus = bus
        self.show_algorithms = bool(show_algorithms)

        self._menubar = tk.Menu(master)
        self._file_menu = tk.Menu(self._menubar, tearoff=False)
        self._recent_menu = tk.Menu(self._file_menu, tearoff=False)
        self._view_menu = tk.Menu(self._menubar, tearoff=False)
        self._run_menu = tk.Menu(self._menubar, tearoff=False)
        self._help_menu = tk.Menu(self._menubar, tearoff=False)

        self._recent_files: List[str] = []

        self._build()
        self._attach()
        self._bind_accelerators()

    # -----------------------------
    # Public API
    # -----------------------------

    def set_bus(self, bus: Any) -> None:
        self.bus = bus

    def set_recent_files(self, paths: List[str]) -> None:
        """Ustawia listę „Open Recent…” i odświeża menu."""
        self._recent_files = list(paths or [])
        self._rebuild_recent_menu()

    # -----------------------------
    # Internal: build/attach
    # -----------------------------

    def _cmd_about(self) -> None:
        """Pokazuje okno 'About…' (fallback: event przez bus)."""
        try:
            from tkinter import messagebox
            messagebox.showinfo(
                "About GlitchLab",
                "GlitchLab — controlled glitch for analysis\nGUI v3 (refactor)\n© D2J3 aka Cha0s"
            )
        except Exception:
            # jeśli środowisko headless albo błąd Tk – wyemituj event
            if self.bus:
                try:
                    self.bus.publish("ui.about", {})
                except Exception:
                    pass

    def _build(self) -> None:
        # File
        self._menubar.add_cascade(label="File", menu=self._file_menu)
        self._file_menu.add_command(
            label=self._label_with_accel("Open…", "Ctrl+O"),
            command=self._cmd_open_image,
        )
        self._file_menu.add_cascade(label="Open Recent", menu=self._recent_menu)
        self._file_menu.add_separator()
        self._file_menu.add_command(
            label=self._label_with_accel("Save As…", "Ctrl+S"),
            command=self._cmd_save_image,
        )
        self._file_menu.add_separator()
        self._file_menu.add_command(
            label=self._label_with_accel("Open Preset…", "Ctrl+P"),
            command=self._cmd_open_preset,
        )
        self._file_menu.add_command(
            label="Save Preset As…",
            command=self._cmd_save_preset,
        )
        self._file_menu.add_separator()
        self._file_menu.add_command(
            label=self._label_with_accel("Quit", "Ctrl+Q"),
            command=self._cmd_quit,
        )

        # View
        self._menubar.add_cascade(label="View", menu=self._view_menu)
        self._view_menu.add_command(label="Toggle HUD (F10)", command=self._cmd_toggle_hud)
        self._view_menu.add_command(label="Toggle Left Tools (F8)", command=self._cmd_toggle_left)
        self._view_menu.add_command(label="Toggle Right Panel (F9)", command=self._cmd_toggle_right)
        self._view_menu.add_separator()
        self._view_menu.add_command(label="Fullscreen", command=self._cmd_fullscreen)

        # Run
        self._menubar.add_cascade(label="Run", menu=self._run_menu)
        self._run_menu.add_command(label=self._label_with_accel("Apply Filter", "Enter"), command=self._cmd_apply_filter)
        self._run_menu.add_command(label="Apply Preset", command=self._cmd_apply_preset)
        self._run_menu.add_separator()
        self._run_menu.add_command(label="Cancel", command=self._cmd_cancel_run)

        # Help
        self._menubar.add_cascade(label="Help", menu=self._help_menu)
        self._help_menu.add_command(label="About…", command=self._cmd_about)

        # open recent initially empty
        self._rebuild_recent_menu()

    def _attach(self) -> None:
        self.master.configure(menu=self._menubar)

    # -----------------------------
    # Accelerators
    # -----------------------------

    def _bind_accelerators(self) -> None:
        is_mac = sys.platform == "darwin"
        mod = "Command" if is_mac else "Control"

        self.master.bind_all(f"<{mod}-o>", lambda e: (self._prevent_default(e), self._cmd_open_image()))
        self.master.bind_all(f"<{mod}-s>", lambda e: (self._prevent_default(e), self._cmd_save_image()))
        self.master.bind_all(f"<{mod}-p>", lambda e: (self._prevent_default(e), self._cmd_open_preset()))
        self.master.bind_all(f"<{mod}-q>", lambda e: (self._prevent_default(e), self._cmd_quit()))
        self.master.bind_all("<F8>", lambda e: (self._prevent_default(e), self._cmd_toggle_left()))
        self.master.bind_all("<F9>", lambda e: (self._prevent_default(e), self._cmd_toggle_right()))
        self.master.bind_all("<F10>", lambda e: (self._prevent_default(e), self._cmd_toggle_hud()))
        self.master.bind_all("<Return>", lambda e: (self._prevent_default(e), self._cmd_apply_filter()))

    @staticmethod
    def _prevent_default(event: tk.Event) -> None:
        try:
            event.widget.focus_set()
        except Exception:
            pass

    @staticmethod
    def _label_with_accel(label: str, accel: str) -> str:
        return f"{label}\t{accel}"

    # -----------------------------
    # Recent submenu
    # -----------------------------

    def _rebuild_recent_menu(self) -> None:
        self._recent_menu.delete(0, "end")
        if not self._recent_files:
            self._recent_menu.add_command(label="(empty)", state="disabled")
            return
        for p in self._recent_files[:12]:
            disp = self._shorten_path(p)
            self._recent_menu.add_command(
                label=disp,
                command=lambda path=p: self._publish("ui.files.open", {"path": path}),
            )
        self._recent_menu.add_separator()
        self._recent_menu.add_command(label="Clear List", command=self._cmd_clear_recent)

    @staticmethod
    def _shorten_path(path: str, maxlen: int = 60) -> str:
        if len(path) <= maxlen:
            return path
        head, tail = path[: maxlen // 2 - 2], path[-maxlen // 2 + 2 :]
        return f"{head}…{tail}"

    # -----------------------------
    # Commands → EventBus
    # -----------------------------

    def _cmd_open_image(self) -> None:
        fn = filedialog.askopenfilename(
            title="Open Image",
            filetypes=[
                ("Images", "*.png;*.jpg;*.jpeg;*.webp;*.bmp;*.tif;*.tiff"),
                ("All files", "*.*"),
            ],
        )
        if fn:
            self._publish("ui.files.open", {"path": fn})

    def _cmd_save_image(self) -> None:
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
            self._publish("ui.files.save", {"path": fn})

    def _cmd_open_preset(self) -> None:
        fn = filedialog.askopenfilename(
            title="Open Preset YAML",
            filetypes=[("YAML", "*.yml;*.yaml"), ("All files", "*.*")],
        )
        if fn:
            self._publish("ui.presets.open", {"path": fn})

    def _cmd_save_preset(self) -> None:
        fn = filedialog.asksaveasfilename(
            title="Save Preset As",
            defaultextension=".yml",
            filetypes=[("YAML", "*.yml;*.yaml")],
        )
        if fn:
            self._publish("ui.presets.save", {"path": fn})

    def _cmd_quit(self) -> None:
        self._publish("ui.app.quit", {})
        # na wszelki wypadek: jeśli nikt nie obsłuży, zamknij lokalnie
        try:
            self.master.quit()
        except Exception:
            pass

    def _cmd_toggle_hud(self) -> None:
        self._publish("ui.view.toggle_hud", {})

    def _cmd_toggle_left(self) -> None:
        self._publish("ui.view.toggle_left", {})

    def _cmd_toggle_right(self) -> None:
        self._publish("ui.view.toggle_right", {})

    def _cmd_fullscreen(self) -> None:
        self._publish("ui.view.fullscreen", {})

    def _cmd_apply_filter(self) -> None:
        self._publish("ui.run.apply_filter", {})

    def _cmd_apply_preset(self) -> None:
        self._publish("ui.run.apply_preset", {})

    def _cmd_cancel_run(self) -> None:
        self._publish("ui.run.cancel", {})

    def _cmd_clear_recent(self) -> None:
        self._publish("ui.files.clear_recent", {})

    # -----------------------------
    # Bus helper
    # -----------------------------

    def _publish(self, topic: str, payload: Dict[str, Any]) -> None:
        if self.bus is not None and hasattr(self.bus, "publish"):
            try:
                self.bus.publish(topic, dict(payload))
                return
            except Exception:
                pass
        # Fallback: pokaż info
        # (pomocne przy debugowaniu bez podłączonego Bus)
        try:
            messagebox.showinfo("Event", f"{topic}\n{payload}")
        except Exception:
            pass
