# glitchlab/gui/main.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os, sys, platform, traceback
import tkinter as tk
from tkinter import ttk, messagebox

# Upewnij się, że mamy na ścieżce katalog pakietu "glitchlab"
THIS = os.path.abspath(os.path.dirname(__file__))  # .../glitchlab/gui
PKG = os.path.dirname(THIS)  # .../glitchlab
ROOT = os.path.dirname(PKG)  # projekt

for p in (ROOT, PKG):
    if p not in sys.path:
        sys.path.insert(0, p)

# Importuj App absolutnie (jako pakiet)
try:
    from glitchlab.gui.app import App
except Exception:
    # ostatnia deska (uruchamianie z katalogu gui)
    from app import App  # type: ignore


def _enable_windows_dpi():
    if platform.system().lower() != "windows":
        return
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass


def _apply_theme(root: tk.Tk):
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass
    bg = "#121417";
    fg = "#E6E6E6";
    sel = "#1B1F24"
    style.configure(".", background=bg, foreground=fg, fieldbackground=bg)
    style.configure("TFrame", background=bg)
    style.configure("TLabel", background=bg, foreground=fg)
    style.configure("TLabelframe", background=bg, foreground=fg)
    style.configure("TLabelframe.Label", background=bg, foreground=fg)
    style.configure("TButton", background=bg, foreground=fg)
    style.map("TButton", background=[("active", sel)])
    root.option_add("*Label.background", bg)
    root.option_add("*Label.foreground", fg)


def _install_excepthook():
    def showbox(exc_type, exc, tb):
        msg = "".join(traceback.format_exception(exc_type, exc, tb))
        try:
            messagebox.showerror("GlitchLab — błąd", msg)
        except Exception:
            sys.stderr.write(msg + "\n")

    sys.excepthook = showbox


def _bind_shortcuts(app: App):
    app.bind_all("<Control-o>", lambda e: app.on_open())
    app.bind_all("<Control-s>", lambda e: app.on_save())
    app.bind_all("<F5>", lambda e: getattr(app, "run_pipeline", lambda: None)())
    app.bind_all("<Control-Return>", lambda e: getattr(app, "_cmd_apply_single", lambda: None)())


def main() -> int:
    _enable_windows_dpi()
    _install_excepthook()

    app = App()
    _apply_theme(app)

    app.update_idletasks()
    w, h = 1400, 900
    sw, sh = app.winfo_screenwidth(), app.winfo_screenheight()
    x, y = max(0, (sw - w) // 2), max(0, (sh - h) // 2)
    app.geometry(f"{w}x{h}+{x}+{y}")
    app.minsize(1100, 720)

    _bind_shortcuts(app)
    app.title("GlitchLab — Studio")
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
