from __future__ import annotations
from tkinter import ttk

class WelcomePanel(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=10)
        title = ttk.Label(self, text="👋 GlitchLab v3 — Welcome", font=("Segoe UI", 14, "bold"))
        title.pack(anchor="w", pady=(0,8))
        txt = (
            "Quick start:\n"
            "1) File → Open image… (or use the placeholder)\n"
            "2) Pick a filter (default_identity is preselected)\n"
            "3) Adjust params on the right and Run → Apply filter\n"
            "\n"
            "Tips:\n"
            "• F9/F10/F11 toggle left/right/bottom panels\n"
            "• Ctrl+O / Ctrl+S / Ctrl+R for open/save/run\n"
            "• Load mask… to enable mask_overlay; set mask_key in panel\n"
        )
        ttk.Label(self, text=txt, justify="left").pack(anchor="w")

        # Footer
        ttk.Label(self, text="HUD shows diagnostics from ctx.cache (diag/*, stage/*, ast/json).",
                  foreground="#8aa").pack(anchor="w", pady=(12,0))
