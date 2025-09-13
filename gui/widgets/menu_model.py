# -*- coding: utf-8 -*-
from __future__ import annotations

# Declarative menu spec for app.py to consume (Part 3)
DEFAULT_MENU = {
    "File": [
        ("Open Image…", "open_image"),
        ("Save Result…", "save_result"),
        ("-", None),
        ("Exit", "exit_app"),
    ],
    "View": [
        ("Toggle Left Panel", "toggle_left"),
        ("Toggle Right Panel", "toggle_right"),
        ("Toggle Bottom Panel", "toggle_bottom"),
        ("-", None),
        ("Zoom In", "zoom_in"),
        ("Zoom Out", "zoom_out"),
        ("Zoom 100%", "zoom_100"),
        ("Center", "center"),
        ("Fit", "fit"),
        ("Show Crosshair", "toggle_crosshair"),
    ],
    "Presets": [
        ("Load Preset…", "preset_load"),
        ("Save Preset As…", "preset_save_as"),
        ("Apply Steps", "preset_apply"),
        ("-", None),
        ("Preset Folder…", "preset_folder"),
    ],
    "Help": [
        ("About GlitchLab…", "about"),
    ]
}
