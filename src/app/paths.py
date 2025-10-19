# glitchlab/app/paths.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os


def project_root() -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    return os.path.normpath(os.path.join(here, "..", ".."))


def get_default_preset_dir() -> str:
    root = project_root()
    cand = os.path.join(root, "presets")
    return cand if os.path.isdir(cand) else root
