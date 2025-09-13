# -*- coding: utf-8 -*-
from .hud import Hud  # noqa
# Optional/aux widgets (wired from app.py in Part 3)
try:
    from .mask_chooser import MaskChooser  # noqa
except Exception:
    MaskChooser = None
try:
    from .mask_browser import MaskBrowser  # noqa
except Exception:
    MaskBrowser = None
try:
    from .pipeline_preview import PipelinePreview  # noqa
except Exception:
    PipelinePreview = None
