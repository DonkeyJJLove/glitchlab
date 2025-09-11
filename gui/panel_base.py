from __future__ import annotations
import tkinter as tk
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

@dataclass
class PanelContext:
    filter_name: str
    defaults: Dict[str, Any]
    params: Dict[str, Any]
    on_change: Callable[[Dict[str, Any]], None]
    cache_ref: Dict[str, Any]  # e.g., last_ctx.cache

class PanelBase(tk.Frame):
    def __init__(self, master, ctx: Optional[PanelContext] = None, **kwargs):
        super().__init__(master, **kwargs)
        self._ctx = ctx
        self.on_change = (ctx.on_change if ctx else None)
        self.build()

    def build(self) -> None:
        pass

    # contracts
    def get_params(self) -> Dict[str, Any]:
        return dict(self._ctx.params if self._ctx else {})

    def set_params(self, params: Dict[str, Any]) -> None:
        if self._ctx:
            self._ctx.params = dict(params)
        if self.on_change:
            self.on_change(self._ctx.params if self._ctx else params)

    def load_defaults(self, defaults: Dict[str, Any]) -> None:
        if self._ctx:
            self._ctx.defaults = dict(defaults)
