# glitchlab/app/docking.py
from __future__ import annotations
import tkinter as tk
from typing import Dict


class DockManager:
    def __init__(self, root: tk.Tk | tk.Toplevel, slots: Dict[str, tk.Frame]) -> None:
        self.root = root
        self.slots = slots
        self.floats: Dict[str, tk.Toplevel] = {}

    def undock(self, slot_id: str, title: str = "Panel") -> None:
        if slot_id in self.floats: return
        src = self.slots[slot_id]
        children = list(src.children.values())
        if not children: return
        top = tk.Toplevel(self.root)
        top.title(title)
        for ch in children:
            ch.master = top
            ch.pack(fill="both", expand=True)
        self.floats[slot_id] = top

        def on_close():
            self.dock(slot_id)

        top.protocol("WM_DELETE_WINDOW", on_close)

    def dock(self, slot_id: str) -> None:
        top = self.floats.pop(slot_id, None)
        if not top: return
        dst = self.slots[slot_id]
        children = list(top.children.values())
        for ch in children:
            ch.master = dst
            ch.pack(fill="both", expand=True)
        top.destroy()

    def save_layout(self) -> dict:
        return {"floating": sorted(self.floats.keys())}

    def load_layout(self, d: dict) -> None:
        for slot_id in d.get("floating", []):
            self.undock(slot_id, title=slot_id)
