# gui/services/layer_manager.py
from __future__ import annotations
from dataclasses import replace
from typing import Optional, Callable, Dict, Any
import uuid
import numpy as np
from PIL import Image
try:
    from gui.app import AppState
except Exception:
    from dataclasses import dataclass, field
    from typing import List, Literal
    BlendMode = Literal["normal","multiply","screen","overlay","add","subtract","darken","lighten"]
    @dataclass
    class Layer:
        id: str
        name: str
        image: np.ndarray | Image.Image
        visible: bool = True
        opacity: float = 1.0
        blend: BlendMode = "normal"
        mask: Optional[np.ndarray] = None
        meta: Dict[str, Any] = field(default_factory=dict)
    @dataclass
    class AppState:
        image_in:  Image.Image | np.ndarray | None = None
        layers: List[Layer] = field(default_factory=list)
        active_layer_id: Optional[str] = None
from gui.services.compositor import composite_stack, BlendMode
class LayerManager:
    def __init__(self, state: AppState, publish: Callable[[str, Dict[str, Any]], None]) -> None:
        self.state = state
        self.publish = publish
    def _ensure_active(self) -> None:
        if not getattr(self.state, "layers", None) and getattr(self.state, "image_in", None) is not None:
            self.add_layer(self.state.image_in, name="Background")
        if getattr(self.state, "active_layer_id", None) is None and getattr(self.state, "layers", None):
            self.state.active_layer_id = self.state.layers[0].id
    def add_layer(self, image: np.ndarray | Image.Image, name: str = "Layer",
                  blend: BlendMode = "normal", opacity: float = 1.0, visible: bool = True) -> str:
        lid = str(uuid.uuid4())
        if isinstance(image, Image.Image):
            img_u8 = np.asarray(image.convert("RGB"), dtype=np.uint8)
        else:
            img_u8 = image
        layer = Layer(id=lid, name=name, image=img_u8, visible=visible, opacity=opacity, blend=blend)
        self.state.layers.append(layer)  # type: ignore
        self.state.active_layer_id = lid
        self.publish("ui.layers.changed", {})
        return lid
    def remove_layer(self, lid: str) -> None:
        self.state.layers = [l for l in self.state.layers if l.id != lid]  # type: ignore
        if getattr(self.state, "active_layer_id", None) == lid:
            self.state.active_layer_id = self.state.layers[0].id if self.state.layers else None  # type: ignore
        self.publish("ui.layers.changed", {})
    def update_layer(self, lid: str, **patch) -> None:
        for i, l in enumerate(self.state.layers):  # type: ignore
            if l.id == lid:
                self.state.layers[i] = replace(l, **patch)  # type: ignore
                break
        self.publish("ui.layers.changed", {})
    def get_composite_for_viewport(self) -> Optional[np.ndarray]:
        self._ensure_active()
        vis = [l for l in getattr(self.state, "layers", []) if getattr(l, "visible", True)]
        if not vis:
            return None
        stack = []
        for l in vis:
            mask = l.mask.astype(np.float32) if getattr(l, "mask", None) is not None else None
            img = np.asarray(l.image) if isinstance(l.image, Image.Image) else l.image
            stack.append((img, float(l.opacity), l.blend, mask))
        return composite_stack(stack)
