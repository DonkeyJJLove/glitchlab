from typing import Dict, Type, Optional
from ..panel_base import PanelBase

_REG: Dict[str, Type[PanelBase]] = {}

def register_panel(filter_name: str, cls: Type[PanelBase]) -> None:
    _REG[filter_name.lower()] = cls

def get_panel(filter_name: str) -> Optional[Type[PanelBase]]:
    return _REG.get(filter_name.lower())

def list_panels() -> list[str]:
    return sorted(_REG.keys())
