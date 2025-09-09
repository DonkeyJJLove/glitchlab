from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np

@dataclass
class Ctx:
    rng: np.random.Generator
    masks: Dict[str, np.ndarray]
    amplitude: Optional[np.ndarray] = None
    meta: Dict[str, Any] = None
