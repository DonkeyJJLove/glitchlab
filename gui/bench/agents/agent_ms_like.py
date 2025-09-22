# -*- coding: utf-8 -*-
from __future__ import annotations
import time
from typing import Dict, Any


# To jest STUB. Podmień ciało generate_code na realny pipeline „B”.
def generate_code(task: Dict[str, Any], mode: str = "B", **kwargs) -> Dict[str, Any]:
    """
    Powinno zwrócić: dict(code:str, metrics:dict, time_s:float).
    - code: kod źródłowy Pythona z funkcją o nazwie `entrypoint`
    - metrics: dodać swoje metryki (np. liczba iteracji — opcjonalnie)
    - time_s: czas generacji
    """
    t0 = time.time()
    # PRZYKŁADOWY (pusty) kod – użyj własnej strategii!
    name = task["entrypoint"]
    code = f"def {name}(*args, **kwargs):\n    raise NotImplementedError\n"
    return dict(code=code, metrics={}, time_s=time.time() - t0)
