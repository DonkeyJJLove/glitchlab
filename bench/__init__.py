# bench/__init__.py
# importuj z modułów wewnętrznych i wystaw je jako API pakietu
from .templates import get_template          # jeśli funkcja jest w templates.py
from .runner import agent_ms_like, plot_results  # przykładowe pliki

__all__ = [
    "get_template",
    "agent_ms_like",
    "plot_results",
]
