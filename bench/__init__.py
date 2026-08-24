# bench/__init__.py
# importuj z modułów wewnętrznych i wystaw je jako API pakietu
from .runner import agent_ms_like, plot_results  # przykładowe pliki
from .templates import get_template  # jeśli funkcja jest w templates.py

__all__ = [
    "agent_ms_like",
    "get_template",
    "plot_results",
]
