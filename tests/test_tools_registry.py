# tests/test_tools_registry.py
# -*- coding: utf-8 -*-
"""
Testy rejestru narzędzi canvasa + tryb "demo" z widocznym outputem.

Uruchomienia:
- Jako test (polecane):        pytest -q
- Z widocznymi printami:       pytest -s -q
- Jako samodzielny skrypt:     python tests/test_tools_registry.py
"""

import importlib
import types
import sys


def _load_registry():
    mod = importlib.import_module("glitchlab.gui.widgets.tools.__init__")
    assert isinstance(mod, types.ModuleType)
    return mod


def test_available_and_default(capsys=None):
    tools = _load_registry()
    names = tools.available()
    # PRINT dla czytelności (widoczny z pytest -s)
    print("[tools.available]", names)

    assert isinstance(names, list) and len(names) >= 1
    assert all(isinstance(n, str) for n in names)

    default_name = getattr(tools, "DEFAULT_TOOL")
    print("[DEFAULT_TOOL]", default_name)
    assert default_name in names


def test_get_tool_known_unknown():
    tools = _load_registry()
    for candidate in ("rect", "ellipse", "view"):
        ToolCls = tools.get_tool(candidate)
        print(f"[get_tool('{candidate}')] ->", getattr(ToolCls, "__name__", str(ToolCls)))
        assert isinstance(ToolCls, type), f"{candidate} nie zwrócił klasy"

    ToolUnknown = tools.get_tool("nope-nope-123")
    ToolDefault = tools.get_tool(getattr(tools, "DEFAULT_TOOL"))
    print("[get_tool('nope-nope-123')] -> fallback:", getattr(ToolUnknown, "__name__", str(ToolUnknown)))
    assert ToolUnknown is ToolDefault


def test_debug_errors_contract():
    tools = _load_registry()
    errs = tools.debug_errors()
    print("[debug_errors]", errs)
    assert isinstance(errs, dict)


# ───────────────────────────── tryb samodzielny ─────────────────────────────
if __name__ == "__main__":
    # Pozwala szybko zobaczyć wynik bez pytesta.
    try:
        tools = _load_registry()
        names = tools.available()
        default_name = getattr(tools, "DEFAULT_TOOL")
        errs = tools.debug_errors()

        print("=== tools registry demo ===")
        print("available:", names)
        print("DEFAULT_TOOL:", default_name)
        print("get_tool('rect'):", getattr(tools.get_tool("rect"), "__name__", "<?>"))
        print("get_tool('ellipse'):", getattr(tools.get_tool("ellipse"), "__name__", "<?>"))
        print("get_tool('view'):", getattr(tools.get_tool("view"), "__name__", "<?>"))
        print("get_tool('<unknown>') ->", getattr(tools.get_tool("___"), "__name__", "<?>"), "(fallback)")
        print("import errors:", errs if errs else "{}")

        # Minimalne asercje w trybie standalone, by nie przeoczyć problemów:
        assert isinstance(names, list) and len(names) >= 1, "Brak jakichkolwiek narzędzi w registry"
        assert default_name in names, "DEFAULT_TOOL nie wskazuje istniejącego narzędzia"
        print("\nOK: registry wygląda zdrowo.")
        sys.exit(0)
    except Exception as ex:
        print("ERROR:", type(ex).__name__, ex)
        sys.exit(1)
