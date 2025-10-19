"""
glitchlab.app.widgets.param_form
--------------------------------

This module defines the ``ParamForm`` class extracted from ``app/app.py``.
ParamForm is a fallback dynamic parameter form used when no dedicated panel
exists for a filter. It introspects a callable to build a simple form
with labels and entry fields based on the filter's signature or custom
schema.

Moving ``ParamForm`` out of the monolithic ``app.py`` helps to decouple UI
components and makes the application easier to maintain.  See ``app.py``
for usage.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Dict, Any, List, Union, Optional, Iterable


class ParamForm(ttk.Frame):
    """Fallback dynamic form for filter parameters.

    ``ParamForm`` takes a callable (typically a filter registered in
    ``glitchlab.core.registry``) and introspects its signature or a
    user-provided schema in order to present a simple form to the user.
    Supported parameter types are bool, int, float and str. Enum values
    can be provided via a ``choices`` list in the schema.

    Parameters
    ----------
    parent : tkinter.Widget
        The parent widget in which to place this frame.
    get_filter_callable : Callable[[str], Callable]
        A function that, given a filter name, returns the corresponding
        callable. This is used to introspect signatures when no schema
        is provided on the callable.
    """

    def __init__(self, parent: tk.Widget, get_filter_callable: Callable[[str], Callable]):
        super().__init__(parent)
        self._get_filter_callable = get_filter_callable
        self._controls: Dict[str, tk.Variable] = {}
        self._current_name: Optional[str] = None

    def build_for(self, name: str) -> None:
        """Build the form for a given filter name.

        Existing widgets are destroyed and new controls are created based on
        the filter's parameter schema. If no schema can be resolved, an
        informative label is shown instead.

        Parameters
        ----------
        name : str
            The canonical filter name to build a form for.
        """
        # clear previous contents
        for w in self.winfo_children():
            w.destroy()
        self._controls.clear()
        self._current_name = name

        spec = self._resolve_spec(name)
        if not spec:
            ttk.Label(
                self,
                text="(Brak znanego schematu parametrów; użyj presetów)"
            ).pack(anchor="w", padx=6, pady=6)
            return

        grid = ttk.Frame(self)
        grid.pack(fill="x", expand=True, padx=6, pady=6)
        for i, item in enumerate(spec):
            n = item["name"]
            typ = item.get("type", "float")
            default = item.get("default")
            choices = item.get("choices")
            ttk.Label(grid, text=n).grid(row=i, column=0, sticky="w", padx=(2, 6), pady=2)

            if typ == "bool":
                var: tk.Variable = tk.BooleanVar(value=bool(default))
                ctrl: tk.Widget = ttk.Checkbutton(grid, variable=var)
            elif typ == "enum" and choices:
                default_choice = default if default in choices else (choices[0] if choices else "")
                var = tk.StringVar(value=str(default_choice))
                ctrl = ttk.Combobox(grid, textvariable=var, state="readonly", values=list(choices))
            elif typ == "int":
                var = tk.StringVar(value=str(int(default if default is not None else 0)))
                ctrl = ttk.Entry(grid, textvariable=var, width=10)
            elif typ == "float":
                var = tk.StringVar(value=str(float(default if default is not None else 0.0)))
                ctrl = ttk.Entry(grid, textvariable=var, width=10)
            else:
                var = tk.StringVar(value=str(default if default is not None else ""))
                ctrl = ttk.Entry(grid, textvariable=var)

            ctrl.grid(row=i, column=1, sticky="ew", padx=(0, 6), pady=2)
            grid.columnconfigure(1, weight=1)
            self._controls[n] = var

    def values(self) -> Dict[str, Any]:
        """Return the current parameter values.

        The values are cast to the types defined in the resolved schema.

        Returns
        -------
        dict
            A dictionary mapping parameter names to typed values.
        """
        spec = self._resolve_spec(self._current_name) if self._current_name else []
        type_map = {i["name"]: i.get("type", "float") for i in spec}
        out: Dict[str, Any] = {}
        for k, var in self._controls.items():
            v = var.get()
            typ = type_map.get(k, "float")
            try:
                if typ == "bool":
                    out[k] = bool(v)
                elif typ == "int":
                    out[k] = int(v)
                elif typ == "float":
                    out[k] = float(v)
                else:
                    out[k] = str(v)
            except Exception:
                out[k] = v
        return out

    def _resolve_spec(self, name: str) -> List[Dict[str, Any]]:
        """Resolve a parameter specification for a filter.

        This tries multiple strategies:
        1. Look for an attribute called ``schema``/``params_schema``/``PARAMS``/``SPEC`` on
           the filter callable. If it exists and returns a list/tuple/dict, use it.
        2. Use ``inspect.signature`` to introspect the filter's parameters (excluding
           ``img`` and ``ctx``).
        3. Return an empty list if nothing is found.

        Parameters
        ----------
        name : str
            The canonical filter name.

        Returns
        -------
        list of dict
            A list of parameter definitions.
        """
        # Resolve callable
        try:
            f = self._get_filter_callable(name)
        except Exception:
            return []
        # Try known schema attributes
        for attr in ("schema", "params_schema", "PARAMS", "SPEC"):
            try:
                obj = getattr(f, attr)
                schema = obj() if callable(obj) else obj
                if isinstance(schema, (list, tuple)):
                    return [dict(x) for x in schema]
                if isinstance(schema, dict) and "params" in schema:
                    return [dict(x) for x in (schema["params"] or [])]
            except Exception:
                pass
        # Fallback: use inspect.signature
        try:
            import inspect
            sig = inspect.signature(f)
            spec: List[Dict[str, Any]] = []
            for p in sig.parameters.values():
                if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                    continue
                if p.name in ("img", "ctx"):
                    continue
                default = p.default if p.default is not inspect._empty else None
                typ = "float"
                if isinstance(default, bool):
                    typ = "bool"
                elif isinstance(default, int):
                    typ = "int"
                elif isinstance(default, str):
                    typ = "str"
                spec.append({"name": p.name, "type": typ, "default": default})
            return spec
        except Exception:
            return []