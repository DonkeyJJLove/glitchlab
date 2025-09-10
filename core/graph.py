"""
Graph specification and execution engine for glitchlab.

This module defines the data structures used to represent a
processing graph (DAG) and provides a simple evaluator to run
graphs. A graph consists of a set of global parameters and a
list of node definitions. Each node refers to an operator by
name and specifies local parameters and input dependencies.

During evaluation, global parameters can be referenced in the
local parameter values by name or via the ``@map(x, a, b)``
expression, which linearly maps a normalised global value ``x``
to the range ``[a, b]``. For example ``@map(depth, 20, 100)``
will take the global parameter ``depth`` (assumed between 0 and
1) and produce a value between 20 and 100.

Graphs are evaluated in the order nodes are defined; users
should ensure that dependencies appear before their dependants.
Output artifacts are cached under the node id for reuse by
subsequent nodes.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Callable, Optional
import re

from .artifact import Artifact
from .operator import OperatorSpec, get_operator, available as available_operators


@dataclass
class Node:
    """A single node in a processing graph.

    Each node applies a specified operator to its inputs with
    local parameters. The operator must be registered in the
    operator registry. Inputs are given as a list of strings; each
    string references either an input alias (like ``"image_in"``)
    or the ``id`` of another node in the graph.
    """

    id: str
    op: str
    inputs: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphSpec:
    """A specification of a processing graph.

    Attributes
    ----------
    globals:
        Mapping of global parameter names to values. Values are
        assumed to be floats in the range [0, 1] for mapping
        operations, but this is not enforced.
    nodes:
        A sequence of node definitions. The order of nodes
        dictates the evaluation order and should satisfy all
        dependencies.
    """

    globals: Dict[str, Any] = field(default_factory=dict)
    nodes: List[Node] = field(default_factory=list)


class GraphRunner:
    """Evaluate a processing graph on a set of input artifacts.

    This runner interprets parameter expressions referencing
    global variables and caches intermediate artifacts under
    node identifiers. It is a minimal execution engine and does
    not implement sophisticated scheduling or parallelism.
    """

    def __init__(self, spec: GraphSpec, ctx: Any):
        self.spec = spec
        self.ctx = ctx
        self.cache: Dict[str, Artifact] = {}

    def _eval_expr(self, expr: Any) -> Any:
        """Evaluate a parameter expression.

        Parameter values may be either raw values, names of
        global parameters, or mapping expressions of the form
        ``@map(var, a, b)``. Mapping expressions linearly map
        ``var`` (looked up in ``spec.globals``) to the interval
        [a, b]. If the expression is not a string or does not
        start with ``@map``, the value is returned unchanged.
        """
        if isinstance(expr, str):
            # direct global reference
            if expr in self.spec.globals:
                return self.spec.globals[expr]
            # mapping pattern
            m = re.match(r"@map\(([^,]+),\s*([^,]+),\s*([^\)]+)\)", expr)
            if m:
                var_name = m.group(1).strip()
                a = float(m.group(2))
                b = float(m.group(3))
                base = float(self.spec.globals.get(var_name, 0.0))
                return a + (b - a) * base
        return expr

    def run(self, inputs: Dict[str, Artifact]) -> Dict[str, Artifact]:
        """Execute the graph and return a mapping of node ids to artifacts."""
        # Start with provided inputs; e.g. {"image_in": artifact}
        for node in self.spec.nodes:
            # Resolve operator spec
            spec = get_operator(node.op)
            # Evaluate local parameters
            local_params = {}
            if node.params:
                for k, v in node.params.items():
                    local_params[k] = self._eval_expr(v)
            # Build transformer from factory
            transformer = spec.fn_factory(self.ctx, **(local_params or {}))
            # Gather input artifacts
            input_art: Optional[Artifact] = None
            if spec.inputs == 0:
                # Operators with zero inputs (e.g. generators) ignore input_art
                input_art = None
            else:
                # For unary operators, take the first available input
                if not node.inputs:
                    # fallback to a default alias
                    input_art = inputs.get("image_in")
                else:
                    src = node.inputs[0]
                    if src in inputs:
                        input_art = inputs[src]
                    elif src in self.cache:
                        input_art = self.cache[src]
                    else:
                        raise KeyError(f"Missing input '{src}' for node {node.id}")
            # Execute operator
            try:
                out_art = transformer(input_art) if input_art is not None else transformer(None)
            except Exception as e:
                # Represent the error as a special artifact
                out_art = Artifact("error", {"node": node.id, "exc": repr(e)}, {"origin": node.op})
            # Cache result under node id
            self.cache[node.id] = out_art
        return self.cache