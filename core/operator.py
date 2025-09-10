"""
Operator registration and specification for the new glitchlab architecture.

This module introduces ``OperatorSpec`` and a global registry of
operators used to build and execute processing graphs. Each operator
encapsulates a pure transformation from one or more input artifacts
into a new artifact. Operators declare the number of inputs they
require and provide a factory function that accepts a context and
parameter values and returns a callable to perform the transformation.

By centralising operator specifications in this registry, the
application can dynamically discover available operators, validate
their parameters, and assemble them into arbitrary directed
acyclic graphs (DAGs) for generative experimentation.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Any, List, Optional

from .artifact import Artifact


@dataclass(frozen=True)
class OperatorSpec:
    """Describe a single operator in the processing graph.

    Parameters
    ----------
    name:
        Unique identifier for the operator. Used to reference the
        operator from presets and graph descriptions.
    fn_factory:
        A callable that accepts a context (any type) and arbitrary
        keyword parameters, and returns a function that takes an
        ``Artifact`` and returns a new ``Artifact``. This design
        allows for operator factories to capture the context and
        perform any initialisation only once.
    inputs:
        The number of input artifacts required. For simple unary
        operators, this will be ``1``. Operators that merge
        multiple inputs (e.g., blending two images) should set
        ``inputs`` accordingly.
    params_schema:
        A mapping from parameter names to default values or
        descriptors. This can be used by the UI to build dynamic
        forms and by presets for validation. If a parameter is
        optional, its default may be provided here.
    """
    name: str
    fn_factory: Callable[..., Callable[[Artifact], Artifact]]
    inputs: int = 1
    params_schema: Dict[str, Any] = None


# Global registry of available operators
OPERATORS: Dict[str, OperatorSpec] = {}


def register_operator(spec: OperatorSpec) -> None:
    """Register a new operator specification.

    If an operator with the same name already exists, a KeyError
    will be raised.
    """
    if spec.name in OPERATORS:
        raise KeyError(f"Operator '{spec.name}' is already registered")
    OPERATORS[spec.name] = spec


def get_operator(name: str) -> OperatorSpec:
    """Retrieve an operator specification by name.

    Raises KeyError if the operator is not registered.
    """
    if name not in OPERATORS:
        raise KeyError(f"Unknown operator '{name}'")
    return OPERATORS[name]


def available() -> List[str]:
    """Return a list of registered operator names."""
    return list(OPERATORS.keys())