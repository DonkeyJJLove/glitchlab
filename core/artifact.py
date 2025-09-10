"""
Artifact definitions for the new glitchlab architecture.

This module introduces the `Artifact` class, which represents
an immutable data container used throughout the processing graph.
Each artifact has a `kind` describing the type of data it carries
(for example ``"image"``, ``"array"``, or ``"error"``), the raw
``data`` payload, and a ``meta`` dictionary to hold provenance
information, metrics, masks, and any other contextual details.

Artifacts are designed to be passed between operators in a
directed acyclic graph (DAG). Operators may read an artifact's
metadata to inform their behaviour and should produce new artifacts
without mutating the input.

The ``with_metric`` method is a convenience helper for adding
metrics to an artifact's metadata while keeping the rest of the
metadata intact.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class Artifact:
    """Represents a unit of data flowing through the glitchlab graph.

    Attributes
    ----------
    kind:
        A short identifier describing the type of data stored in
        ``data``. Examples include ``"image"`` for RGB or RGBA
        arrays, ``"array"`` for generic NumPy arrays, ``"graph"`` for
        network structures, or ``"error"`` for error payloads.
    data:
        The underlying payload. The exact structure depends on
        ``kind``. For an image, this will typically be a NumPy
        ``ndarray``. For an error, it might be an exception info
        dictionary.
    meta:
        A dictionary for storing ancillary information such as
        provenance, masks, per-operator metrics, and any other
        context needed by downstream operators. The ``metrics``
        subkey should be a mapping from metric names to scalar
        values (or small structures) representing analysis results.
    """

    kind: str
    data: Any
    meta: Dict[str, Any] = field(default_factory=dict)

    def with_metric(self, key: str, value: Any) -> "Artifact":
        """Return a new artifact with an additional metric.

        This method does not mutate the current artifact; instead it
        creates a shallow copy of the metadata, adds or updates the
        ``metrics`` mapping with the given key/value, and returns
        a new ``Artifact`` instance sharing the original data.

        Parameters
        ----------
        key:
            The metric name to add or update.
        value:
            The corresponding metric value. Should be JSON-serialisable.

        Returns
        -------
        Artifact
            A new artifact instance with the updated metrics.
        """
        # Use a local variable to avoid mutating the original meta
        new_meta = dict(self.meta)
        metrics = new_meta.get("metrics")
        if metrics is None:
            metrics = {}
            new_meta["metrics"] = metrics
        # Update the metric
        metrics[key] = value
        return Artifact(self.kind, self.data, new_meta)