# Annotation type aliases. Each framework defines the same names to keep implementations aligned

from typing import TypeAlias, Union

import jax


__all__ = ["Device", "Tensor"]


Tensor: TypeAlias = jax.Array
"""JAX array."""

# `Union` instead of `|`: the latter fails at import time when the framework is mocked (e.g. Sphinx autodoc)
Device: TypeAlias = Union[str, jax.Device, None]
"""Device name (e.g. ``"cuda:0"``), JAX device instance, or ``None`` (default device)."""
