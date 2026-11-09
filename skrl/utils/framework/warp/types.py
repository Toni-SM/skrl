# Annotation type aliases. Each framework defines the same names to keep implementations aligned

from typing import TypeAlias, Union

import warp as wp


__all__ = ["Device", "Tensor"]


Tensor: TypeAlias = wp.array
"""Warp array."""

# `Union` instead of `|`: the latter fails at import time when the framework is mocked (e.g. Sphinx autodoc)
Device: TypeAlias = Union[str, wp.Device, None]
"""Device name (e.g. ``"cuda:0"``), Warp device instance, or ``None`` (default device)."""
