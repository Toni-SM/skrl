# Annotation type aliases. Each framework defines the same names to keep implementations aligned

from typing import TypeAlias, Union

import torch


__all__ = ["Device", "Tensor"]


Tensor: TypeAlias = torch.Tensor
"""PyTorch tensor."""

# `Union` instead of `|`: the latter fails at import time when the framework is mocked (e.g. Sphinx autodoc)
Device: TypeAlias = Union[str, torch.device, None]
"""Device name (e.g. ``"cuda:0"``), PyTorch device instance, or ``None`` (default device)."""
