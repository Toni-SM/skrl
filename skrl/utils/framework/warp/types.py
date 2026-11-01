# Annotation type aliases. Each framework defines the same names to keep implementations aligned

from typing import TypeAlias

import warp as wp


__all__ = ["Tensor"]


#: Warp array.
Tensor: TypeAlias = wp.array
