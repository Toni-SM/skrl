# Annotation type aliases. Each framework defines the same names to keep implementations aligned

from typing import TypeAlias

import jax


__all__ = ["Tensor"]


#: JAX array.
Tensor: TypeAlias = jax.Array
