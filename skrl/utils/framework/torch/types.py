# Annotation type aliases. Each framework defines the same names to keep implementations aligned

from typing import TypeAlias

import torch


__all__ = ["Tensor"]


#: PyTorch tensor.
Tensor: TypeAlias = torch.Tensor
