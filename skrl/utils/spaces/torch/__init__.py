from typing import Any, Optional, Sequence, Union

import gymnasium
from gymnasium import spaces

import numpy as np
import torch


__all__ = ["tensorize_space", "flatten_tensorized_space", "compute_space_size"]


def tensorize_space(space: spaces.Space, x: Any, device: Optional[Union[str, torch.device]] = None) -> Any:
    """Convert the sample/value items of a given gymnasium space to PyTorch tensors.

    Fundamental spaces (:py:class:`~gymnasium.spaces.Box`, :py:class:`~gymnasium.spaces.Discrete`, and
    :py:class:`~gymnasium.spaces.MultiDiscrete`) are converted to :py:class:`~torch.Tensor` with shape
    (-1, space's shape). Composite spaces (:py:class:`~gymnasium.spaces.Dict`, :py:class:`~gymnasium.spaces.Tuple`,
    and :py:class:`~gymnasium.spaces.Sequence`) are converted by recursively calling this function on their elements.

    :param space: Gymnasium space
    :param x: Sample/value of the given space to tensorize to
    :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                   This parameter is used when the space value is not a PyTorch tensor (e.g.: numpy array, number)

    :raises ValueError: The conversion of the sample/value type is not supported for the given space

    :return: Sample/value space with items converted to tensors
    """
    # fundamental spaces
    # Box
    if isinstance(space, spaces.Box):
        if isinstance(x, torch.Tensor):
            return x.view(-1, *space.shape)
        elif isinstance(x, np.ndarray):
            return torch.tensor(x, device=device, dtype=torch.float32).view(-1, *space.shape)
        else:
            raise ValueError(f"Unsupported type ({type(x)}) for the given space ({space})")
    # Discrete
    elif isinstance(space, spaces.Discrete):
        if isinstance(x, torch.Tensor):
            return x.view(-1, 1)
        elif isinstance(x, np.ndarray):
            return torch.tensor(x, device=device, dtype=torch.int32).view(-1, 1)
        elif isinstance(x, np.number) or type(x) in [int, float]:
            return torch.tensor([x], device=device, dtype=torch.int32).view(-1, 1)
        else:
            raise ValueError(f"Unsupported type ({type(x)}) for the given space ({space})")
    # MultiDiscrete
    elif isinstance(space, spaces.MultiDiscrete):
        if isinstance(x, torch.Tensor):
            return x.view(-1, *space.shape)
        elif isinstance(x, np.ndarray):
            return torch.tensor(x, device=device, dtype=torch.int32).view(-1, *space.shape)
        elif type(x) in [list, tuple]:
            return torch.tensor([x], device=device, dtype=torch.int32).view(-1, *space.shape)
        else:
            raise ValueError(f"Unsupported type ({type(x)}) for the given space ({space})")
    # composite spaces
    # Dict
    elif isinstance(space, spaces.Dict):
        return {k: tensorize_space(s, x[k], device) for k, s in space.items()}
    # Tuple
    elif isinstance(space, spaces.Tuple):
        return tuple([tensorize_space(s, _x, device) for s, _x in zip(space, x)])
    # Sequence
    elif isinstance(space, spaces.Sequence):
        return tuple([tensorize_space(space.feature_space, _x, device) for _x in x])

def flatten_tensorized_space(x: Any) -> torch.Tensor:
    """Flatten a tensorized space (see :py:func:`~skrl.utils.spaces.torch.tensorize_space`).

    :param x: Tensorized space sample/value

    :return: A tensor. The returned tensor will have shape (batch, space size)
    """
    if isinstance(x, torch.Tensor):
        return x.view(x.shape[0], -1) if x.ndim > 1 else x.view(1, -1)
    elif isinstance(x, dict):
        return torch.cat([flatten_tensorized_space(x[k])for k in sorted(x.keys())], dim=-1)
    elif type(x) in [list, tuple]:
        return torch.cat([flatten_tensorized_space(_x) for _x in x], dim=-1)

def compute_space_size(space: Union[spaces.Space, Sequence[int], int], occupied_size: bool = False) -> int:
    """Get the size (number of elements) of a space

    :param space: Gymnasium space
    :param occupied_size: Whether the number of elements occupied by the space is returned (default: ``False``).
                          It only affects :py:class:`~gymnasium.spaces.Discrete` (occupied space is 1),
                          and :py:class:`~gymnasium.spaces.MultiDiscrete` (occupied space is the number of discrete spaces)

    :return: Size of the space (number of elements)
    """
    if occupied_size:
        # fundamental spaces
        # Discrete
        if isinstance(space, spaces.Discrete):
            return 1
        # MultiDiscrete
        elif isinstance(space, spaces.MultiDiscrete):
            return space.nvec.shape[0]
        # composite spaces
        # Dict
        elif isinstance(space, spaces.Dict):
            return sum([compute_space_size(s, occupied_size) for s in space.values()])
        # Tuple
        elif isinstance(space, spaces.Tuple):
            return sum([compute_space_size(s, occupied_size) for s in space])
    # non-gymnasium spaces
    if type(space) in [int, float]:
        return space
    elif type(space) in [tuple, list]:
        return int(np.prod(space))
    # gymnasium computation
    return gymnasium.spaces.flatdim(space)
