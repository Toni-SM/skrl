from typing import Any, Literal, Optional, Sequence, Tuple, Union

import gymnasium
from gymnasium import spaces

import numpy as np
import torch

from skrl import config


def convert_gym_space(
    space: Optional["gym.Space"], *, squeeze_batch_dimension: bool = False
) -> Optional[gymnasium.Space]:
    """Converts a gym space to a gymnasium space.

    :param space: Gym space to convert to.
    :param squeeze_batch_dimension: Whether to remove fundamental spaces' first dimension.
        It currently affects ``Box`` space only.

    :return: Converted space, or ``None`` if the given space is ``None``.

    :raises ValueError: The given space is not supported.
    """
    import gym

    if space is None:
        return None
    # fundamental spaces
    # - Box
    elif isinstance(space, gym.spaces.Box):
        if squeeze_batch_dimension:
            return spaces.Box(low=space.low[0], high=space.high[0], shape=space.shape[1:], dtype=space.dtype)
        return spaces.Box(low=space.low, high=space.high, shape=space.shape, dtype=space.dtype)
    # - Discrete
    elif isinstance(space, gym.spaces.Discrete):
        return spaces.Discrete(n=space.n)
    # - MultiDiscrete
    elif isinstance(space, gym.spaces.MultiDiscrete):
        return spaces.MultiDiscrete(nvec=space.nvec)
    # composite spaces
    # - Tuple
    elif isinstance(space, gym.spaces.Tuple):
        return spaces.Tuple(
            spaces=tuple([convert_gym_space(s, squeeze_batch_dimension=squeeze_batch_dimension) for s in space.spaces])
        )
    # - Dict
    elif isinstance(space, gym.spaces.Dict):
        return spaces.Dict(
            spaces={
                k: convert_gym_space(v, squeeze_batch_dimension=squeeze_batch_dimension)
                for k, v in space.spaces.items()
            }
        )
    raise ValueError(f"Unsupported space ({space})")


def tensorize_space(space: Optional[spaces.Space], x: Any, *, device: Optional[Union[str, torch.device]] = None) -> Any:
    """Convert the sample/value items of a given gymnasium space to PyTorch tensors.

    :param space: Gymnasium space.
    :param x: Sample/value of the given space to tensorize to.
    :param device: Device on which a tensor/array is or will be allocated.
        This parameter is used when the space value is not a PyTorch tensor (e.g.: NumPy array, number).

    :return: Sample/value space with items converted to tensors,
        or ``None`` if the given space or the sample/value is ``None``.

    :raises ValueError: The given space or the sample/value type is not supported.
    """
    if space is None or x is None:
        return None
    device = config.torch.parse_device(device)
    # fundamental spaces
    # - Box
    if isinstance(space, spaces.Box):
        if isinstance(x, torch.Tensor):
            return x.reshape(-1, *space.shape)
        elif isinstance(x, np.ndarray):
            return torch.tensor(x, device=device, dtype=torch.float32).reshape(-1, *space.shape)
        else:
            raise ValueError(f"Unsupported type ({type(x)}) for the given space ({space})")
    # - Discrete
    elif isinstance(space, spaces.Discrete):
        if isinstance(x, torch.Tensor):
            return x.reshape(-1, 1)
        elif isinstance(x, np.ndarray):
            return torch.tensor(x, device=device, dtype=torch.int32).reshape(-1, 1)
        elif isinstance(x, np.number) or type(x) in [int, float]:
            return torch.tensor([x], device=device, dtype=torch.int32).reshape(-1, 1)
        else:
            raise ValueError(f"Unsupported type ({type(x)}) for the given space ({space})")
    # - MultiDiscrete
    elif isinstance(space, spaces.MultiDiscrete):
        if isinstance(x, torch.Tensor):
            return x.reshape(-1, *space.shape)
        elif isinstance(x, np.ndarray):
            return torch.tensor(x, device=device, dtype=torch.int32).reshape(-1, *space.shape)
        elif type(x) in [list, tuple]:
            return torch.tensor([x], device=device, dtype=torch.int32).reshape(-1, *space.shape)
        else:
            raise ValueError(f"Unsupported type ({type(x)}) for the given space ({space})")
    # composite spaces
    # - Tuple
    elif isinstance(space, spaces.Tuple):
        return tuple([tensorize_space(s, _x, device=device) for s, _x in zip(space, x)])
    # - Dict
    elif isinstance(space, spaces.Dict):
        return {k: tensorize_space(s, x[k], device=device) for k, s in space.items()}
    raise ValueError(f"Unsupported space ({space})")


def untensorize_space(space: Optional[spaces.Space], x: Any, *, squeeze_batch_dimension: bool = True) -> Any:
    """Convert a tensorized space to a gymnasium space with expected sample/value item types.

    :param space: Gymnasium space.
    :param x: Tensorized space (sample/value space where items are tensors).
    :param squeeze_batch_dimension: Whether to remove the batch dimension.
        If True, only the sample/value with a batch dimension of size 1 will be affected.

    :return: Sample/value space with expected item types,
        or ``None`` if the given space or the tensorized sample/value is ``None``.

    :raises ValueError: The given space or the sample/value type is not supported.
    """
    if space is None or x is None:
        return None
    # fundamental spaces
    # - Box
    elif isinstance(space, spaces.Box):
        if isinstance(x, torch.Tensor):
            # avoid TypeError: Got unsupported ScalarType BFloat16
            if x.dtype == torch.bfloat16:
                array = np.array(x.to(dtype=torch.float32).cpu().numpy(), dtype=space.dtype)
            else:
                array = np.array(x.cpu().numpy(), dtype=space.dtype)
            if squeeze_batch_dimension and array.shape[0] == 1:
                return array.reshape(space.shape)
            return array.reshape(-1, *space.shape)
        raise ValueError(f"Unsupported type ({type(x)}) for the given space ({space})")
    # - Discrete
    elif isinstance(space, spaces.Discrete):
        if isinstance(x, torch.Tensor):
            # avoid TypeError: Got unsupported ScalarType BFloat16
            if x.dtype == torch.bfloat16:
                array = np.array(x.to(dtype=torch.float32).cpu().numpy(), dtype=space.dtype)
            else:
                array = np.array(x.cpu().numpy(), dtype=space.dtype)
            if squeeze_batch_dimension and array.shape[0] == 1:
                return array.item()
            return array.reshape(-1, 1)
        raise ValueError(f"Unsupported type ({type(x)}) for the given space ({space})")
    # - MultiDiscrete
    elif isinstance(space, spaces.MultiDiscrete):
        if isinstance(x, torch.Tensor):
            # avoid TypeError: Got unsupported ScalarType BFloat16
            if x.dtype == torch.bfloat16:
                array = np.array(x.to(dtype=torch.float32).cpu().numpy(), dtype=space.dtype)
            else:
                array = np.array(x.cpu().numpy(), dtype=space.dtype)
            if squeeze_batch_dimension and array.shape[0] == 1:
                return array.reshape(space.nvec.shape)
            return array.reshape(-1, *space.nvec.shape)
        raise ValueError(f"Unsupported type ({type(x)}) for the given space ({space})")
    # composite spaces
    # - Tuple
    elif isinstance(space, spaces.Tuple):
        return tuple(
            [untensorize_space(s, _x, squeeze_batch_dimension=squeeze_batch_dimension) for s, _x in zip(space, x)]
        )
    # - Dict
    elif isinstance(space, spaces.Dict):
        return {
            k: untensorize_space(s, x[k], squeeze_batch_dimension=squeeze_batch_dimension) for k, s in space.items()
        }
    raise ValueError(f"Unsupported space ({space})")


def flatten_tensorized_space(x: Any) -> Optional[torch.Tensor]:
    """Flatten a tensorized space.

    :param x: Tensorized space sample/value.

    :return: A tensor. The returned tensor will have shape (batch, space size),
        or ``None`` if the given tensorized sample/value is ``None``.

    :raises ValueError: The given sample/value type is not supported.
    """
    if x is None:
        return None
    # fundamental spaces (Box, Discrete and MultiDiscrete)
    elif isinstance(x, torch.Tensor):
        return x.reshape(x.shape[0], -1) if x.ndim > 1 else x.reshape(1, -1)
    # composite spaces
    # - Tuple
    elif type(x) in [list, tuple]:
        return torch.cat([flatten_tensorized_space(_x) for _x in x], dim=-1)
    # - Dict
    elif isinstance(x, dict):
        return torch.cat([flatten_tensorized_space(x[k]) for k in sorted(x.keys())], dim=-1)
    raise ValueError(f"Unsupported sample/value type ({type(x)})")


def unflatten_tensorized_space(
    space: Optional[Union[spaces.Space, Sequence[int], int]], x: Optional[torch.Tensor]
) -> Any:
    """Unflatten a tensor to create a tensorized space.

    :param space: Gymnasium space.
    :param x: A tensor with shape (batch, space size).

    :return: Tensorized space value, or ``None`` if the given space or the tensor is ``None``.

    :raises ValueError: The given space is not supported.
    """
    if space is None or x is None:
        return None
    # fundamental spaces
    # - Box
    elif isinstance(space, spaces.Box):
        return x.reshape(-1, *space.shape)
    # - Discrete
    elif isinstance(space, spaces.Discrete):
        return x.reshape(-1, 1)
    # - MultiDiscrete
    elif isinstance(space, spaces.MultiDiscrete):
        return x.reshape(-1, *space.shape)
    # composite spaces
    # - Tuple
    elif isinstance(space, spaces.Tuple):
        start = 0
        output = []
        for s in space:
            end = start + compute_space_size(s, occupied_size=True)
            output.append(unflatten_tensorized_space(s, x[:, start:end]))
            start = end
        return output
    # - Dict
    elif isinstance(space, spaces.Dict):
        start = 0
        output = {}
        for k in sorted(space.keys()):
            end = start + compute_space_size(space[k], occupied_size=True)
            output[k] = unflatten_tensorized_space(space[k], x[:, start:end])
            start = end
        return output
    raise ValueError(f"Unsupported space ({space})")


def compute_space_size(space: Optional[Union[spaces.Space, Sequence[int], int]], *, occupied_size: bool = False) -> int:
    """Get the size (number of elements) of a space.

    :param space: Gymnasium space.
    :param occupied_size: Whether the number of elements occupied by the space is returned.
        It only affects :py:class:`~gymnasium.spaces.Discrete` (occupied space is 1),
        and :py:class:`~gymnasium.spaces.MultiDiscrete` (occupied space is the number of discrete spaces).

    :return: Size of the space (number of elements), or ``0`` if the given space is ``None``.
    """
    if space is None:
        return 0
    if occupied_size:
        # fundamental spaces
        # - Discrete
        if isinstance(space, spaces.Discrete):
            return 1
        # - MultiDiscrete
        elif isinstance(space, spaces.MultiDiscrete):
            return space.nvec.shape[0]
        # composite spaces
        # - Tuple
        elif isinstance(space, spaces.Tuple):
            return sum([compute_space_size(s, occupied_size=occupied_size) for s in space])
        # - Dict
        elif isinstance(space, spaces.Dict):
            return sum([compute_space_size(s, occupied_size=occupied_size) for s in space.values()])
    # non-gymnasium spaces
    if type(space) in [int, float]:
        return space
    elif type(space) in [tuple, list]:
        return int(np.prod(space))
    # gymnasium computation
    return gymnasium.spaces.flatdim(space)


def compute_space_limits(
    space: Optional[spaces.Space],
    *,
    occupied_size: bool = False,
    device: Optional[Union[str, torch.device]] = None,
    none_if_unbounded: Optional[Literal["both", "below", "above", "any"]] = None,
) -> Tuple[Union[torch.Tensor, None], Union[torch.Tensor, None]]:
    """Get the low and high limits of a space.

    .. note::

        Only the :py:class:`~gymnasium.spaces.Box` space has low and high limits.
        Other spaces are not bounded (low is ``-inf`` and high is ``inf``).

    :param space: Gymnasium space.
    :param occupied_size: Whether the limits are returned for the number of elements occupied by the space.
        It only affects :py:class:`~gymnasium.spaces.Discrete` (occupied space is 1),
        and :py:class:`~gymnasium.spaces.MultiDiscrete` (occupied space is the number of discrete spaces).
    :param device: Device on which a tensor/array is or will be allocated.
    :param none_if_unbounded: Whether to return ``None`` if the space is unbounded.
        If ``"both"``, low and high limits will be ``None`` if the space is unbounded in both directions.
        If ``"below"``, low limit will be ``None`` if the space is unbounded below.
        If ``"above"``, high limit will be ``None`` if the space is unbounded above.
        If ``"any"``, low or high limit will be ``None`` if the space is unbounded below or above, respectively.
        If not specified, low and high limits will be defined using ``-inf`` and ``inf``,
        respectively, when the space is unbounded.

    :return: Low and high limits of the space, or ``None`` if the given space is ``None``
        or unbounded (and ``none_if_unbounded`` is specified).
    """

    def _compute_limits(space: spaces.Space, *, low: np.ndarray, high: np.ndarray, index: int, occupied_size: bool):
        # fundamental spaces
        # - Box
        if isinstance(space, spaces.Box):
            size = compute_space_size(space, occupied_size=occupied_size)
            low[index : index + size] = space.low.flatten()
            high[index : index + size] = space.high.flatten()
        # composite spaces
        # - Tuple
        elif isinstance(space, spaces.Tuple):
            for s in space:
                _compute_limits(s, low=low, high=high, index=index, occupied_size=occupied_size)
                index += compute_space_size(s, occupied_size=occupied_size)
        # - Dict
        elif isinstance(space, spaces.Dict):
            for k in sorted(space.keys()):
                _compute_limits(space[k], low=low, high=high, index=index, occupied_size=occupied_size)
                index += compute_space_size(space[k], occupied_size=occupied_size)

    if space is None:
        return None, None
    size = compute_space_size(space, occupied_size=occupied_size)
    low = np.full((size,), -float("inf"), dtype=np.float32)
    high = np.full((size,), float("inf"), dtype=np.float32)
    _compute_limits(space, low=low, high=high, index=0, occupied_size=occupied_size)
    # check for unbounded spaces
    if none_if_unbounded == "both":
        if (np.isinf(low) & np.isinf(high)).all():
            low, high = None, None
    elif none_if_unbounded == "below":
        if np.isinf(low).all():
            low = None
    elif none_if_unbounded == "above":
        if np.isinf(high).all():
            high = None
    elif none_if_unbounded == "any":
        if np.isinf(low).all():
            low = None
        if np.isinf(high).all():
            high = None
    # convert to tensors
    device = config.torch.parse_device(device)
    if low is not None:
        low = torch.tensor(low, device=device)
    if high is not None:
        high = torch.tensor(high, device=device)
    return low, high


def sample_space(
    space: Optional[spaces.Space],
    *,
    batch_size: int = 1,
    backend: Literal["numpy", "native"] = "numpy",
    device: Optional[Union[str, torch.device]] = None,
) -> Any:
    """Generates a random sample from the specified space.

    :param space: Gymnasium space.
    :param batch_size: Size of the sampled batch.
    :param backend: Whether backend will be used to construct the fundamental spaces.
    :param device: Device on which a tensor/array is or will be allocated.
        This parameter is used when the backend is ``"native"`` (PyTorch).

    :return: Sample of the space, or ``None`` if the given space is ``None``.

    :raises ValueError: The given space or backend is not supported.
    """
    if space is None:
        return None
    device = config.torch.parse_device(device)
    # fundamental spaces
    # - Box
    if isinstance(space, spaces.Box):
        sample = gymnasium.vector.utils.batch_space(space, batch_size).sample()
        if backend == "numpy":
            return np.array(sample).reshape(batch_size, *space.shape)
        elif backend == "native":
            return torch.tensor(sample, device=device).reshape(batch_size, *space.shape)
        else:
            raise ValueError(f"Unsupported backend type ({backend})")
    # - Discrete
    elif isinstance(space, spaces.Discrete):
        sample = gymnasium.vector.utils.batch_space(space, batch_size).sample()
        if backend == "numpy":
            return np.array(sample).reshape(batch_size, -1)
        elif backend == "native":
            return torch.tensor(sample, device=device).reshape(batch_size, -1)
        else:
            raise ValueError(f"Unsupported backend type ({backend})")
    # - MultiDiscrete
    elif isinstance(space, spaces.MultiDiscrete):
        sample = gymnasium.vector.utils.batch_space(space, batch_size).sample()
        if backend == "numpy":
            return np.array(sample).reshape(batch_size, *space.nvec.shape)
        elif backend == "native":
            return torch.tensor(sample, device=device).reshape(batch_size, *space.nvec.shape)
        else:
            raise ValueError(f"Unsupported backend type ({backend})")
    # composite spaces
    # - Tuple
    elif isinstance(space, spaces.Tuple):
        return tuple([sample_space(s, batch_size=batch_size, backend=backend, device=device) for s in space])
    # - Dict
    elif isinstance(space, spaces.Dict):
        return {k: sample_space(s, batch_size=batch_size, backend=backend, device=device) for k, s in space.items()}
    raise ValueError(f"Unsupported space ({space})")
