from typing import Any, Literal, Optional, Sequence, Union

import gymnasium
from gymnasium import spaces

import jax
import jax.numpy as jnp
import numpy as np

from skrl import config


def convert_gym_space(space: "gym.Space", squeeze_batch_dimension: bool = False) -> gymnasium.Space:
    """Converts a gym space to a gymnasium space.

    :param space: Gym space to convert to.
    :param squeeze_batch_dimension: Whether to remove fundamental spaces' first dimension.
                                    It currently affects ``Box`` space only.

    :raises ValueError: The given space is not supported.

    :return: Converted space.
    """
    import gym

    if isinstance(space, gym.spaces.Discrete):
        return spaces.Discrete(n=space.n)
    elif isinstance(space, gym.spaces.Box):
        if squeeze_batch_dimension:
            return spaces.Box(low=space.low[0], high=space.high[0], shape=space.shape[1:], dtype=space.dtype)
        return spaces.Box(low=space.low, high=space.high, shape=space.shape, dtype=space.dtype)
    elif isinstance(space, gym.spaces.MultiDiscrete):
        return spaces.MultiDiscrete(nvec=space.nvec)
    elif isinstance(space, gym.spaces.Tuple):
        return spaces.Tuple(spaces=tuple(map(convert_gym_space, space.spaces)))
    elif isinstance(space, gym.spaces.Dict):
        return spaces.Dict(spaces={k: convert_gym_space(v) for k, v in space.spaces.items()})
    raise ValueError(f"Unsupported space ({space})")

def tensorize_space(space: spaces.Space, x: Any, device: Optional[Union[str, jax.Device]] = None) -> Any:
    """Convert the sample/value items of a given gymnasium space to JAX array.

    Fundamental spaces (:py:class:`~gymnasium.spaces.Box`, :py:class:`~gymnasium.spaces.Discrete`, and
    :py:class:`~gymnasium.spaces.MultiDiscrete`) are converted to :py:class:`~jax.Array` with shape
    (-1, space's shape). Composite spaces (:py:class:`~gymnasium.spaces.Dict` and :py:class:`~gymnasium.spaces.Tuple`)
    are converted by recursively calling this function on their elements.

    :param space: Gymnasium space.
    :param x: Sample/value of the given space to tensorize to.
    :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                   This parameter is used when the space value is not a JAX array (e.g.: numpy array, number).

    :raises ValueError: The given space or the sample/value type is not supported.

    :return: Sample/value space with items converted to tensors.
    """
    if x is None:
        return None
    # fundamental spaces
    # Box
    if isinstance(space, spaces.Box):
        if isinstance(x, jax.Array):
            return x.reshape(-1, *space.shape)
        elif isinstance(x, np.ndarray):
            return jax.device_put(x.reshape(-1, *space.shape), config.jax.parse_device(device))
        else:
            raise ValueError(f"Unsupported type ({type(x)}) for the given space ({space})")
    # Discrete
    elif isinstance(space, spaces.Discrete):
        if isinstance(x, jax.Array):
            return x.reshape(-1, 1)
        elif isinstance(x, np.ndarray):
            return jax.device_put(x.reshape(-1, 1), config.jax.parse_device(device))
        elif isinstance(x, np.number) or type(x) in [int, float]:
            return jnp.array([x], device=device, dtype=jnp.int32).reshape(-1, 1)
        else:
            raise ValueError(f"Unsupported type ({type(x)}) for the given space ({space})")
    # MultiDiscrete
    elif isinstance(space, spaces.MultiDiscrete):
        if isinstance(x, jax.Array):
            return x.reshape(-1, *space.shape)
        elif isinstance(x, np.ndarray):
            return jax.device_put(x.reshape(-1, *space.shape), config.jax.parse_device(device))
        elif type(x) in [list, tuple]:
            return jnp.array(x, device=device, dtype=jnp.int32).reshape(-1, *space.shape)
        else:
            raise ValueError(f"Unsupported type ({type(x)}) for the given space ({space})")
    # composite spaces
    # Dict
    elif isinstance(space, spaces.Dict):
        return {k: tensorize_space(s, x[k], device) for k, s in space.items()}
    # Tuple
    elif isinstance(space, spaces.Tuple):
        return tuple([tensorize_space(s, _x, device) for s, _x in zip(space, x)])
    raise ValueError(f"Unsupported space ({space})")

def untensorize_space(space: spaces.Space, x: Any, squeeze_batch_dimension: bool = True) -> Any:
    """Convert a tensorized space to a gymnasium space with expected sample/value item types.

    :param space: Gymnasium space.
    :param x: Tensorized space (Sample/value space where items are tensors).
    :param squeeze_batch_dimension: Whether to remove the batch dimension. If True, only the
                                    sample/value with a batch dimension of size 1 will be affected

    :raises ValueError: The given space or the sample/value type is not supported.

    :return: Sample/value space with expected item types.
    """
    if x is None:
        return None
    # fundamental spaces
    # Box
    if isinstance(space, spaces.Box):
        if isinstance(x, jax.Array):
            array = np.asarray(jax.device_get(x), dtype=space.dtype)
            if squeeze_batch_dimension and array.shape[0] == 1:
                return array.reshape(space.shape)
            return array.reshape(-1, *space.shape)
        raise ValueError(f"Unsupported type ({type(x)}) for the given space ({space})")
    # Discrete
    elif isinstance(space, spaces.Discrete):
        if isinstance(x, jax.Array):
            array = np.asarray(jax.device_get(x), dtype=space.dtype)
            if squeeze_batch_dimension and array.shape[0] == 1:
                return array.item()
            return array.reshape(-1, 1)
        raise ValueError(f"Unsupported type ({type(x)}) for the given space ({space})")
    # MultiDiscrete
    elif isinstance(space, spaces.MultiDiscrete):
        if isinstance(x, jax.Array):
            array = np.asarray(jax.device_get(x), dtype=space.dtype)
            if squeeze_batch_dimension and array.shape[0] == 1:
                return array.reshape(space.nvec.shape)
            return array.reshape(-1, *space.nvec.shape)
        raise ValueError(f"Unsupported type ({type(x)}) for the given space ({space})")
    # composite spaces
    # Dict
    elif isinstance(space, spaces.Dict):
        return {k: untensorize_space(s, x[k], squeeze_batch_dimension) for k, s in space.items()}
    # Tuple
    elif isinstance(space, spaces.Tuple):
        return tuple([untensorize_space(s, _x, squeeze_batch_dimension) for s, _x in zip(space, x)])
    raise ValueError(f"Unsupported space ({space})")

def flatten_tensorized_space(x: Any) -> jax.Array:
    """Flatten a tensorized space.

    :param x: Tensorized space sample/value.

    :raises ValueError: The given sample/value type is not supported.

    :return: A tensor. The returned tensor will have shape (batch, space size).
    """
    # fundamental spaces
    # Box / Discrete / MultiDiscrete
    if isinstance(x, jax.Array):
        return x.reshape(x.shape[0], -1) if x.ndim > 1 else x.reshape(1, -1)
    # composite spaces
    # Dict
    elif isinstance(x, dict):
        return jnp.concatenate([flatten_tensorized_space(x[k])for k in sorted(x.keys())], axis=-1)
    # Tuple
    elif type(x) in [list, tuple]:
        return jnp.concatenate([flatten_tensorized_space(_x) for _x in x], axis=-1)
    raise ValueError(f"Unsupported sample/value type ({type(x)})")

def unflatten_tensorized_space(space: Union[spaces.Space, Sequence[int], int], x: jax.Array) -> Any:
    """Unflatten a tensor to create a tensorized space.

    :param space: Gymnasium space.
    :param x: A tensor with shape (batch, space size).

    :raises ValueError: The given space is not supported.

    :return: Tensorized space value.
    """
    if x is None:
        return None
    # fundamental spaces
    # Box
    if isinstance(space, spaces.Box):
        return x.reshape(-1, *space.shape)
    # Discrete
    elif isinstance(space, spaces.Discrete):
        return x.reshape(-1, 1)
    # MultiDiscrete
    elif isinstance(space, spaces.MultiDiscrete):
        return x.reshape(-1, *space.shape)
    # composite spaces
    # Dict
    elif isinstance(space, spaces.Dict):
        start = 0
        output = {}
        for k in sorted(space.keys()):
            end = start + compute_space_size(space[k], occupied_size=True)
            output[k] = unflatten_tensorized_space(space[k], x[:, start:end])
            start = end
        return output
    # Tuple
    elif isinstance(space, spaces.Tuple):
        start = 0
        output = []
        for s in space:
            end = start + compute_space_size(s, occupied_size=True)
            output.append(unflatten_tensorized_space(s, x[:, start:end]))
            start = end
        return output
    raise ValueError(f"Unsupported space ({space})")

def compute_space_size(space: Union[spaces.Space, Sequence[int], int], occupied_size: bool = False) -> int:
    """Get the size (number of elements) of a space.

    :param space: Gymnasium space.
    :param occupied_size: Whether the number of elements occupied by the space is returned (default: ``False``).
                          It only affects :py:class:`~gymnasium.spaces.Discrete` (occupied space is 1),
                          and :py:class:`~gymnasium.spaces.MultiDiscrete` (occupied space is the number of discrete spaces).

    :return: Size of the space (number of elements).
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

def sample_space(space: spaces.Space, batch_size: int = 1, backend: str = Literal["numpy", "jax"], device = None) -> Any:
    """Generates a random sample from the specified space.

    :param space: Gymnasium space.
    :param batch_size: Size of the sampled batch (default: ``1``).
    :param backend: Whether backend will be used to construct the fundamental spaces (default: ``"numpy"``).
    :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                   This parameter is used when the backend is ``"jax"``.

    :raises ValueError: The given space or backend is not supported.

    :return: Sample of the space
    """
    # fundamental spaces
    # Box
    if isinstance(space, spaces.Box):
        if backend == "numpy":
            return np.stack([space.sample() for _ in range(batch_size)])
        elif backend == "jax":
            return jnp.array(np.stack([space.sample() for _ in range(batch_size)]), device=device)
        else:
            raise ValueError(f"Unsupported backend type ({backend})")
    # Discrete
    elif isinstance(space, spaces.Discrete):
        if backend == "numpy":
            return np.stack([[space.sample()] for _ in range(batch_size)])
        elif backend == "jax":
            return jnp.array(np.stack([[space.sample()] for _ in range(batch_size)]), device=device)
        else:
            raise ValueError(f"Unsupported backend type ({backend})")
    # MultiDiscrete
    elif isinstance(space, spaces.MultiDiscrete):
        if backend == "numpy":
            return np.stack([space.sample() for _ in range(batch_size)])
        elif backend == "jax":
            return jnp.array(np.stack([space.sample() for _ in range(batch_size)]), device=device)
        else:
            raise ValueError(f"Unsupported backend type ({backend})")
    # composite spaces
    # Dict
    elif isinstance(space, spaces.Dict):
        return {k: sample_space(s, batch_size, backend, device) for k, s in space.items()}
    # Tuple
    elif isinstance(space, spaces.Tuple):
        return tuple([sample_space(s, batch_size, backend, device) for s in space])
    raise ValueError(f"Unsupported space ({space})")
