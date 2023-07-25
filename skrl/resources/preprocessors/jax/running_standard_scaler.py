from typing import Mapping, Optional, Tuple, Union

import gym
import gymnasium

import jax
import jax.numpy as jnp
import numpy as np

from skrl import config


# https://jax.readthedocs.io/en/latest/faq.html#strategy-1-jit-compiled-helper-function
@jax.jit
def _copyto(dst, src):
    """NumPy function copyto not yet implemented
    """
    return dst.at[:].set(src)


@jax.jit
def _parallel_variance(running_mean: jax.Array,
                       running_variance: jax.Array,
                       current_count: jax.Array,
                       array: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:  # yapf: disable
    # ddof = 1: https://github.com/pytorch/pytorch/issues/50010
    if array.ndim == 3:
        input_mean = jnp.mean(array, axis=(0, 1))
        input_var = jnp.var(array, axis=(0, 1), ddof=1)
        input_count = array.shape[0] * array.shape[1]
    else:
        input_mean = jnp.mean(array, axis=0)
        input_var = jnp.var(array, axis=0, ddof=1)
        input_count = array.shape[0]

    delta = input_mean - running_mean
    total_count = current_count + input_count
    M2 = (running_variance * current_count) + (input_var * input_count) \
       + delta ** 2 * current_count * input_count / total_count

    return running_mean + delta * input_count / total_count, M2 / total_count, total_count


@jax.jit
def _inverse(running_mean: jax.Array,
             running_variance: jax.Array,
             clip_threshold: float,
             array: jax.Array) -> jax.Array:  # yapf: disable
    return jnp.sqrt(running_variance) * jnp.clip(array, -clip_threshold, clip_threshold) + running_mean


@jax.jit
def _standardization(running_mean: jax.Array,
                     running_variance: jax.Array,
                     clip_threshold: float,
                     epsilon: float,
                     array: jax.Array) -> jax.Array:
    return jnp.clip((array - running_mean) / (jnp.sqrt(running_variance) + epsilon), -clip_threshold, clip_threshold)


class RunningStandardScaler:
    def __init__(self,
                 size: Union[int, Tuple[int], gym.Space, gymnasium.Space],
                 epsilon: float = 1e-8,
                 clip_threshold: float = 5.0,
                 device: Optional[Union[str, jax.Device]] = None) -> None:
        """Standardize the input data by removing the mean and scaling by the standard deviation

        The implementation is adapted from the rl_games library
        (https://github.com/Denys88/rl_games/blob/master/rl_games/algos_torch/running_mean_std.py)

        Example::

            >>> running_standard_scaler = RunningStandardScaler(size=2)
            >>> data = jax.random.uniform(jax.random.PRNGKey(0), (3,2))  # tensor of shape (N, 2)
            >>> running_standard_scaler(data)
            Array([[0.57450044, 0.09968603],
                   [0.7419659 , 0.8941783 ],
                   [0.59656656, 0.45325184]], dtype=float32)

        :param size: Size of the input space
        :type size: int, tuple or list of integers, gym.Space, or gymnasium.Space
        :param epsilon: Small number to avoid division by zero (default: ``1e-8``)
        :type epsilon: float
        :param clip_threshold: Threshold to clip the data (default: ``5.0``)
        :type clip_threshold: float
        :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda"`` if available or ``"cpu"``
        :type device: str or jax.Device, optional
        """
        self._jax = config.jax.backend == "jax"

        self.epsilon = epsilon
        self.clip_threshold = clip_threshold
        if device is None:
            self.device = jax.devices()[0]
        else:
            self.device = device if isinstance(device, jax.Device) else jax.devices(device)[0]

        size = self._get_space_size(size)

        if self._jax:
            self.running_mean = jnp.zeros(size, dtype=jnp.float32)
            self.running_variance = jnp.ones(size, dtype=jnp.float32)
            self.current_count = jnp.ones((1,), dtype=jnp.float32)
        else:
            self.running_mean = np.zeros(size, dtype=np.float32)
            self.running_variance = np.ones(size, dtype=np.float32)
            self.current_count = np.ones((1,), dtype=np.float32)

    @property
    def state_dict(self) -> Mapping[str, Union[np.ndarray, jax.Array]]:
        """Dictionary containing references to the whole state of the module
        """
        class _StateDict:
            def __init__(self, params):
                self.params = params

            def replace(self, params):
                return params

        return _StateDict({
            "running_mean": self.running_mean,
            "running_variance": self.running_variance,
            "current_count": self.current_count
        })

    @state_dict.setter
    def state_dict(self, value: Mapping[str, Union[np.ndarray, jax.Array]]) -> None:
        if self._jax:
            self.running_mean = _copyto(self.running_mean, value["running_mean"])
            self.running_variance = _copyto(self.running_variance, value["running_variance"])
            self.current_count = _copyto(self.current_count, value["current_count"])
        else:
            np.copyto(self.running_mean, value["running_mean"])
            np.copyto(self.running_variance, value["running_variance"])
            np.copyto(self.current_count, value["current_count"])

    def _get_space_size(self, space: Union[int, Tuple[int], gym.Space, gymnasium.Space]) -> int:
        """Get the size (number of elements) of a space

        :param space: Space or shape from which to obtain the number of elements
        :type space: int, tuple or list of integers, gym.Space, or gymnasium.Space

        :raises ValueError: If the space is not supported

        :return: Size of the space data
        :rtype: Space size (number of elements)
        """
        if type(space) in [int, float]:
            return int(space)
        elif type(space) in [tuple, list]:
            return np.prod(space)
        elif issubclass(type(space), gym.Space):
            if issubclass(type(space), gym.spaces.Discrete):
                return 1
            elif issubclass(type(space), gym.spaces.Box):
                return np.prod(space.shape)
            elif issubclass(type(space), gym.spaces.Dict):
                return sum([self._get_space_size(space.spaces[key]) for key in space.spaces])
        elif issubclass(type(space), gymnasium.Space):
            if issubclass(type(space), gymnasium.spaces.Discrete):
                return 1
            elif issubclass(type(space), gymnasium.spaces.Box):
                return np.prod(space.shape)
            elif issubclass(type(space), gymnasium.spaces.Dict):
                return sum([self._get_space_size(space.spaces[key]) for key in space.spaces])
        raise ValueError(f"Space type {type(space)} not supported")

    def _parallel_variance(self,
                           input_mean: Union[np.ndarray, jax.Array],
                           input_var: Union[np.ndarray, jax.Array],
                           input_count: int) -> None:
        """Update internal variables using the parallel algorithm for computing variance

        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param input_mean: Mean of the input data
        :type input_mean: np.ndarray or jax.Array
        :param input_var: Variance of the input data
        :type input_var: np.ndarray or jax.Array
        :param input_count: Batch size of the input data
        :type input_count: int
        """
        delta = input_mean - self.running_mean
        total_count = self.current_count + input_count
        M2 = (self.running_variance * self.current_count) + (input_var * input_count) \
            + delta ** 2 * self.current_count * input_count / total_count

        # update internal variables
        self.running_mean = self.running_mean + delta * input_count / total_count
        self.running_variance = M2 / total_count
        self.current_count = total_count

    def __call__(self,
                 x: Union[np.ndarray, jax.Array],
                 train: bool = False,
                 inverse: bool = False) -> Union[np.ndarray, jax.Array]:
        """Forward pass of the standardizer

        Example::

            >>> x = jax.random.uniform(jax.random.PRNGKey(0), (3,2))
            >>> running_standard_scaler(x)
            Array([[0.57450044, 0.09968603],
                   [0.7419659 , 0.8941783 ],
                   [0.59656656, 0.45325184]], dtype=float32)

            >>> running_standard_scaler(x, train=True)
            Array([[ 0.167439  , -0.4292293 ],
                   [ 0.45878986,  0.8719094 ],
                   [ 0.20582889,  0.14980486]], dtype=float32)

            >>> running_standard_scaler(x, inverse=True)
            Array([[0.80847514, 0.4226486 ],
                   [0.9047325 , 0.90777594],
                   [0.8211585 , 0.6385405 ]], dtype=float32)

        :param x: Input tensor
        :type x: np.ndarray or jax.Array
        :param train: Whether to train the standardizer (default: ``False``)
        :type train: bool, optional
        :param inverse: Whether to inverse the standardizer to scale back the data (default: ``False``)
        :type inverse: bool, optional

        :return: Standardized tensor
        :rtype: np.ndarray or jax.Array
        """
        if train:
            if self._jax:
                self.running_mean, self.running_variance, self.current_count = \
                    _parallel_variance(self.running_mean, self.running_variance, self.current_count, x)
            else:
                # ddof = 1: https://github.com/pytorch/pytorch/issues/50010
                if x.ndim == 3:
                    self._parallel_variance(np.mean(x, axis=(0, 1)),
                                            np.var(x, axis=(0, 1), ddof=1),
                                            x.shape[0] * x.shape[1])
                else:
                    self._parallel_variance(np.mean(x, axis=0), np.var(x, axis=0, ddof=1), x.shape[0])

        # scale back the data to the original representation
        if inverse:
            if self._jax:
                return _inverse(self.running_mean, self.running_variance, self.clip_threshold, x)
            return np.sqrt(self.running_variance) * np.clip(x, -self.clip_threshold,
                                                            self.clip_threshold) + self.running_mean
        # standardization by centering and scaling
        if self._jax:
            return _standardization(self.running_mean, self.running_variance, self.clip_threshold, self.epsilon, x)
        return np.clip((x - self.running_mean) / (np.sqrt(self.running_variance) + self.epsilon),
                       a_min=-self.clip_threshold,
                       a_max=self.clip_threshold)
