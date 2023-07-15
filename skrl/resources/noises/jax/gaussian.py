from typing import Optional, Union, Tuple

import numpy as np
from functools import partial

import jax
import jaxlib
import jax.numpy as jnp

from skrl.resources.noises.jax import Noise

from skrl import config


# https://jax.readthedocs.io/en/latest/faq.html#strategy-1-jit-compiled-helper-function
@partial(jax.jit, static_argnames=("shape"))
def _sample(mean, std, key, iterator, shape):
    subkey = jax.random.fold_in(key, iterator)
    return jax.random.normal(subkey, shape) * std + mean


class GaussianNoise(Noise):
    def __init__(self, mean: float, std: float, device: Optional[Union[str, jaxlib.xla_extension.Device]] = None) -> None:
        """Class representing a Gaussian noise

        :param mean: Mean of the normal distribution
        :type mean: float
        :param std: Standard deviation of the normal distribution
        :type std: float
        :param device: Device on which a jax array is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda:0"`` if available or ``"cpu"``
        :type device: str or jaxlib.xla_extension.Device, optional

        Example::

            >>> noise = GaussianNoise(mean=0, std=1)
        """
        super().__init__(device)

        if self._jax:
            self.mean = jnp.array(mean)
            self.std = jnp.array(std)

            self._i = 0
            self._key = config.jax.key
        else:
            self.mean = np.array(mean)
            self.std = np.array(std)

    def sample(self, size: Tuple[int]) -> Union[np.ndarray, jnp.ndarray]:
        """Sample a Gaussian noise

        :param size: Shape of the sampled tensor
        :type size: tuple or list of integers

        :return: Sampled noise
        :rtype: np.ndarray or jnp.ndarray

        Example::

            >>> noise.sample((3, 2))
            Array([[ 0.01878439, -0.12833427],
                   [ 0.06494182,  0.12490594],
                   [ 0.024447  , -0.01174496]], dtype=float32)

            >>> x = jax.random.uniform(jax.random.PRNGKey(0), (3, 2))
            >>> noise.sample(x.shape)
            Array([[ 0.17988093, -1.2289404 ],
                   [ 0.6218886 ,  1.1961104 ],
                   [ 0.23410667, -0.11247082]], dtype=float32)
        """
        if self._jax:
            self._i += 1
            return _sample(self.mean, self.std, self._key, self._i, size)
        return np.random.normal(self.mean, self.std, size)
