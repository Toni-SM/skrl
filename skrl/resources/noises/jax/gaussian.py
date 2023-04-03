from typing import Optional, Union, Tuple

import numpy as np

import jax
import jaxlib
import jax.numpy as jnp

from skrl import config
from skrl.resources.noises.jax import Noise


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

        # normal distribution
        if config.jax.backend == "jax":
            class _Normal:
                def __init__(self, loc, scale):
                    self._loc = loc
                    self._scale = scale

                    self._i = 0
                    self._key = jax.random.PRNGKey(0)

                def sample(self, size):
                    self._i += 1
                    subkey = jax.random.fold_in(self._key, self._i)
                    return jax.random.normal(subkey, size) * self._scale + self._loc

            # just-in-time compilation with XLA
            self.sample = jax.jit(self.sample, static_argnames=("size"))

        elif config.jax.backend == "numpy":
            class _Normal:
                def __init__(self, loc, scale):
                    self._loc = loc
                    self._scale = scale

                def sample(self, size):
                    return np.random.normal(self._loc, self._scale, size)

        self.distribution = _Normal(loc=mean, scale=std)

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
        return self.distribution.sample(size)
