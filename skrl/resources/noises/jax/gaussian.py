from typing import Optional, Tuple, Union

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from skrl import config
from skrl.resources.noises.jax import Noise


# https://jax.readthedocs.io/en/latest/faq.html#strategy-1-jit-compiled-helper-function
@partial(jax.jit, static_argnames=("shape"))
def _sample(mean, std, key, iterator, shape):
    subkey = jax.random.fold_in(key, iterator)
    return jax.random.normal(subkey, shape) * std + mean


class GaussianNoise(Noise):
    def __init__(self, *, mean: float, std: float, device: Optional[Union[str, jax.Device]] = None) -> None:
        """Gaussian noise.

        :param mean: Mean of the normal distribution.
        :param std: Standard deviation of the normal distribution.
        :param device: Data allocation and computation device. If not specified, the default device will be used.

        Example::

            >>> noise = GaussianNoise(mean=0, std=1)
        """
        super().__init__(device=device)

        if self._jax:
            self._i = 0
            self._key = config.jax.key
            self.mean = jnp.array(mean)
            self.std = jnp.array(std)
        else:
            self.mean = np.array(mean)
            self.std = np.array(std)

    def sample(self, size: Tuple[int]) -> Union[np.ndarray, jax.Array]:
        """Sample a Gaussian noise.

        :param size: Noise shape.

        :return: Sampled noise.

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
