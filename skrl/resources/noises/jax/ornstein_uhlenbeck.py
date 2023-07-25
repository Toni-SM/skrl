from typing import Optional, Tuple, Union

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from skrl import config
from skrl.resources.noises.jax import Noise


# https://jax.readthedocs.io/en/latest/faq.html#strategy-1-jit-compiled-helper-function
@partial(jax.jit, static_argnames=("shape"))
def _sample(theta, sigma, state, mean, std, key, iterator, shape):
    subkey = jax.random.fold_in(key, iterator)
    return state * theta + sigma * (jax.random.normal(subkey, shape) * std + mean)


class OrnsteinUhlenbeckNoise(Noise):
    def __init__(self,
                 theta: float,
                 sigma: float,
                 base_scale: float,
                 mean: float = 0,
                 std: float = 1,
                 device: Optional[Union[str, jax.Device]] = None) -> None:
        """Class representing an Ornstein-Uhlenbeck noise

        :param theta: Factor to apply to current internal state
        :type theta: float
        :param sigma: Factor to apply to the normal distribution
        :type sigma: float
        :param base_scale: Factor to apply to returned noise
        :type base_scale: float
        :param mean: Mean of the normal distribution (default: ``0.0``)
        :type mean: float, optional
        :param std: Standard deviation of the normal distribution (default: ``1.0``)
        :type std: float, optional
        :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda"`` if available or ``"cpu"``
        :type device: str or jax.Device, optional

        Example::

            >>> noise = OrnsteinUhlenbeckNoise(theta=0.1, sigma=0.2, base_scale=0.5)
        """
        super().__init__(device)

        self.state = 0
        self.theta = theta
        self.sigma = sigma
        self.base_scale = base_scale

        if self._jax:
            self.mean = jnp.array(mean)
            self.std = jnp.array(std)

            self._i = 0
            self._key = config.jax.key
        else:
            self.mean = np.array(mean)
            self.std = np.array(std)

    def sample(self, size: Tuple[int]) -> Union[np.ndarray, jax.Array]:
        """Sample an Ornstein-Uhlenbeck noise

        :param size: Shape of the sampled tensor
        :type size: tuple or list of int

        :return: Sampled noise
        :rtype: np.ndarray or jax.Array

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
        if hasattr(self.state, "shape") and self.state.shape != size:
            self.state = 0
        if self._jax:
            self._i += 1
            self.state = _sample(self.theta, self.sigma, self.state, self.mean, self.std, self._key, self._i, size)
        else:
            self.state += -self.state * self.theta + self.sigma * np.random.normal(self.mean, self.std, size)
        return self.base_scale * self.state
