from typing import Optional, Union, Tuple

import numpy as np

import jax
import jaxlib
import jax.numpy as jnp

from skrl import config
from skrl.resources.noises.jax import Noise
from skrl.resources.distributions.jax import Normal


# https://jax.readthedocs.io/en/latest/faq.html#strategy-1-jit-compiled-helper-function
@jax.jit
def _sample(theta, sigma, state, samples):
    return state * theta + sigma * samples


class OrnsteinUhlenbeckNoise(Noise):
    def __init__(self,
                 theta: float,
                 sigma: float,
                 base_scale: float,
                 mean: float = 0,
                 std: float = 1,
                 device: Optional[Union[str, jaxlib.xla_extension.Device]] = None) -> None:
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
        :param device: Device on which a jax array is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda:0"`` if available or ``"cpu"``
        :type device: str or jaxlib.xla_extension.Device, optional

        Example::

            >>> noise = OrnsteinUhlenbeckNoise(theta=0.1, sigma=0.2, base_scale=0.5)
        """
        super().__init__(device)
        self._jax = config.jax.backend == "jax"

        self.state = 0
        self.theta = theta
        self.sigma = sigma
        self.base_scale = base_scale

        self.distribution = Normal(loc=mean, scale=std)

    def sample(self, size: Tuple[int]) -> Union[np.ndarray, jnp.ndarray]:
        """Sample an Ornstein-Uhlenbeck noise

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
        if hasattr(self.state, "shape") and self.state.shape != size:
            self.state = 0
        if self._jax:
            self.state = _sample(self.theta, self.sigma, self.state, self.distribution.sample(size))
        else:
            self.state += -self.state * self.theta + self.sigma * self.distribution.sample(size)
        return self.base_scale * self.state
