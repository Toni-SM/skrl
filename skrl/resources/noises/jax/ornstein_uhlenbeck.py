from typing import Optional, Union, Tuple

import numpy as np

import jax
import jaxlib
import jax.numpy as jnp

from skrl import jax_backend
from skrl.resources.noises.jax import Noise


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

        self.state = 0
        self.theta = theta
        self.sigma = sigma
        self.base_scale = base_scale

        # normal distribution
        if jax_backend == "jax":
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

        elif jax_backend == "numpy":
            class _Normal:
                def __init__(self, loc, scale):
                    self._loc = loc
                    self._scale = scale

                def sample(self, size):
                    return np.random.normal(self._loc, self._scale, size)

        self.distribution = _Normal(loc=mean, scale=std)

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
        self.state += -self.state * self.theta + self.sigma * self.distribution.sample(size)

        return self.base_scale * self.state
