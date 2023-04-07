from typing import Optional, Union, Tuple

import numpy as np

import jax
import jaxlib
import jax.numpy as jnp

from skrl import config
from skrl.resources.noises.jax import Noise
from skrl.resources.distributions.jax import Normal


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
        self._jax = config.jax.backend == "jax"

        self.distribution = Normal(loc=mean, scale=std)

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
