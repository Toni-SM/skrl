from typing import Union, Tuple

import numpy as np
from functools import partial

import jax
import jax.numpy as jnp

from skrl import config


# https://jax.readthedocs.io/en/latest/faq.html#strategy-1-jit-compiled-helper-function
@partial(jax.jit, static_argnames=("shape"))
def _sample(loc, scale, key, data, shape):
    subkey = jax.random.fold_in(key, data)
    return jax.random.normal(subkey, shape) * scale + loc

@jax.jit
def _log_prob(loc, scale, value):
    return -jnp.square(value - loc) / (2 * jnp.square(scale)) - jnp.log(scale) - 0.5 * jnp.log(2 * jnp.pi)

@jax.jit
def _entropy(scale):
    return 0.5 + 0.5 * jnp.log(2 * jnp.pi) + jnp.log(scale)


class Normal:
    def __init__(self, loc: Union[float, np.ndarray, jnp.ndarray], scale: Union[float, np.ndarray, jnp.ndarray]) -> None:
        """Normal (also called Gaussian) distribution parameterized by ``loc`` and ``scale``

        :param loc: Mean of the normal distribution
        :type loc: float, np.ndarray or jnp.ndarray
        :param scale: Standard deviation of the normal distribution
        :type scale: float, np.ndarray or jnp.ndarray
        """
        self.loc = jnp.array(loc)
        self.scale = jnp.array(scale)

        self._i = 0
        self._key = jax.random.PRNGKey(0)

        self._jax = config.jax.backend == "jax"

    @property
    def mean(self) -> Union[np.ndarray, jnp.ndarray]:
        """Mean of the distribution
        """
        return self.loc

    @property
    def stddev(self) -> Union[np.ndarray, jnp.ndarray]:
        """Standard deviation of the distribution
        """
        return self.scale

    @property
    def variance(self) -> Union[np.ndarray, jnp.ndarray]:
        """Variance of the distribution
        """
        return jnp.square(self.scale)

    def sample(self, sample_shape: Tuple[int]) -> Union[np.ndarray, jnp.ndarray]:
        """
        Generates samples from the distribution

        :return: Samples with shape ``sample_shape``
        :rtype: np.ndarray or jnp.ndarray
        """
        if self._jax:
            self._i += 1
            return _sample(self.loc, self.scale, self._key, self._i, sample_shape)
        return np.random.normal(self.loc, self.scale, sample_shape)

    def log_prob(self, value: Union[np.ndarray, jnp.ndarray]) -> Union[np.ndarray, jnp.ndarray]:
        """Log of the probability density/mass function evaluated at value

        :param value: Value to be evaluated
        :type value: np.ndarray or jnp.ndarray

        :return: Log of the probability density/mass function
        :rtype: np.ndarray or jnp.ndarray
        """
        if self._jax:
            return _log_prob(self.loc, self.scale, value)
        return -np.square(value - self.loc) / (2 * np.square(self.scale)) - np.log(self.scale) - 0.5 * np.log(2 * np.pi)

    def entropy(self) -> Union[np.ndarray, jnp.ndarray]:
        """Entropy of distribution

        :return: Entropy
        :rtype: np.ndarray or jnp.ndarray
        """
        if self._jax:
            return _entropy(self.scale)
        return 0.5 + 0.5 * np.log(2 * np.pi) + np.log(self.scale)
