from typing import Optional, Tuple, Union

from abc import ABC, abstractmethod

import jax
import numpy as np

from skrl import config


class Noise(ABC):
    def __init__(self, *, device: Optional[Union[str, jax.Device]] = None) -> None:
        """Base noise class for implementing custom noises.

        :param device: Data allocation and computation device. If not specified, the default device will be used.
        """
        self._jax = config.jax.backend == "jax"

        self.device = config.jax.parse_device(device)

    def sample_like(self, tensor: Union[np.ndarray, jax.Array]) -> Union[np.ndarray, jax.Array]:
        """Sample noise with the same size (shape) as the input tensor.

        This method will call the sampling method as follows ``.sample(tensor.shape)``.

        :param tensor: Input tensor used to determine output tensor size (shape).

        :return: Sampled noise.

        Example::

            >>> x = jax.random.uniform(jax.random.PRNGKey(0), (3, 2))
            >>> noise.sample_like(x)
            Array([[0.57450044, 0.09968603],
                   [0.7419659 , 0.8941783 ],
                   [0.59656656, 0.45325184]], dtype=float32)
        """
        return self.sample(tensor.shape)

    @abstractmethod
    def sample(self, size: Tuple[int]) -> Union[np.ndarray, jax.Array]:
        """Sample noise.

        :param size: Noise shape.

        :return: Sampled noise.

        :raises NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("The sampling method (.sample()) is not implemented")
