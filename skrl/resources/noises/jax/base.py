from typing import Optional, Tuple, Union

import jax
import numpy as np

from skrl import config


class Noise():
    def __init__(self, device: Optional[Union[str, jax.Device]] = None) -> None:
        """Base class representing a noise

        :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda"`` if available or ``"cpu"``
        :type device: str or jax.Device, optional

        Custom noises should override the ``sample`` method::

            import jax
            from skrl.resources.noises.jax import Noise

            class CustomNoise(Noise):
                def __init__(self, device=None):
                    super().__init__(device)

                def sample(self, size):
                    return jax.random.uniform(jax.random.PRNGKey(0), size)
        """
        self._jax = config.jax.backend == "jax"

        if device is None:
            self.device = jax.devices()[0]
        else:
            self.device = device if isinstance(device, jax.Device) else jax.devices(device)[0]

    def sample_like(self, tensor: Union[np.ndarray, jax.Array]) -> Union[np.ndarray, jax.Array]:
        """Sample a noise with the same size (shape) as the input tensor

        This method will call the sampling method as follows ``.sample(tensor.shape)``

        :param tensor: Input tensor used to determine output tensor size (shape)
        :type tensor: np.ndarray or jax.Array

        :return: Sampled noise
        :rtype: np.ndarray or jax.Array

        Example::

            >>> x = jax.random.uniform(jax.random.PRNGKey(0), (3, 2))
            >>> noise.sample_like(x)
            Array([[0.57450044, 0.09968603],
                   [0.7419659 , 0.8941783 ],
                   [0.59656656, 0.45325184]], dtype=float32)
        """
        return self.sample(tensor.shape)

    def sample(self, size: Tuple[int]) -> Union[np.ndarray, jax.Array]:
        """Noise sampling method to be implemented by the inheriting classes

        :param size: Shape of the sampled tensor
        :type size: tuple or list of int

        :raises NotImplementedError: The method is not implemented by the inheriting classes

        :return: Sampled noise
        :rtype: np.ndarray or jax.Array
        """
        raise NotImplementedError("The sampling method (.sample()) is not implemented")
