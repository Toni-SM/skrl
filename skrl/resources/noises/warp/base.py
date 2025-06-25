from typing import Optional, Tuple, Union

import warp as wp

from skrl import config


class Noise:
    def __init__(self, device: Optional[Union[str, wp.context.Device]] = None) -> None:
        """Base class representing a noise

        :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda"`` if available or ``"cpu"``
        :type device: str or context.Device, optional

        Custom noises should override the ``sample`` method::

            import jax
            from skrl.resources.noises.warp import Noise

            class CustomNoise(Noise):
                def __init__(self, device=None):
                    super().__init__(device)

                def sample(self, size):
                    return jax.random.uniform(jax.random.PRNGKey(0), size)
        """
        self.device = config.warp.parse_device(device)

    def sample_like(self, tensor: wp.array) -> wp.array:
        """Sample a noise with the same size (shape) as the input tensor

        This method will call the sampling method as follows ``.sample(tensor.shape)``

        :param tensor: Input tensor used to determine output tensor size (shape)
        :type tensor: wp.array

        :return: Sampled noise
        :rtype: wp.array

        Example::

            >>> x = jax.random.uniform(jax.random.PRNGKey(0), (3, 2))
            >>> noise.sample_like(x)
            Array([[0.57450044, 0.09968603],
                   [0.7419659 , 0.8941783 ],
                   [0.59656656, 0.45325184]], dtype=float32)
        """
        return self.sample(tensor.shape)

    def sample(self, size: Tuple[int]) -> wp.array:
        """Noise sampling method to be implemented by the inheriting classes

        :param size: Shape of the sampled tensor
        :type size: tuple or list of int

        :raises NotImplementedError: The method is not implemented by the inheriting classes

        :return: Sampled noise
        :rtype: wp.array
        """
        raise NotImplementedError("The sampling method (.sample()) is not implemented")
