from typing import Optional, Tuple, Union

import torch


class Noise():
    def __init__(self, device: Optional[Union[str, torch.device]] = None) -> None:
        """Base class representing a noise

        :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda"`` if available or ``"cpu"``
        :type device: str or torch.device, optional

        Custom noises should override the ``sample`` method::

            import torch
            from skrl.resources.noises.torch import Noise

            class CustomNoise(Noise):
                def __init__(self, device=None):
                    super().__init__(device)

                def sample(self, size):
                    return torch.rand(size, device=self.device)
        """
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

    def sample_like(self, tensor: torch.Tensor) -> torch.Tensor:
        """Sample a noise with the same size (shape) as the input tensor

        This method will call the sampling method as follows ``.sample(tensor.shape)``

        :param tensor: Input tensor used to determine output tensor size (shape)
        :type tensor: torch.Tensor

        :return: Sampled noise
        :rtype: torch.Tensor

        Example::

            >>> x = torch.rand(3, 2, device="cuda:0")
            >>> noise.sample_like(x)
            tensor([[-0.0423, -0.1325],
                    [-0.0639, -0.0957],
                    [-0.1367,  0.1031]], device='cuda:0')
        """
        return self.sample(tensor.shape)

    def sample(self, size: Union[Tuple[int], torch.Size]) -> torch.Tensor:
        """Noise sampling method to be implemented by the inheriting classes

        :param size: Shape of the sampled tensor
        :type size: tuple or list of int, or torch.Size

        :raises NotImplementedError: The method is not implemented by the inheriting classes

        :return: Sampled noise
        :rtype: torch.Tensor
        """
        raise NotImplementedError("The sampling method (.sample()) is not implemented")
