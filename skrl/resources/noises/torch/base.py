from typing import Optional, Union, Tuple

import torch


class Noise():
    def __init__(self, device: Optional[Union[str, torch.device]] = None) -> None:
        """Base class representing a noise

        :param device: Device on which a torch tensor is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda:0"`` if available or ``"cpu"
        :type device: str or torch.device, optional
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)

    def sample_like(self, tensor: torch.Tensor) -> torch.Tensor:
        """Sample a noise with the same size (shape) as the input tensor

        This method will call the sampling method as follows ``.sample(tensor.size())``

        :param tensor: Input tensor used to determine output tensor size (shape)
        :type tensor: torch.Tensor

        :return: Sampled noise
        :rtype: torch.Tensor
        """
        return self.sample(tensor.size())

    def sample(self, size: Union[Tuple[int], torch.Size]) -> torch.Tensor:
        """Noise sampling method to be implemented by the inheriting classes

        :param size: Shape of the sampled tensor
        :type size: tuple or list of integers, or torch.Size

        :raises NotImplementedError: The method is not implemented by the inheriting classes

        :return: Sampled noise
        :rtype: torch.Tensor
        """
        raise NotImplementedError("The sampling method (.sample()) is not implemented")
