from typing import Optional, Tuple, Union

from abc import ABC, abstractmethod

import torch

from skrl import config


class Noise(ABC):
    def __init__(self, *, device: Optional[Union[str, torch.device]] = None) -> None:
        """Base noise class for implementing custom noises.

        :param device: Data allocation and computation device. If not specified, the default device will be used.
        """
        self.device = config.torch.parse_device(device)

    def sample_like(self, tensor: torch.Tensor) -> torch.Tensor:
        """Sample noise with the same size (shape) as the input tensor.

        This method will call the sampling method as follows ``.sample(tensor.shape)``.

        :param tensor: Input tensor used to determine output tensor size (shape).

        :return: Sampled noise.

        Example::

            >>> x = torch.rand(3, 2, device="cuda:0")
            >>> noise.sample_like(x)
            tensor([[-0.0423, -0.1325],
                    [-0.0639, -0.0957],
                    [-0.1367,  0.1031]], device='cuda:0')
        """
        return self.sample(tensor.shape)

    @abstractmethod
    def sample(self, size: Union[Tuple[int], torch.Size]) -> torch.Tensor:
        """Sample noise.

        :param size: Noise shape.

        :return: Sampled noise.

        :raises NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("The sampling method (.sample()) is not implemented")
