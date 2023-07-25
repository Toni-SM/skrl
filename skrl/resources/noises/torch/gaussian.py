from typing import Optional, Tuple, Union

import torch
from torch.distributions import Normal

from skrl.resources.noises.torch import Noise


class GaussianNoise(Noise):
    def __init__(self, mean: float, std: float, device: Optional[Union[str, torch.device]] = None) -> None:
        """Class representing a Gaussian noise

        :param mean: Mean of the normal distribution
        :type mean: float
        :param std: Standard deviation of the normal distribution
        :type std: float
        :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda"`` if available or ``"cpu"``
        :type device: str or torch.device, optional

        Example::

            >>> noise = GaussianNoise(mean=0, std=1)
        """
        super().__init__(device)

        self.distribution = Normal(loc=torch.tensor(mean, device=self.device, dtype=torch.float32),
                                   scale=torch.tensor(std, device=self.device, dtype=torch.float32))

    def sample(self, size: Union[Tuple[int], torch.Size]) -> torch.Tensor:
        """Sample a Gaussian noise

        :param size: Shape of the sampled tensor
        :type size: tuple or list of int, or torch.Size

        :return: Sampled noise
        :rtype: torch.Tensor

        Example::

            >>> noise.sample((3, 2))
            tensor([[-0.4901,  1.3357],
                    [-1.2141,  0.3323],
                    [-0.0889, -1.1651]], device='cuda:0')

            >>> x = torch.rand(3, 2, device="cuda:0")
            >>> noise.sample(x.shape)
            tensor([[0.5398, 1.2009],
                    [0.0307, 1.3065],
                    [0.2082, 0.6116]], device='cuda:0')
        """
        return self.distribution.sample(size)
