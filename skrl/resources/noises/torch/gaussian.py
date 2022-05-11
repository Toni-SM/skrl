from typing import Union, Tuple

import torch
from torch.distributions import Normal

from . import Noise


class GaussianNoise(Noise):
    def __init__(self, mean: float, std: float, device: Union[str, torch.device] = "cuda:0") -> None:
        """Class representing a Gaussian noise

        :param mean: Mean of the normal distribution
        :type mean: float
        :param std: Standard deviation of the normal distribution
        :type std: float
        :param device: Device on which a torch tensor is or will be allocated (default: "cuda:0")
        :type device: str or torch.device, optional
        """
        super().__init__(device)

        self.distribution = Normal(loc=torch.tensor(mean, device=self.device, dtype=torch.float32),
                                   scale=torch.tensor(std, device=self.device, dtype=torch.float32))
        
    def sample(self, size: Union[Tuple[int], torch.Size]) -> torch.Tensor:
        """Sample a Gaussian noise

        :param size: Shape of the sampled tensor
        :type size: tuple or list of integers, or torch.Size
        
        :return: Sampled noise
        :rtype: torch.Tensor
        """
        return self.distribution.sample(size)
