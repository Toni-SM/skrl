from typing import Union, Tuple

import torch
from torch.distributions import Normal

from . import Noise


class GaussianNoise(Noise):
    def __init__(self, mean: float, std: float, device: str = "cuda:0") -> None:
        """
        Class representing a Gaussian noise

        Parameters
        ----------
        mean: float
            Mean of the normal distribution
        std
            Standard deviation of the normal distribution
        device: str, optional
            Device on which a torch tensor is or will be allocated (default: "cuda:0")
        """
        super().__init__(device)

        self.distribution = Normal(mean, std)
        
    def sample(self, size: Union[Tuple[int], torch.Size]) -> torch.Tensor:
        """
        Sample a Gaussian noise

        Parameters
        ----------
        size: tuple or list of integers or torch.Size
            Shape of the sampled tensor

        Returns
        -------
        torch.Tensor
            Sampled noise
        """
        return self.distribution.sample(size).to(self.device)
