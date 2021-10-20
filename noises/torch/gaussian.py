from typing import Union

import torch
from torch.distributions import Normal

from . import Noise


class GaussianNoise(Noise):
    def __init__(self, mean: float, std: float, device: str = "cuda:0") -> None:
        """
        Gaussian noise

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
        
    def sample(self, shape: Union[tuple[int], list[int], torch.Size]) -> torch.Tensor:
        return self.distribution.sample(shape).to(self.device)
