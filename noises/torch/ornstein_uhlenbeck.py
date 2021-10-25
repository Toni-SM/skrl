from typing import Union, Tuple

import torch
from torch.distributions import Normal

from . import Noise


class OrnsteinUhlenbeckNoise(Noise):
    def __init__(self, theta: float, sigma: float, base_scale: float, mean: float = 0, std: float = 1, device: str = "cuda:0") -> None:
        """
        Class representing an Ornstein Uhlenbeck noise

        Parameters
        ----------
        theta: float
            Factor to apply to current internal state
        sigma: float
            Factor to apply to the normal distribution
        base_scale: float
            Factor to apply to returned noise
        mean: float, optional
            Mean of the normal distribution (default: 0.0)
        std: float, optional
            Standard deviation of the normal distribution (default: 1.0)
        device: str, optional
            Device on which a torch tensor is or will be allocated (default: "cuda:0")
        """
        super().__init__(device)

        self.theta = theta
        self.sigma = sigma
        self.base_scale = base_scale

        self.state = 0

        self.distribution = Normal(mean, std)
        
    def sample(self, size: Union[Tuple[int], torch.Size]) -> torch.Tensor:
        """
        Sample an Ornstein Uhlenbeck noise

        Parameters
        ----------
        size: tuple or list of ints or torch.Size
            Shape of the sampled tensor

        Returns
        -------
        torch.Tensor
            Sampled noise
        """
        gaussian_noise = self.distribution.sample(size).to(self.device)
        self.state += -self.state * self.theta + self.sigma * gaussian_noise
        
        return self.base_scale * self.state
