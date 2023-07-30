from typing import Optional, Tuple, Union

import torch
from torch.distributions import Normal

from skrl.resources.noises.torch import Noise


class OrnsteinUhlenbeckNoise(Noise):
    def __init__(self,
                 theta: float,
                 sigma: float,
                 base_scale: float,
                 mean: float = 0,
                 std: float = 1,
                 device: Optional[Union[str, torch.device]] = None) -> None:
        """Class representing an Ornstein-Uhlenbeck noise

        :param theta: Factor to apply to current internal state
        :type theta: float
        :param sigma: Factor to apply to the normal distribution
        :type sigma: float
        :param base_scale: Factor to apply to returned noise
        :type base_scale: float
        :param mean: Mean of the normal distribution (default: ``0.0``)
        :type mean: float, optional
        :param std: Standard deviation of the normal distribution (default: ``1.0``)
        :type std: float, optional
        :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda"`` if available or ``"cpu"``
        :type device: str or torch.device, optional

        Example::

            >>> noise = OrnsteinUhlenbeckNoise(theta=0.1, sigma=0.2, base_scale=0.5)
        """
        super().__init__(device)

        self.state = 0
        self.theta = theta
        self.sigma = sigma
        self.base_scale = base_scale

        self.distribution = Normal(loc=torch.tensor(mean, device=self.device, dtype=torch.float32),
                                   scale=torch.tensor(std, device=self.device, dtype=torch.float32))

    def sample(self, size: Union[Tuple[int], torch.Size]) -> torch.Tensor:
        """Sample an Ornstein-Uhlenbeck noise

        :param size: Shape of the sampled tensor
        :type size: tuple or list of int, or torch.Size

        :return: Sampled noise
        :rtype: torch.Tensor

        Example::

            >>> noise.sample((3, 2))
            tensor([[-0.0452,  0.0162],
                    [ 0.0649, -0.0708],
                    [-0.0211,  0.0066]], device='cuda:0')

            >>> x = torch.rand(3, 2, device="cuda:0")
            >>> noise.sample(x.shape)
            tensor([[-0.0540,  0.0461],
                    [ 0.1117, -0.1157],
                    [-0.0074,  0.0420]], device='cuda:0')
        """
        if hasattr(self.state, "shape") and self.state.shape != torch.Size(size):
            self.state = 0
        self.state += -self.state * self.theta + self.sigma * self.distribution.sample(size)

        return self.base_scale * self.state
