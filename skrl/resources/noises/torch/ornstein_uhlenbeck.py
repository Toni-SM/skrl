from typing import Optional, Tuple, Union

import torch
from torch.distributions import Normal

from skrl.resources.noises.torch import Noise


# speed up distribution construction by disabling checking
Normal.set_default_validate_args(False)


class OrnsteinUhlenbeckNoise(Noise):
    def __init__(
        self,
        *,
        theta: float,
        sigma: float,
        base_scale: float,
        mean: float = 0,
        std: float = 1,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """Ornstein-Uhlenbeck noise.

        :param theta: Factor to apply to current internal state.
        :param sigma: Factor to apply to the normal distribution.
        :param base_scale: Factor to apply to returned noise.
        :param mean: Mean of the normal distribution.
        :param std: Standard deviation of the normal distribution.
        :param device: Data allocation and computation device. If not specified, the default device will be used.

        Example::

            >>> noise = OrnsteinUhlenbeckNoise(theta=0.1, sigma=0.2, base_scale=0.5)
        """
        super().__init__(device=device)

        self.state = 0
        self.theta = theta
        self.sigma = sigma
        self.base_scale = base_scale

        self.distribution = Normal(
            loc=torch.tensor(mean, device=self.device, dtype=torch.float32),
            scale=torch.tensor(std, device=self.device, dtype=torch.float32),
        )

    def sample(self, size: Union[Tuple[int], torch.Size]) -> torch.Tensor:
        """Sample an Ornstein-Uhlenbeck noise.

        :param size: Noise shape.

        :return: Sampled noise.

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
