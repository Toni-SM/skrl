from torch.distributions import Normal

from . import Noise


class OrnsteinUhlenbeckNoise(Noise):
    def __init__(self, theta: float, sigma: float, base_scale: float, device) -> None:
        """
        Ornstein Uhlenbeck noise

        Parameters
        ----------
        theta: float
            Factor to apply to internal state
        sigma
            Factor to apply to the normal distribution
        base_scale
            Factor to apply to returned noise
        device: str
            Device on which a PyTorch tensor is or will be allocated
        """
        super().__init__(device)

        self.theta = theta
        self.sigma = sigma
        self.base_scale = base_scale

        self.state = 0

        self.distribution = Normal(0.0, 1.0)
        
    def sample(self, shape):
        gaussian_sample = self.distribution.sample(shape).to(self.device)
        self.state += -self.state * self.theta + self.sigma * gaussian_sample
        
        return self.base_scale * self.state
