from torch.distributions import Normal

from . import Noise


class GaussianNoise(Noise):
    def __init__(self, mean: float, std: float, device) -> None:
        """
        Gaussian noise

        Parameters
        ----------
        mean: float
            Mean of each output element's normal distribution
        std
            Standard deviation of each output element's normal distribution
        """
        super().__init__(device)

        self.distribution = Normal(mean, std)
        
    def sample(self, shape):
        return self.distribution.sample(shape).to(self.device)
