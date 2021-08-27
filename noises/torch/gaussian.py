from torch.distributions import Normal

from . import Noise


class GaussianNoise(Noise):
    def __init__(self, mean: float, std: float, device) -> None:
        """
        Gaussian noise

        Parameters
        ----------
        mean: float
            Mean of the normal distribution
        std
            Standard deviation of the normal distribution
        device: str
            Device on which a PyTorch tensor is or will be allocated
        """
        super().__init__(device)

        self.distribution = Normal(mean, std)
        
    def sample(self, shape):
        return self.distribution.sample(shape).to(self.device)
