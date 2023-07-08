# [start-base-class-torch]
from typing import Union, Tuple

import torch

from skrl.resources.noises.torch import Noise


class CustomNoise(Noise):
    def __init__(self, device: Union[str, torch.device] = "cuda:0") -> None:
        """
        :param device: Device on which a torch tensor is or will be allocated (default: "cuda:0")
        :type device: str or torch.device, optional
        """
        super().__init__(device)

    def sample(self, size: Union[Tuple[int], torch.Size]) -> torch.Tensor:
        """Sample noise

        :param size: Shape of the sampled tensor
        :type size: tuple or list of integers, or torch.Size

        :return: Sampled noise
        :rtype: torch.Tensor
        """
        # ================================
        # - sample noise
        # ================================
# [end-base-class-torch]


# [start-base-class-jax]
from typing import Optional, Union, Tuple

import numpy as np

import jaxlib
import jax.numpy as jnp

from skrl.resources.noises.torch import Noise


class CustomNoise(Noise):
    def __init__(self, device: Optional[Union[str, jaxlib.xla_extension.Device]] = None) -> None:
        """Custom noise

        :param device: Device on which a jax array is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda:0"`` if available or ``"cpu"``
        :type device: str or jaxlib.xla_extension.Device, optional
        """
        super().__init__(device)

    def sample(self, size: Tuple[int]) -> Union[np.ndarray, jnp.ndarray]:
        """Sample noise

        :param size: Shape of the sampled tensor
        :type size: tuple or list of integers

        :return: Sampled noise
        :rtype: np.ndarray or jnp.ndarray
        """
        # ================================
        # - sample noise
        # ================================
# [end-base-class-jax]

# =============================================================================

# [torch-start-gaussian]
from skrl.resources.noises.torch import GaussianNoise

cfg = DEFAULT_CONFIG.copy()
cfg["exploration"]["noise"] = GaussianNoise(mean=0, std=0.2, device="cuda:0")
# [torch-end-gaussian]


# [jax-start-gaussian]
from skrl.resources.noises.jax import GaussianNoise

cfg = DEFAULT_CONFIG.copy()
cfg["exploration"]["noise"] = GaussianNoise(mean=0, std=0.2)
# [jax-end-gaussian]

# =============================================================================

# [torch-start-ornstein-uhlenbeck]
from skrl.resources.noises.torch import OrnsteinUhlenbeckNoise

cfg = DEFAULT_CONFIG.copy()
cfg["exploration"]["noise"] = OrnsteinUhlenbeckNoise(theta=0.15, sigma=0.2, base_scale=1.0, device="cuda:0")
# [torch-end-ornstein-uhlenbeck]


# [jax-start-ornstein-uhlenbeck]
from skrl.resources.noises.jax import OrnsteinUhlenbeckNoise

cfg = DEFAULT_CONFIG.copy()
cfg["exploration"]["noise"] = OrnsteinUhlenbeckNoise(theta=0.15, sigma=0.2, base_scale=1.0)
# [jax-end-ornstein-uhlenbeck]
