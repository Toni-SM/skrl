from typing import Union, Tuple

import torch

from skrl.resources.noises.torch import Noise     # from . import Noise


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
