from typing import Union, Tuple

import numpy as np
import torch


class Noise():
    def __init__(self, device: str) -> None:
        """
        Base class that represent a noise

        Parameters
        ----------
        device: str
            Device on which a PyTorch tensor is or will be allocated
        """
        self.device = device

    def sample(self, shape: Union[Tuple[int], torch.Size]) -> Tuple[torch.Tensor]:
        """
        Sample a batch from the memory

        Parameters
        ----------
        shape: torch.Size or tuple of ints
            Number of element on each dimension

        Returns
        -------
        tuple
            Sampled tensors
        """
        raise NotImplementedError("The sampling method (.sample()) is not implemented")