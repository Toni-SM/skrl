from typing import Union, Tuple

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

    def sample(self, shape: Union[Tuple[int], torch.Size]) -> torch.Tensor:
        """
        Sample noise

        Parameters
        ----------
        shape: torch.Size or tuple of ints
            Number of element on each dimension

        Returns
        -------
        tuple
            Sampled tensor
        """
        raise NotImplementedError("The sampling method (.sample()) is not implemented")