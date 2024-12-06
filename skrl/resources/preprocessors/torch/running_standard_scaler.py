from typing import Optional, Tuple, Union

import gymnasium

import torch
import torch.nn as nn

from skrl import config
from skrl.utils.spaces.torch import compute_space_size


class RunningStandardScaler(nn.Module):
    def __init__(
        self,
        size: Union[int, Tuple[int], gymnasium.Space],
        epsilon: float = 1e-8,
        clip_threshold: float = 5.0,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """Standardize the input data by removing the mean and scaling by the standard deviation

        The implementation is adapted from the rl_games library
        (https://github.com/Denys88/rl_games/blob/master/rl_games/algos_torch/running_mean_std.py)

        Example::

            >>> running_standard_scaler = RunningStandardScaler(size=2)
            >>> data = torch.rand(3, 2)  # tensor of shape (N, 2)
            >>> running_standard_scaler(data)
            tensor([[0.1954, 0.3356],
                    [0.9719, 0.4163],
                    [0.8540, 0.1982]])

        :param size: Size of the input space
        :type size: int, tuple or list of integers, or gymnasium.Space
        :param epsilon: Small number to avoid division by zero (default: ``1e-8``)
        :type epsilon: float
        :param clip_threshold: Threshold to clip the data (default: ``5.0``)
        :type clip_threshold: float
        :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda"`` if available or ``"cpu"``
        :type device: str or torch.device, optional
        """
        super().__init__()

        self.epsilon = epsilon
        self.clip_threshold = clip_threshold

        self.device = config.torch.parse_device(device)

        size = compute_space_size(size, occupied_size=True)

        self.register_buffer("running_mean", torch.zeros(size, dtype=torch.float64, device=self.device))
        self.register_buffer("running_variance", torch.ones(size, dtype=torch.float64, device=self.device))
        self.register_buffer("current_count", torch.ones((), dtype=torch.float64, device=self.device))

    def _parallel_variance(self, input_mean: torch.Tensor, input_var: torch.Tensor, input_count: int) -> None:
        """Update internal variables using the parallel algorithm for computing variance

        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param input_mean: Mean of the input data
        :type input_mean: torch.Tensor
        :param input_var: Variance of the input data
        :type input_var: torch.Tensor
        :param input_count: Batch size of the input data
        :type input_count: int
        """
        delta = input_mean - self.running_mean
        total_count = self.current_count + input_count
        M2 = (
            (self.running_variance * self.current_count)
            + (input_var * input_count)
            + delta**2 * self.current_count * input_count / total_count
        )

        # update internal variables
        self.running_mean = self.running_mean + delta * input_count / total_count
        self.running_variance = M2 / total_count
        self.current_count = total_count

    def _compute(self, x: torch.Tensor, train: bool = False, inverse: bool = False) -> torch.Tensor:
        """Compute the standardization of the input data

        :param x: Input tensor
        :type x: torch.Tensor
        :param train: Whether to train the standardizer (default: ``False``)
        :type train: bool, optional
        :param inverse: Whether to inverse the standardizer to scale back the data (default: ``False``)
        :type inverse: bool, optional

        :return: Standardized tensor
        :rtype: torch.Tensor
        """
        if train:
            if x.dim() == 3:
                self._parallel_variance(torch.mean(x, dim=(0, 1)), torch.var(x, dim=(0, 1)), x.shape[0] * x.shape[1])
            else:
                self._parallel_variance(torch.mean(x, dim=0), torch.var(x, dim=0), x.shape[0])

        # scale back the data to the original representation
        if inverse:
            return (
                torch.sqrt(self.running_variance.float())
                * torch.clamp(x, min=-self.clip_threshold, max=self.clip_threshold)
                + self.running_mean.float()
            )
        # standardization by centering and scaling
        return torch.clamp(
            (x - self.running_mean.float()) / (torch.sqrt(self.running_variance.float()) + self.epsilon),
            min=-self.clip_threshold,
            max=self.clip_threshold,
        )

    def forward(
        self, x: torch.Tensor, train: bool = False, inverse: bool = False, no_grad: bool = True
    ) -> torch.Tensor:
        """Forward pass of the standardizer

        Example::

            >>> x = torch.rand(3, 2, device="cuda:0")
            >>> running_standard_scaler(x)
            tensor([[0.6933, 0.1905],
                    [0.3806, 0.3162],
                    [0.1140, 0.0272]], device='cuda:0')

            >>> running_standard_scaler(x, train=True)
            tensor([[ 0.8681, -0.6731],
                    [ 0.0560, -0.3684],
                    [-0.6360, -1.0690]], device='cuda:0')

            >>> running_standard_scaler(x, inverse=True)
            tensor([[0.6260, 0.5468],
                    [0.5056, 0.5987],
                    [0.4029, 0.4795]], device='cuda:0')

        :param x: Input tensor
        :type x: torch.Tensor
        :param train: Whether to train the standardizer (default: ``False``)
        :type train: bool, optional
        :param inverse: Whether to inverse the standardizer to scale back the data (default: ``False``)
        :type inverse: bool, optional
        :param no_grad: Whether to disable the gradient computation (default: ``True``)
        :type no_grad: bool, optional

        :return: Standardized tensor
        :rtype: torch.Tensor
        """
        if no_grad:
            with torch.no_grad():
                return self._compute(x, train, inverse)
        return self._compute(x, train, inverse)
