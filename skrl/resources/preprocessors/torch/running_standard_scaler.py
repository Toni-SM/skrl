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
        *,
        epsilon: float = 1e-8,
        clip_threshold: float = 5.0,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """Standardize the input data by removing the mean and scaling by the standard deviation.

        :param size: Size of the input space.
        :param epsilon: Small number to avoid division by zero.
        :param clip_threshold: Threshold to clip the data.
        :param device: Data allocation and computation device. If not specified, the default device will be used.

        Example::

            >>> running_standard_scaler = RunningStandardScaler(size=2)
            >>> data = torch.rand(3, 2)  # tensor of shape (N, 2)
            >>> running_standard_scaler(data)
            tensor([[0.1954, 0.3356],
                    [0.9719, 0.4163],
                    [0.8540, 0.1982]])
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
        """Update internal variables using the parallel algorithm for computing variance.

        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param input_mean: Mean of the input data.
        :param input_var: Variance of the input data.
        :param input_count: Batch size of the input data.
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

    def _compute(self, x: torch.Tensor, *, train: bool = False, inverse: bool = False) -> torch.Tensor:
        """Compute the standardization of the input data.

        :param x: Input tensor.
        :param train: Whether to train the standardizer.
        :param inverse: Whether to inverse the standardizer to scale back the data.

        :return: Standardized tensor.
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
        self, x: Union[torch.Tensor, None], *, train: bool = False, inverse: bool = False, no_grad: bool = True
    ) -> Union[torch.Tensor, None]:
        """Forward pass of the standardizer.

        :param x: Input tensor.
        :param train: Whether to train the standardizer.
        :param inverse: Whether to inverse the standardizer to scale back the data.
        :param no_grad: Whether to disable the gradient computation.

        :return: Standardized tensor.

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
        """
        if x is None:
            return None
        if no_grad:
            with torch.no_grad():
                return self._compute(x, train=train, inverse=inverse)
        return self._compute(x, train=train, inverse=inverse)
