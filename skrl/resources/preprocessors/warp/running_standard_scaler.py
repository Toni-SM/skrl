from typing import Mapping, Optional, Tuple, Union

import gymnasium

import numpy as np
import warp as wp

from skrl import config
from skrl.utils.spaces.warp import compute_space_size


@wp.kernel(enable_backward=False)
def _inverse_2d(
    src: wp.array(ndim=2),
    running_mean: wp.array(ndim=1),
    running_variance: wp.array(ndim=1),
    clip_threshold: float,
    dst: wp.array(ndim=2),
):
    i, j = wp.tid()
    dst[i, j] = wp.sqrt(running_variance[j]) * wp.clamp(src[i, j], -clip_threshold, clip_threshold) + running_mean[j]


def _inverse_3d(
    src: wp.array(ndim=3),
    running_mean: wp.array(ndim=1),
    running_variance: wp.array(ndim=1),
    clip_threshold: float,
    dst: wp.array(ndim=3),
):
    i, j, k = wp.tid()
    dst[i, j, k] = (
        wp.sqrt(running_variance[k]) * wp.clamp(src[i, j, k], -clip_threshold, clip_threshold) + running_mean[k]
    )


@wp.kernel(enable_backward=False)
def _standardization_2d(
    src: wp.array(ndim=2),
    running_mean: wp.array(ndim=1),
    running_variance: wp.array(ndim=1),
    clip_threshold: float,
    epsilon: float,
    dst: wp.array(ndim=2),
):
    i, j = wp.tid()
    dst[i, j] = wp.clamp(
        (src[i, j] - running_mean[j]) / (wp.sqrt(running_variance[j]) + epsilon), -clip_threshold, clip_threshold
    )


@wp.kernel(enable_backward=False)
def _standardization_3d(
    src: wp.array(ndim=3),
    running_mean: wp.array(ndim=1),
    running_variance: wp.array(ndim=1),
    clip_threshold: float,
    epsilon: float,
    dst: wp.array(ndim=3),
):
    i, j, k = wp.tid()
    dst[i, j, k] = wp.clamp(
        (src[i, j, k] - running_mean[k]) / (wp.sqrt(running_variance[k]) + epsilon), -clip_threshold, clip_threshold
    )


@wp.kernel(enable_backward=False)
def _mean_2d_axis_0(src: wp.array(ndim=2), n: int, dst: wp.array(ndim=1)):
    i, j = wp.tid()
    wp.atomic_add(dst, j, dst.dtype(src[i, j]) / dst.dtype(n))


@wp.kernel(enable_backward=False)
def _mean_3d_axis_0_1(src: wp.array(ndim=3), n: int, dst: wp.array(ndim=1)):
    i, j, k = wp.tid()
    wp.atomic_add(dst, k, dst.dtype(src[i, j, k]) / dst.dtype(n))


@wp.kernel(enable_backward=False)
def _var_2d_axis_0(src: wp.array(ndim=2), mean: wp.array(ndim=1), n: int, dst: wp.array(ndim=1)):
    i, j = wp.tid()
    wp.atomic_add(dst, j, wp.pow(dst.dtype(src[i, j]) - mean[j], 2.0) / dst.dtype(n))


@wp.kernel(enable_backward=False)
def _var_3d_axis_0_1(src: wp.array(ndim=3), mean: wp.array(ndim=1), n: int, dst: wp.array(ndim=1)):
    i, j, k = wp.tid()
    wp.atomic_add(dst, k, wp.pow(dst.dtype(src[i, j, k]) - mean[k], 2.0) / dst.dtype(n))


@wp.kernel(enable_backward=False)
def _parallel_variance(
    running_mean: wp.array(ndim=1),
    running_variance: wp.array(ndim=1),
    current_count: float,
    input_mean: wp.array(ndim=1),
    input_variance: wp.array(ndim=1),
    input_count: float,
):
    i = wp.tid()
    delta = input_mean[i] - running_mean[i]
    total_count = current_count + input_count
    M2 = (
        (running_variance[i] * current_count)
        + (input_variance[i] * input_count)
        + wp.pow(delta, 2.0) * current_count * input_count / total_count
    )

    running_mean[i] = running_mean[i] + delta * input_count / total_count
    running_variance[i] = M2 / total_count


class RunningStandardScaler:
    def __init__(
        self,
        size: Union[int, Tuple[int], gymnasium.Space],
        *,
        epsilon: float = 1e-8,
        clip_threshold: float = 5.0,
        device: Optional[Union[str, wp.context.Device]] = None,
    ) -> None:
        """Standardize the input data by removing the mean and scaling by the standard deviation.

        :param size: Size of the input space.
        :param epsilon: Small number to avoid division by zero.
        :param clip_threshold: Threshold to clip the data.
        :param device: Data allocation and computation device. If not specified, the default device will be used.

        Example::

            >>> running_standard_scaler = RunningStandardScaler(size=2)
            >>> data = rand(3, 2)  # tensor of shape (N, 2)
            >>> running_standard_scaler(data)
            tensor([[0.1954, 0.3356],
                    [0.9719, 0.4163],
                    [0.8540, 0.1982]])
        """
        super().__init__()

        self.epsilon = epsilon
        self.clip_threshold = clip_threshold

        self.device = config.warp.parse_device(device)

        size = compute_space_size(size, occupied_size=True)

        self.current_count = 1.0
        self.running_mean = wp.zeros(size, dtype=wp.float32, device=self.device)
        self.running_variance = wp.ones(size, dtype=wp.float32, device=self.device)
        self._cached_mean = wp.empty(size, dtype=wp.float32, device=self.device)
        self._cached_variance = wp.empty(size, dtype=wp.float32, device=self.device)

    def state_dict(self) -> Mapping[str, wp.array]:
        """Dictionary containing references to the whole state of the module."""
        return {
            "running_mean": self.running_mean,
            "running_variance": self.running_variance,
            "current_count": self.current_count,
        }

    def load_state_dict(self, state_dict: Mapping[str, wp.array]) -> None:
        wp.copy(self.running_mean, state_dict["running_mean"])
        wp.copy(self.running_variance, state_dict["running_variance"])
        self.current_count = float(state_dict["current_count"])

    def __call__(
        self, x: Union[wp.array, None], *, train: bool = False, inverse: bool = False, inplace: bool = False
    ) -> Union[wp.array, None]:
        """Forward pass of the standardizer.

        :param x: Input tensor.
        :param train: Whether to train the standardizer.
        :param inverse: Whether to inverse the standardizer to scale back the data.
        :param no_grad: Whether to disable the gradient computation.
        :param inplace: Whether to perform the operation in-place.

        :return: Standardized tensor.

        Example::

            >>> x = rand(3, 2, device="cuda:0")
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
        ndim = x.ndim
        if train:
            n = np.prod(x.shape[:-1]).item()
            self._cached_mean.zero_()
            self._cached_variance.zero_()
            wp.launch(
                _mean_2d_axis_0 if ndim == 2 else _mean_3d_axis_0_1,
                dim=x.shape,
                inputs=[x, n],
                outputs=[self._cached_mean],
                device=self.device,
            )
            wp.launch(
                _var_2d_axis_0 if ndim == 2 else _var_3d_axis_0_1,
                dim=x.shape,
                inputs=[x, self._cached_mean, n - 1],  # ddof = 1: https://github.com/pytorch/pytorch/issues/50010
                outputs=[self._cached_variance],
                device=self.device,
            )
            wp.launch(
                _parallel_variance,
                dim=self.running_mean.shape,
                inputs=[
                    self.running_mean,
                    self.running_variance,
                    self.current_count,
                    self._cached_mean,
                    self._cached_variance,
                    float(n),
                ],
                device=self.device,
            )
            self.current_count += n

        # scale back the data to the original representation
        if inverse:
            output = x if inplace else wp.empty_like(x)
            wp.launch(
                _inverse_2d if ndim == 2 else _inverse_3d,
                dim=x.shape,
                inputs=[x, self.running_mean, self.running_variance, self.clip_threshold],
                outputs=[output],
                device=self.device,
            )
            return output
        # standardization by centering and scaling
        output = x if inplace else wp.empty_like(x)
        wp.launch(
            _standardization_2d if ndim == 2 else _standardization_3d,
            dim=x.shape,
            inputs=[x, self.running_mean, self.running_variance, self.clip_threshold, self.epsilon],
            outputs=[output],
            device=self.device,
        )
        return output
