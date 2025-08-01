from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import warp as wp
import warp.optim as optim

from skrl import config


@wp.kernel(enable_backward=False)
def _sum_squares(src: wp.array(ndim=1), dst: wp.array(ndim=1)):
    wp.atomic_add(dst, 0, wp.pow(src[wp.tid()], 2.0))


@wp.kernel(enable_backward=False)
def _clip_by_total_norm(src: wp.array(ndim=1), sum_squares: wp.array(ndim=1), max_norm: float):
    i = wp.tid()
    norm = wp.sqrt(sum_squares[0])
    if norm > max_norm:
        src[i] = src[i] / norm * max_norm


def clip_by_total_norm(
    gradients: Sequence[wp.array], max_norm: float, sum_squares: Optional[wp.array] = None
) -> Sequence[wp.array]:
    """Clip (scaling down) gradients' values in place by their total norm.

    https://arxiv.org/abs/1211.5063

    :param gradients: List of flattened gradients to clip.
    :param max_norm: Maximum global norm.
    :param sum_squares: Pre-allocated array to store the sum of squares of the gradients for intermediate computation.
        If not provided, a new array will be allocated for computation purposes.

    :return: Clipped gradients.
    """
    if sum_squares is None:
        sum_squares = wp.zeros((1,), dtype=wp.float32, device=gradients[0].device)
    for gradient in gradients:
        wp.launch(_sum_squares, dim=gradient.shape[0], inputs=[gradient], outputs=[sum_squares])
    for gradient in gradients:
        wp.launch(_clip_by_total_norm, dim=gradient.shape[0], inputs=[gradient, sum_squares, max_norm])
    return gradients


class Adam(optim.Adam):
    def __init__(
        self,
        params: Sequence[wp.array],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        device: Optional[Union[str, wp.context.Device]] = None,
    ) -> None:
        """Adam optimizer.

        Adapted from `Warp's Adam <https://nvidia.github.io/warp>`_ to support state dict.

        :param params: Model parameters.
        :param lr: Learning rate.
        :param betas: Coefficients for the running averages of the gradient and its square.
        :param eps: Term added to the denominator to improve numerical stability.
        """
        super().__init__([param.flatten() for param in params], lr=lr, betas=betas, eps=eps)

        self.device = config.warp.parse_device(device)

        self._graph = None
        self._use_graph = self.device.is_cuda
        self._cached_sum_squares = wp.zeros((1,), dtype=wp.float32, device=self.device)

    def step(self, gradients: Sequence[wp.array], *, lr: Optional[float] = None) -> None:
        """Perform an optimization step to update parameters.

        :param gradients: Gradients of the parameters.
        :param lr: Learning rate.
        """
        if lr is not None:
            self.lr = lr
        super().step(gradients)

    def state_dict(self) -> Mapping[str, Any]:
        raise NotImplementedError

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        raise NotImplementedError

    def clip_by_total_norm(self, gradients: Sequence[wp.array], max_norm: float) -> Sequence[wp.array]:
        """Clip (scaling down) arrays' values in place by their total norm.

        .. note::

            This method captures, and launches, the computation done by the ``clip_by_total_norm`` function
            on a CUDA graph for performance reasons.

        https://arxiv.org/abs/1211.5063

        :param arrays: List of flattened arrays to clip.
        :param max_norm: Maximum global norm.

        :return: Clipped arrays.
        """
        self._cached_sum_squares.zero_()
        if self._use_graph:
            if self._graph is None:
                with wp.ScopedCapture() as capture:
                    clip_by_total_norm(gradients, max_norm, self._cached_sum_squares)
                self._graph = capture.graph
            else:
                wp.capture_launch(self._graph)
        else:
            clip_by_total_norm(gradients, max_norm, self._cached_sum_squares)
        return gradients
