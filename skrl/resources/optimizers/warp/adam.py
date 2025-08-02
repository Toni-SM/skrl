from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import warp as wp

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


@wp.kernel(enable_backward=False)
def _adam_step(
    param: wp.array(ndim=1),
    grad: wp.array(ndim=1),
    m1: wp.array(ndim=1),
    m2: wp.array(ndim=1),
    t: wp.array(ndim=1),
    lr: wp.array(ndim=1),
    beta1: float,
    beta2: float,
    eps: float,
):
    i = wp.tid()
    m1[i] = beta1 * m1[i] + (1.0 - beta1) * grad[i]
    m2[i] = beta2 * m2[i] + (1.0 - beta2) * grad[i] * grad[i]
    m1_hat = m1[i] / (1.0 - wp.pow(beta1, wp.float32(t[0])))
    m2_hat = m2[i] / (1.0 - wp.pow(beta2, wp.float32(t[0])))
    param[i] = param[i] - lr[0] * m1_hat / (wp.sqrt(m2_hat) + eps)


@wp.kernel(enable_backward=False)
def _increase_timestep(t: wp.array(ndim=1)):
    t[0] += 1


def clip_by_total_norm(
    gradients: Sequence[wp.array], max_norm: float, sum_squares: Optional[wp.array] = None
) -> Sequence[wp.array]:
    """Clip (scaling down) gradients' values in-place by their total norm.

    https://arxiv.org/abs/1211.5063

    :param gradients: Gradients to clip.
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


def adam_step(
    params: Sequence[wp.array],
    gradients: Sequence[wp.array],
    m1: Sequence[wp.array],
    m2: Sequence[wp.array],
    t: wp.array,
    lr: wp.array,
    betas: Tuple[float, float],
    eps: float,
) -> None:
    """Perform an optimization step to update parameters.

    :param params: Parameters.
    :param gradients: Gradients.
    :param m1: First moment of the parameters.
    :param m2: Second moment of the parameters.
    :param t: Timestep.
    :param lr: Learning rate.
    :param betas: Beta coefficients.
    :param eps: Term added to the denominator to improve numerical stability.
    """
    wp.launch(_increase_timestep, dim=1, inputs=[t])
    for i in range(len(params)):
        wp.launch(
            _adam_step,
            dim=params[i].shape[0],
            inputs=[params[i], gradients[i], m1[i], m2[i], t, lr, betas[0], betas[1], eps],
        )


class Adam:
    def __init__(
        self,
        params: Sequence[wp.array],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        device: Optional[Union[str, wp.context.Device]] = None,
    ) -> None:
        """Adam optimizer.

        Adapted from Warp implementation of `warp.optim.Adam <https://nvidia.github.io/warp>`_
        to support CUDA graphs, gradient clipping and state dict.

        :param params: Model parameters.
        :param lr: Learning rate.
        :param betas: Coefficients for the running averages of the gradient and its square.
        :param eps: Term added to the denominator to improve numerical stability.
        """
        self.device = config.warp.parse_device(device)
        self.params = [param.flatten() for param in params]
        self.gradients = [param.grad.flatten() for param in self.params]

        self._betas = betas
        self._eps = eps
        self._t = wp.zeros((1,), dtype=wp.int32, device=self.device)
        self._lr = wp.array([lr], dtype=wp.float32, device=self.device)
        self._m1 = [wp.zeros_like(param) for param in self.params]
        self._m2 = [wp.zeros_like(param) for param in self.params]

        self._graph_adam_step = None
        self._graph_clip_by_total_norm = None
        self._use_graph = self.device.is_cuda
        self._cached_sum_squares = wp.zeros((1,), dtype=wp.float32, device=self.device)

    def step(self, *, lr: Optional[float] = None) -> None:
        """Perform an optimization step to update parameters.

        :param lr: Learning rate.
        """
        if lr is not None:
            self._lr.fill_(lr)
        if self._use_graph:
            if self._graph_adam_step is None:
                with wp.ScopedCapture() as capture:
                    adam_step(
                        self.params, self.gradients, self._m1, self._m2, self._t, self._lr, self._betas, self._eps
                    )
                self._graph_adam_step = capture.graph
            else:
                wp.capture_launch(self._graph_adam_step)
        else:
            adam_step(self.params, self.gradients, self._m1, self._m2, self._t, self._lr, self._betas, self._eps)

    def state_dict(self) -> Mapping[str, Any]:
        raise NotImplementedError

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        raise NotImplementedError

    def clip_by_total_norm(self, max_norm: float):
        """Clip (scaling down) parameters' gradients in-place by their total norm.

        .. note::

            This method captures, and launches, the computation done by the ``clip_by_total_norm`` function
            on a CUDA graph for performance reasons.

        https://arxiv.org/abs/1211.5063

        :param max_norm: Maximum global norm.
        """
        self._cached_sum_squares.zero_()
        if self._use_graph:
            if self._graph_clip_by_total_norm is None:
                with wp.ScopedCapture() as capture:
                    clip_by_total_norm(self.gradients, max_norm, self._cached_sum_squares)
                self._graph_clip_by_total_norm = capture.graph
            else:
                wp.capture_launch(self._graph_clip_by_total_norm)
        else:
            clip_by_total_norm(self.gradients, max_norm, self._cached_sum_squares)
