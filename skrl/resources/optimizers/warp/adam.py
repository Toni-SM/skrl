from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import math

import warp as wp

from skrl import config
from skrl.utils.framework.warp import ScopedCapture


tiled = wp.constant(config.warp.tiled)
tile_dim_0 = wp.constant(config.warp.tile_dim_0)
block_dim = wp.constant(config.warp.block_dim)


def create_clip_by_total_norm_kernels(max_norm: float):
    @wp.func
    def clip_by_norm(x: wp.float32, sum_squares: wp.float32) -> wp.float32:
        norm = wp.sqrt(sum_squares)
        if norm > wp.static(max_norm):
            return x / norm * wp.static(max_norm)
        return x

    @wp.kernel(enable_backward=False)
    def sum_squares(gradients: wp.array(ndim=1), sum_squares: wp.array(ndim=1)):
        i = wp.tid()
        # tiled implementation
        if wp.static(tiled):
            tiled_gradients = wp.tile_load(gradients, shape=(tile_dim_0,), offset=(i * tile_dim_0,))
            wp.tile_atomic_add(sum_squares, wp.tile_sum(wp.tile_map(wp.mul, tiled_gradients, tiled_gradients)))
        # non-tiled implementation
        else:
            wp.atomic_add(sum_squares, 0, gradients[i] * gradients[i])

    @wp.kernel(enable_backward=False)
    def clip_by_total_norm(gradients: wp.array(ndim=1), sum_squares: wp.array(ndim=1)):
        i = wp.tid()
        # tiled implementation
        if wp.static(tiled):
            tiled_sum_squares = wp.tile_broadcast(wp.tile_load(sum_squares, shape=(1,)), shape=(tile_dim_0,))
            tiled_gradients = wp.tile_load(gradients, shape=(tile_dim_0,), offset=(i * tile_dim_0,))
            tiled_gradients = wp.tile_map(clip_by_norm, tiled_gradients, tiled_sum_squares)
            wp.tile_store(gradients, tiled_gradients, offset=(i * tile_dim_0,))
        # non-tiled implementation
        else:
            norm = wp.sqrt(sum_squares[0])
            if norm > wp.static(max_norm):
                gradients[i] = gradients[i] / norm * wp.static(max_norm)

    return sum_squares, clip_by_total_norm


def create_adam_step_kernels(beta1: float, beta2: float, eps: float):
    @wp.func
    def hat_ratio(m1_hat: wp.float32, m2_hat: wp.float32) -> wp.float32:
        return m1_hat / (wp.sqrt(m2_hat) + wp.static(eps))

    @wp.kernel(enable_backward=False)
    def increase_timestep(t: wp.array(ndim=1)):
        t[0] += 1.0

    @wp.kernel(enable_backward=False)
    def adam_step(
        parameters: wp.array(ndim=1),
        gradients: wp.array(ndim=1),
        m1: wp.array(ndim=1),
        m2: wp.array(ndim=1),
        timestep: wp.array(ndim=1),
        lr: wp.array(ndim=1),
    ):
        i = wp.tid()
        # tiled implementation
        if wp.static(tiled):
            tiled_parameters = wp.tile_load(parameters, shape=(tile_dim_0,), offset=(i * tile_dim_0,))
            tiled_gradients = wp.tile_load(gradients, shape=(tile_dim_0,), offset=(i * tile_dim_0,))
            tiled_m1 = wp.tile_load(m1, shape=(tile_dim_0,), offset=(i * tile_dim_0,))
            tiled_m2 = wp.tile_load(m2, shape=(tile_dim_0,), offset=(i * tile_dim_0,))

            tiled_m1 = wp.static(beta1) * tiled_m1 + wp.static(1.0 - beta1) * tiled_gradients
            tiled_m2 = wp.static(beta2) * tiled_m2 + wp.static(1.0 - beta2) * wp.tile_map(
                wp.mul, tiled_gradients, tiled_gradients
            )
            m1_hat = tiled_m1 * (1.0 / (1.0 - wp.pow(wp.static(beta1), timestep[0])))
            m2_hat = tiled_m2 * (1.0 / (1.0 - wp.pow(wp.static(beta2), timestep[0])))

            wp.tile_store(m1, tiled_m1, offset=(i * tile_dim_0,))
            wp.tile_store(m2, tiled_m2, offset=(i * tile_dim_0,))
            wp.tile_store(
                parameters, tiled_parameters - lr[0] * wp.tile_map(hat_ratio, m1_hat, m2_hat), offset=(i * tile_dim_0,)
            )
        # non-tiled implementation
        else:
            m1[i] = wp.static(beta1) * m1[i] + wp.static(1.0 - beta1) * gradients[i]
            m2[i] = wp.static(beta2) * m2[i] + wp.static(1.0 - beta2) * gradients[i] * gradients[i]
            m1_hat = m1[i] / (1.0 - wp.pow(wp.static(beta1), timestep[0]))
            m2_hat = m2[i] / (1.0 - wp.pow(wp.static(beta2), timestep[0]))
            parameters[i] = parameters[i] - lr[0] * m1_hat / (wp.sqrt(m2_hat) + wp.static(eps))

    return increase_timestep, adam_step


class Adam:
    def __init__(
        self,
        parameters: Sequence[wp.array],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        device: Optional[Union[str, wp.context.Device]] = None,
    ) -> None:
        """Adam optimizer.

        Adapted from Warp implementation of `warp.optim.Adam <https://nvidia.github.io/warp>`_
        to support CUDA graphs, gradient clipping and state dict.

        :param parameters: Model parameters.
        :param lr: Learning rate.
        :param betas: Coefficients for the running averages of the gradient and its square.
        :param eps: Term added to the denominator to improve numerical stability.
        """
        self.device = config.warp.parse_device(device)
        self.parameters = [param.flatten() for param in parameters]
        self.gradients = [param.grad.flatten() for param in self.parameters]

        self._betas = betas
        self._eps = eps
        self._m1 = [wp.zeros_like(param, requires_grad=False) for param in self.parameters]
        self._m2 = [wp.zeros_like(param, requires_grad=False) for param in self.parameters]
        self._lr = wp.array([lr], dtype=wp.float32, device=self.device)
        self._timestep = wp.zeros((1,), dtype=wp.float32, device=self.device)

        self._use_graph = self.device.is_cuda
        self._graph_clip_by_total_norm = None
        self._graph_adam_step = None
        self._max_norm = None

        self._sum_squares_kernel, self._clip_by_total_norm_kernel = None, None
        self._increase_timestep_kernel, self._adam_step_kernel = create_adam_step_kernels(*self._betas, self._eps)

    def step(self, *, lr: Optional[float] = None) -> None:
        """Perform an optimization step to update parameters.

        :param lr: Learning rate.
        """
        # update learning rate
        if lr is not None:
            self._lr.fill_(lr)
        # perform optimization step
        if self._graph_adam_step is None:
            with ScopedCapture(device=self.device, enabled=self._use_graph) as capture:
                wp.launch(
                    self._increase_timestep_kernel,
                    dim=1,
                    inputs=[self._timestep],
                    device=self.device,
                    block_dim=block_dim,
                )
                for parameters, gradients, m1, m2 in zip(self.parameters, self.gradients, self._m1, self._m2):
                    wp.launch(
                        self._adam_step_kernel,
                        dim=[math.ceil(parameters.shape[0] / tile_dim_0), block_dim] if tiled else parameters.shape[0],
                        inputs=[parameters, gradients, m1, m2, self._timestep, self._lr],
                        device=self.device,
                        block_dim=block_dim,
                    )
            self._graph_adam_step = capture.graph
        else:
            wp.capture_launch(self._graph_adam_step)

    def state_dict(self) -> Mapping[str, Any]:
        raise NotImplementedError

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        raise NotImplementedError

    def clip_by_total_norm(self, max_norm: float):
        """Clip (scaling down) parameters' gradients in-place by their total norm.

        https://arxiv.org/abs/1211.5063

        :param max_norm: Maximum global norm.
        """
        # create kernels if not already done or if `max_norm` has changed
        if max_norm != self._max_norm:
            self._max_norm = max_norm
            self._graph_clip_by_total_norm = None
            self._sum_squares = wp.zeros((1,), dtype=wp.float32, device=self.device)
            self._sum_squares_kernel, self._clip_by_total_norm_kernel = create_clip_by_total_norm_kernels(max_norm)
        # clip gradients
        self._sum_squares.zero_()
        if self._graph_clip_by_total_norm is None:
            with ScopedCapture(device=self.device, enabled=self._use_graph) as capture:
                for gradient in self.gradients:
                    wp.launch(
                        self._sum_squares_kernel,
                        dim=[math.ceil(gradient.shape[0] / tile_dim_0), block_dim] if tiled else gradient.shape[0],
                        inputs=[gradient],
                        outputs=[self._sum_squares],
                        device=self.device,
                        block_dim=block_dim,
                    )
                for gradient in self.gradients:
                    wp.launch(
                        self._clip_by_total_norm_kernel,
                        dim=[math.ceil(gradient.shape[0] / tile_dim_0), block_dim] if tiled else gradient.shape[0],
                        inputs=[gradient, self._sum_squares],
                        device=self.device,
                        block_dim=block_dim,
                    )
            self._graph_clip_by_total_norm = capture.graph
        else:
            wp.capture_launch(self._graph_clip_by_total_norm)
