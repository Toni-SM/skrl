from typing import Optional, Tuple, Union

import warp as wp

from skrl import config
from skrl.resources.noises.warp import Noise


@wp.kernel
def _sample(
    dst: wp.array2d(dtype=float),
    theta: float,
    sigma: float,
    state: wp.array2d(dtype=float),
    mean: float,
    std: float,
    base_scale: float,
    key: int,
):
    i, j = wp.tid()
    subkey = wp.rand_init(key + i, j)
    state[i, j] = state[i, j] * theta + sigma * (wp.randn(subkey) * std + mean)
    dst[i, j] = base_scale * state[i, j]


class OrnsteinUhlenbeckNoise(Noise):
    def __init__(
        self,
        *,
        theta: float,
        sigma: float,
        base_scale: float,
        mean: float = 0,
        std: float = 1,
        device: Optional[Union[str, wp.context.Device]] = None,
    ) -> None:
        """Ornstein-Uhlenbeck noise.

        :param theta: Factor to apply to current internal state.
        :param sigma: Factor to apply to the normal distribution.
        :param base_scale: Factor to apply to returned noise.
        :param mean: Mean of the normal distribution.
        :param std: Standard deviation of the normal distribution.
        :param device: Data allocation and computation device. If not specified, the default device will be used.

        Example::

            >>> noise = OrnsteinUhlenbeckNoise(theta=0.1, sigma=0.2, base_scale=0.5)
        """
        super().__init__(device=device)

        self.state = None
        self.theta = theta
        self.sigma = sigma
        self.base_scale = base_scale

        self._key = config.warp.key
        self.mean = mean
        self.std = std

    def sample(self, size: Tuple[int]) -> wp.array:
        """Sample an Ornstein-Uhlenbeck noise.

        :param size: Noise shape.

        :return: Sampled noise.

        Example::

            >>> noise.sample((3, 2))
            Array([[ 0.01878439, -0.12833427],
                   [ 0.06494182,  0.12490594],
                   [ 0.024447  , -0.01174496]], dtype=float32)

            >>> x = jax.random.uniform(jax.random.PRNGKey(0), (3, 2))
            >>> noise.sample(x.shape)
            Array([[ 0.17988093, -1.2289404 ],
                   [ 0.6218886 ,  1.1961104 ],
                   [ 0.23410667, -0.11247082]], dtype=float32)
        """
        if self.state is None or self.state.shape != tuple(size):
            self.state = wp.zeros(shape=size, dtype=wp.float32, device=self.device)
        self._key += 1
        output = wp.empty(shape=size, dtype=wp.float32, device=self.device)
        wp.launch(
            _sample,
            dim=output.shape,
            inputs=[output, self.theta, self.sigma, self.state, self.mean, self.std, self.base_scale, self._key],
            device=self.device,
        )
        return output
