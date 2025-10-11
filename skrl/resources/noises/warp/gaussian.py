from __future__ import annotations

import warp as wp

from skrl import config
from skrl.resources.noises.warp import Noise


@wp.kernel
def _sample(dst: wp.array2d(dtype=float), mean: float, std: float, key: int):
    i, j = wp.tid()
    subkey = wp.rand_init(key + i, j)
    dst[i, j] = wp.randn(subkey) * std + mean


class GaussianNoise(Noise):
    def __init__(self, *, mean: float, std: float, device: str | wp.context.Device | None = None) -> None:
        """Gaussian noise.

        :param mean: Mean of the normal distribution.
        :param std: Standard deviation of the normal distribution.
        :param device: Data allocation and computation device. If not specified, the default device will be used.

        Example::

            >>> noise = GaussianNoise(mean=0, std=1)
        """
        super().__init__(device=device)

        self._key = config.warp.key
        self.mean = mean
        self.std = std

    def sample(self, size: list[int]) -> wp.array:
        """Sample a Gaussian noise.

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
        self._key += 1
        output = wp.empty(shape=size, dtype=wp.float32, device=self.device)
        wp.launch(_sample, dim=output.shape, inputs=[output, self.mean, self.std, self._key], device=self.device)
        return output
