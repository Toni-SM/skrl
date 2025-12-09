import hypothesis
import hypothesis.strategies as st
import pytest

import gymnasium

import numpy as np
import warp as wp

from skrl import config
from skrl.resources.preprocessors.warp import RunningStandardScaler

from ....utilities import is_device_available


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_preprocessor(capsys, device):
    def _check(array, ground_truth):
        array = array.numpy()
        stats = [np.min(array), np.max(array), np.mean(array), np.std(array)]
        assert np.allclose(stats, ground_truth, rtol=1e-04, atol=1e-07)

    # check device availability
    if not is_device_available(device, backend="torch"):
        pytest.skip(f"Device {device} not available")

    space = gymnasium.spaces.Box(low=-1, high=1, shape=(100,))
    preprocessor = RunningStandardScaler(size=space, device=device)

    for i in range(10):
        batch_size = 100 + 10 * i
        data = np.linspace(-10 + i, 10 + i, batch_size * 100, dtype=np.float32).reshape(batch_size, -1)

        x = wp.array(data, device=config.warp.parse_device(device))
        y = preprocessor(x)
        y_inverse = preprocessor(x, inverse=True)
        y_train = preprocessor(x, train=True)

        if i == 0:
            _check(x, [-10.0, 10.0, 0.0, 5.77408])
            _check(y, [-5.0, 5.0, 0.0, 4.08258])
            _check(y_inverse, [-5.0, 5.0, 0.0, 4.08258])
            _check(y_train, [-1.71464, 1.71464, 0.0, 0.99980])
        elif i == 9:
            _check(x, [-1.0, 19.0, 9.0, 5.77380])
            _check(y, [-0.86070, 2.28972, 0.71582, 0.91289])
            _check(y_inverse, [-1.92795, 36.14016, 30.40295, 10.56556])
            _check(y_train, [-0.93701, 2.15588, 0.61087, 0.89643])
