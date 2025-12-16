import hypothesis
import hypothesis.strategies as st
import pytest

import gymnasium

import numpy as np
import torch

from skrl import config
from skrl.resources.preprocessors.torch import RunningStandardScaler

from ....utilities import is_device_available


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_preprocessor(capsys, device):
    def _sample(batch_size, offset=0, ndim=2):
        if ndim == 2:
            value = rng.standard_normal((batch_size, 100), dtype=np.float32).reshape(batch_size, 100)
        elif ndim == 3:
            value = rng.standard_normal((batch_size, 50, 100), dtype=np.float32).reshape(batch_size, 50, 100)
        else:
            raise ValueError(f"Unsupported ndim ({ndim})")
        return 2 * value + offset

    def _compute(data):
        x = torch.tensor(data, device=config.torch.parse_device(device))
        y = preprocessor(x)
        y_inverse = preprocessor(x, inverse=True)
        y_train = preprocessor(x, train=True)
        return y, y_inverse, y_train

    def _check(array, ground_truth, rtol=1e-05, atol=1e-05):
        array = array.cpu().numpy()
        stats = [np.min(array), np.max(array), np.mean(array), np.std(array)]
        # with capsys.disabled():
        #     print([round(s, 5) for s in stats])
        assert np.allclose(stats, ground_truth, rtol=rtol, atol=atol)

    # check device availability
    if not is_device_available(device, backend="torch"):
        pytest.skip(f"Device {device} not available")

    space = gymnasium.spaces.Box(low=-1, high=1, shape=(100,))
    preprocessor = RunningStandardScaler(size=space, device=device)

    n = 50
    batch_size = 100
    rng = np.random.default_rng(64)

    for i in range(n):
        batch_size += 10 * i
        y, y_inverse, y_train = _compute(_sample(batch_size, offset=5 * i / n))
        if i == 0:
            _check(y, [-5.0, 5.0, 0.00105, 1.98258])
            _check(y_inverse, [-5.0, 5.0, 0.00105, 1.98258])
            _check(y_train, [-3.80794, 3.55748, 1e-05, 0.99865])

    y, y_inverse, y_train = _compute(_sample(batch_size, ndim=2))
    _check(y, [-5.0, 2.50224, -1.64096, 0.89515])
    _check(y_inverse, [-7.54722, 14.88641, 3.67446, 4.42310])
    _check(y_train, [-5.0, 2.44027, -1.46018, 0.84271])

    y, y_inverse, y_train = _compute(_sample(batch_size, ndim=3))
    _check(y, [-5.0, 3.13003, -1.46106, 0.84212])
    _check(y_inverse, [-8.44294, 15.38363, 3.46896, 4.69524])
    _check(y_train, [-4.89637, 3.82424, -0.35610, 0.76704])
