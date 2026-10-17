import hypothesis
import hypothesis.strategies as st
import pytest

import numpy as np

from skrl.resources.noises.warp import GaussianNoise

from ....utilities import is_device_available


@hypothesis.given(
    shape=st.lists(st.integers(min_value=1, max_value=10), min_size=2, max_size=2),
)
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture],
    deadline=None,
    phases=[hypothesis.Phase.explicit, hypothesis.Phase.reuse, hypothesis.Phase.generate],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_sample_shape(capsys, device, shape):
    # check device availability
    if not is_device_available(device, backend="warp"):
        pytest.skip(f"Device {device} not available")

    noise = GaussianNoise(mean=0, std=1, device=device)
    sample = noise.sample(tuple(shape))
    assert sample.shape == tuple(shape)


@hypothesis.given(
    mean=st.floats(min_value=-5, max_value=5),
    std=st.floats(min_value=0.1, max_value=5),
)
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture],
    deadline=None,
    phases=[hypothesis.Phase.explicit, hypothesis.Phase.reuse, hypothesis.Phase.generate],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_sample_statistics(capsys, device, mean, std):
    # check device availability
    if not is_device_available(device, backend="warp"):
        pytest.skip(f"Device {device} not available")

    noise = GaussianNoise(mean=mean, std=std, device=device)
    sample = noise.sample((16384, 1)).numpy().astype(np.float64)
    assert np.isclose(sample.mean(), mean, atol=0.05 * std)
    assert np.isclose(sample.std(), std, rtol=0.05)
