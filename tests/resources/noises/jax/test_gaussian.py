import hypothesis
import hypothesis.strategies as st
import pytest

import numpy as np

from skrl.resources.noises.jax import GaussianNoise

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
    if not is_device_available(device, backend="jax"):
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
    if not is_device_available(device, backend="jax"):
        pytest.skip(f"Device {device} not available")

    noise = GaussianNoise(mean=mean, std=std, device=device)
    sample = np.array(noise.sample((16384, 1))).astype(np.float64)
    assert np.isclose(sample.mean(), mean, atol=0.05 * std)
    assert np.isclose(sample.std(), std, rtol=0.05)


def test_list_shape_matches_tuple_samples():
    list_noise = GaussianNoise(mean=0.5, std=2, device="cpu")
    tuple_noise = GaussianNoise(mean=0.5, std=2, device="cpu")
    samples = []
    for _ in range(2):
        sample = np.array(list_noise.sample([3, 2]))
        np.testing.assert_array_equal(sample, np.array(tuple_noise.sample((3, 2))))
        samples.append(sample)
    assert not np.array_equal(samples[0], samples[1])
