import hypothesis
import hypothesis.strategies as st
import pytest

import numpy as np
import warp as wp

from skrl.resources.noises.warp import OrnsteinUhlenbeckNoise

from ....utilities import is_device_available


@hypothesis.given(
    shape=st.lists(st.integers(min_value=1, max_value=10), min_size=2, max_size=2),
    theta=st.floats(min_value=0, max_value=1),
    base_scale=st.floats(min_value=0, max_value=10),
)
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture],
    deadline=None,
    phases=[hypothesis.Phase.explicit, hypothesis.Phase.reuse, hypothesis.Phase.generate],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_mean_reversion(capsys, device, shape, theta, base_scale):
    # check device availability
    if not is_device_available(device, backend="warp"):
        pytest.skip(f"Device {device} not available")

    # sigma=0 removes the innovation term: x[n] = (1 - theta)^n * x[0]
    noise = OrnsteinUhlenbeckNoise(theta=theta, sigma=0, base_scale=base_scale, device=device)
    initial_state = np.random.uniform(low=-10, high=10, size=shape).astype(np.float32)
    noise.state = wp.array(initial_state, device=device)
    for step in range(1, 6):
        sample = noise.sample(tuple(shape))
        state = initial_state * (1 - theta) ** step
        assert np.allclose(noise.state.numpy(), state, atol=1e-05, rtol=1e-04)
        assert np.allclose(sample.numpy(), base_scale * state, atol=1e-04, rtol=1e-04)


@hypothesis.given(
    shape=st.lists(st.integers(min_value=1, max_value=10), min_size=2, max_size=2),
)
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture],
    deadline=None,
    phases=[hypothesis.Phase.explicit, hypothesis.Phase.reuse, hypothesis.Phase.generate],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_shape_change(capsys, device, shape):
    # check device availability
    if not is_device_available(device, backend="warp"):
        pytest.skip(f"Device {device} not available")

    noise = OrnsteinUhlenbeckNoise(theta=0.15, sigma=0, base_scale=1, device=device)
    noise.state = wp.ones((11, 11), dtype=wp.float32, device=device)
    sample = noise.sample(tuple(shape))
    assert sample.shape == tuple(shape)
    assert np.allclose(sample.numpy(), 0)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_sample_statistics(capsys, device):
    # check device availability
    if not is_device_available(device, backend="warp"):
        pytest.skip(f"Device {device} not available")

    theta, sigma, mean, std = 0.15, 0.4, 0.5, 2.0
    noise = OrnsteinUhlenbeckNoise(theta=theta, sigma=sigma, base_scale=1, mean=mean, std=std, device=device)
    # first sample (from zero state) is the innovation: sigma * N(mean, std)
    previous = noise.sample((16384, 1)).numpy().astype(np.float64)
    assert np.isclose(previous.mean(), sigma * mean, atol=0.05)
    assert np.isclose(previous.std(), sigma * std, rtol=0.05)
    # the second sample regresses on the first one with slope 1 - theta
    following = noise.sample((16384, 1)).numpy().astype(np.float64) - sigma * mean
    slope = np.sum(previous * following) / np.sum(previous * previous)
    assert np.isclose(slope, 1 - theta, atol=0.05)
