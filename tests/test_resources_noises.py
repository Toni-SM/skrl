import warnings
import hypothesis
import hypothesis.strategies as st
import pytest

import torch

from skrl.resources.noises.torch import GaussianNoise, Noise, OrnsteinUhlenbeckNoise


@pytest.fixture
def classes_and_kwargs():
    return [(GaussianNoise, {"mean": 0, "std": 1}),
            (OrnsteinUhlenbeckNoise, {"theta": 0.1, "sigma": 0.2, "base_scale": 0.3})]


@pytest.mark.parametrize("device", [None, "cpu", "cuda:0"])
def test_device(capsys, classes_and_kwargs, device):
    _device = torch.device(device) if device is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for klass, kwargs in classes_and_kwargs:
        try:
            noise: Noise = klass(device=device, **kwargs)
        except (RuntimeError, AssertionError) as e:
            with capsys.disabled():
                print(e)
            warnings.warn(f"Invalid device: {device}. This test will be skipped")
            continue

        output = noise.sample((1,))
        assert noise.device == _device  # defined device
        assert output.device == _device  # runtime device

@hypothesis.given(size=st.lists(st.integers(min_value=1, max_value=10), max_size=5))
@hypothesis.settings(suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture], deadline=None)
def test_sample(capsys, classes_and_kwargs, size):
    for klass, kwargs in classes_and_kwargs:
        noise: Noise = klass(**kwargs)

        # sample
        output = noise.sample(size)
        assert output.size() == torch.Size(size)

        # sample like
        tensor = torch.rand(size, device="cpu")
        output = noise.sample_like(tensor)
        assert output.size() == torch.Size(size)
