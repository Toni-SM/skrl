import pytest

import torch

from skrl.resources.noises.torch import GaussianNoise
from skrl.resources.noises.torch import OrnsteinUhlenbeckNoise


@pytest.fixture
def classes_and_kwargs():
    return [(GaussianNoise, {"mean": 0, "std": 1}),
            (OrnsteinUhlenbeckNoise, {"theta": 0.1, "sigma": 0.2, "base_scale": 0.3})]


@pytest.mark.parametrize("device", [None, "cpu", "cuda:0"])
def test_device(classes_and_kwargs, device):
    _device = torch.device(device) if device is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for klass, kwargs in classes_and_kwargs:
        noise = klass(device=device, **kwargs)

        output = noise.sample((1,))
        assert noise.device == _device  # defined device
        assert output.device == _device  # runtime device

@pytest.mark.parametrize("size", [(10,), [20, 1], torch.Size([30, 1, 2])])
def test_sampling(classes_and_kwargs, size):
    for klass, kwargs in classes_and_kwargs:
        noise = klass(**kwargs)

        # sample
        output = noise.sample(size)
        assert output.size() == torch.Size(size)

        # sample like
        tensor = torch.rand(size, device="cpu")
        output = noise.sample_like(tensor)
        assert output.size() == torch.Size(size)
