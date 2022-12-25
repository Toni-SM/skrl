import pytest

import gym
import gymnasium
import numpy as np

import torch

from skrl.resources.preprocessors.torch import RunningStandardScaler


@pytest.fixture
def classes_and_kwargs():
    return [(RunningStandardScaler, {"size": 1})]


@pytest.mark.parametrize("device", [None, "cpu", "cuda:0"])
def test_device(classes_and_kwargs, device):
    _device = torch.device(device) if device is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for klass, kwargs in classes_and_kwargs:
        preprocessor = klass(device=device, **kwargs)

        assert preprocessor.device == _device  # defined device
        assert preprocessor(torch.ones(kwargs["size"], device=_device)).device == _device  # runtime device

@pytest.mark.parametrize("space_and_size", [(gym.spaces.Box(low=-1, high=1, shape=(2, 3)), 6),
                                            (gymnasium.spaces.Box(low=-1, high=1, shape=(2, 3)), 6),
                                            (gym.spaces.Discrete(n=3), 1),
                                            (gymnasium.spaces.Discrete(n=3), 1)])
def test_forward(classes_and_kwargs, space_and_size):
    for klass, kwargs in classes_and_kwargs:
        space, size = space_and_size
        preprocessor = klass(size=space, device="cpu")

        output = preprocessor(torch.rand((10, size), device="cpu"))
        assert output.shape == torch.Size((10, size))