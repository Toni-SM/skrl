import warnings
import hypothesis
import hypothesis.strategies as st
import pytest

import torch

from skrl.resources.schedulers.torch import KLAdaptiveRL


@pytest.fixture
def classes_and_kwargs():
    return [(KLAdaptiveRL, {})]


@pytest.mark.parametrize("optimizer", [torch.optim.Adam([torch.ones((1,))], lr=0.1),
                                       torch.optim.SGD([torch.ones((1,))], lr=0.1)])
def test_step(capsys, classes_and_kwargs, optimizer):
    for klass, kwargs in classes_and_kwargs:
        scheduler = klass(optimizer, **kwargs)

        scheduler.step(0.0)
