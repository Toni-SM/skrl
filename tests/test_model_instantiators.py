import pytest
import warnings
import hypothesis
import hypothesis.strategies as st

import torch

from skrl.models.torch import Model

from skrl.utils.model_instantiators import Shape
from skrl.utils.model_instantiators import categorical_model
from skrl.utils.model_instantiators import deterministic_model
from skrl.utils.model_instantiators import gaussian_model
from skrl.utils.model_instantiators import multivariate_gaussian_model


@pytest.fixture
def classes_and_kwargs():
    return []


@pytest.mark.parametrize("device", [None, "cpu", "cuda:0"])
def test_device(capsys, classes_and_kwargs, device):
    _device = torch.device(device) if device is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
