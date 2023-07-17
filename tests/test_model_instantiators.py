import warnings
import hypothesis
import hypothesis.strategies as st
import pytest

import torch

from skrl.models.torch import Model
from skrl.utils.model_instantiators import (
    Shape,
    categorical_model,
    deterministic_model,
    gaussian_model,
    multivariate_gaussian_model
)


@pytest.fixture
def classes_and_kwargs():
    return [(categorical_model, {}),
            (deterministic_model, {}),
            (gaussian_model, {}),
            (multivariate_gaussian_model, {})]


def test_models(capsys, classes_and_kwargs):
    for klass, kwargs in classes_and_kwargs:
        model: Model = klass(observation_space=1, action_space=1, device="cpu", **kwargs)
