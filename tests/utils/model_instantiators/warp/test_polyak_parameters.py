import pytest

from gymnasium import spaces

import numpy as np
import warp as wp
import warp_nn.nn as nn

from skrl.models.warp import DeterministicMixin, Model
from skrl.utils.model_instantiators.warp import gaussian_model


class ParameterModel(DeterministicMixin, Model):
    def __init__(self, values):
        Model.__init__(self, device="cpu")
        DeterministicMixin.__init__(self)
        self.parameter = nn.Parameter(wp.array(values, device="cpu"), requires_grad=True)
        self.__post_init__()

    def compute(self, inputs, role=""):
        return inputs["observations"], {}


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize("shape", [(8,), (2, 4), (2, 2, 2), (1, 2, 2, 2)])
@pytest.mark.parametrize("polyak", [0.0, 0.25, 1.0])
def test_soft_updates_preserve_parameter_shape_dtype_and_storage(dtype, shape, polyak):
    target_values = np.arange(8, dtype=dtype).reshape(shape)
    source_values = np.arange(8, dtype=dtype).reshape(shape) + 2
    target = ParameterModel(target_values)
    source = ParameterModel(source_values)
    target_parameters = list(target.parameters())
    assert len(target_parameters) == 1
    pointers = [parameter.ptr for parameter in target_parameters]
    target.update_parameters(source, polyak=polyak)
    assert [parameter.ptr for parameter in target.parameters()] == pointers
    for parameter in target.parameters():
        assert parameter.shape == shape
        assert parameter.requires_grad
        np.testing.assert_allclose(
            parameter.numpy(), (1 - polyak) * target_values + polyak * source_values, rtol=1e-6, atol=1e-6
        )
    for parameter in source.parameters():
        np.testing.assert_array_equal(parameter.numpy(), source_values)


@pytest.mark.parametrize("polyak", [0.0, 0.25, 0.75, 1.0])
def test_soft_updates_include_gaussian_log_standard_deviation(polyak):
    kwargs = {
        "observation_space": spaces.Box(-1, 1, shape=(3,)),
        "action_space": spaces.Box(-1, 1, shape=(2,)),
        "device": "cpu",
        "network": [{"name": "net", "input": "OBSERVATIONS", "layers": [4], "activations": "tanh"}],
        "output": "ACTIONS",
    }
    target = gaussian_model(**kwargs)
    source = gaussian_model(**kwargs)
    for model in [target, source]:
        model.init_state_dict()
    for parameter in target.parameters():
        parameter.fill_(0.5)
    for parameter in source.parameters():
        parameter.fill_(2.5)
    assert any(parameter.ndim == 1 for parameter in target.parameters())
    pointers = [parameter.ptr for parameter in target.parameters()]
    target.update_parameters(source, polyak=polyak)
    assert [parameter.ptr for parameter in target.parameters()] == pointers
    for parameter in target.parameters():
        np.testing.assert_allclose(parameter.numpy(), 0.5 + 2 * polyak)
    for parameter in source.parameters():
        np.testing.assert_array_equal(parameter.numpy(), 2.5)
