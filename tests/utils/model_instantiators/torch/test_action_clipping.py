from __future__ import annotations

from typing import Callable

import pytest

from gymnasium import spaces

import torch

from skrl.utils.model_instantiators.torch import deterministic_model, gaussian_model, multivariate_gaussian_model


@pytest.mark.parametrize("factory", [deterministic_model, gaussian_model, multivariate_gaussian_model])
@pytest.mark.parametrize("bounds", [(-float("inf"), float("inf")), (-1.0, 1.0), (0.0, float("inf"))])
def test_action_clipping_with_bounded_and_unbounded_spaces(factory: Callable, bounds: tuple[float, float]) -> None:
    kwargs = {} if factory is deterministic_model else {"clip_mean_actions": True}
    model = factory(
        observation_space=spaces.Box(-10, 10, shape=(2,)),
        action_space=spaces.Box(*bounds, shape=(2,)),
        device="cpu",
        clip_actions=True,
        network=[{"name": "net", "input": "OBSERVATIONS", "layers": [4], "activations": "relu"}],
        output="ACTIONS",
        **kwargs,
    )
    observations = torch.tensor([[1.0, 2.0], [-3.0, -4.0]], requires_grad=True)
    inputs = {"observations": observations}
    model.init_state_dict(inputs)
    model.init_weights("constant_", val=1.0)
    model.init_biases("constant_", val=0.0)
    raw_mean, _ = model.compute(inputs, "")

    actions, outputs = model.act(inputs)

    assert actions.shape == (2, 2)
    assert torch.isfinite(actions).all()
    assert torch.all(actions >= bounds[0])
    assert torch.all(actions <= bounds[1])
    expected_mean = raw_mean.clamp(min=bounds[0], max=bounds[1])
    torch.testing.assert_close(actions if factory is deterministic_model else outputs["mean_actions"], expected_mean)
    actions.sum().backward()
    assert torch.isfinite(observations.grad).all()
