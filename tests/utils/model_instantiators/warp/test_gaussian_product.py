import pytest

from gymnasium import spaces

import numpy as np
import warp as wp

from skrl.models.warp import GaussianMixin, Model
from skrl.utils.framework.warp import mean


class IdentityGaussian(GaussianMixin, Model):
    def __init__(self, reduction, log_std):
        size = len(log_std)
        Model.__init__(
            self,
            observation_space=spaces.Box(low=-1, high=1, shape=(size,)),
            action_space=spaces.Box(low=-1, high=1, shape=(size,)),
            device="cpu",
        )
        GaussianMixin.__init__(self, reduction=reduction, clip_log_std=False)
        self.log_std = wp.array(log_std, dtype=wp.float32, device="cpu", requires_grad=True)

    def compute(self, inputs, role=""):
        return inputs["observations"], {"log_std": self.log_std}


def check_reduction(reduction, num_actions, taken, log_std, offset=0.3):
    model = IdentityGaussian(reduction, log_std)
    locations = np.arange(2 * num_actions, dtype=np.float32).reshape(2, num_actions) / 10 - 0.2
    observations = wp.array(locations, device="cpu", requires_grad=True)
    inputs = {"observations": observations}
    if taken:
        inputs["taken_actions"] = wp.array(locations + offset, device="cpu")
    with wp.Tape() as tape:
        actions, outputs = model.act(inputs)
        loss = mean(outputs["log_prob"])
    values = inputs["taken_actions"].numpy() if taken else actions.numpy()
    scale = outputs["stddev"].numpy()
    components = -0.5 * ((values - locations) / scale) ** 2 - np.log(scale) - 0.5 * np.log(2 * np.pi)
    expected = {
        "prod": lambda: np.prod(components, axis=-1, keepdims=True),
        "sum": lambda: np.sum(components, axis=-1, keepdims=True),
        "mean": lambda: np.mean(components, axis=-1, keepdims=True),
        "none": lambda: components,
    }[reduction]()
    np.testing.assert_allclose(outputs["log_prob"].numpy(), expected, rtol=1e-5, atol=1e-6)
    if reduction == "prod" and taken:
        tape.backward(loss)
        other_factors = np.stack(
            [np.prod(np.delete(components, j, axis=1), axis=1) for j in range(num_actions)], axis=1
        )
        expected_mean_grad = other_factors * (values - locations) / scale**2 / len(locations)
        expected_log_std_grad = np.mean(other_factors * (((values - locations) / scale) ** 2 - 1), axis=0)
        np.testing.assert_allclose(observations.grad.numpy(), expected_mean_grad, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(model.log_std.grad.numpy(), expected_log_std_grad, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("reduction", ["prod", "sum", "mean", "none"])
@pytest.mark.parametrize("num_actions", [1, 3, 4])
@pytest.mark.parametrize("taken", [False, True])
def test_gaussian_log_probability_reductions(reduction, num_actions, taken):
    check_reduction(reduction, num_actions, taken, np.linspace(-0.4, 0.2, num_actions, dtype=np.float32))


def test_product_gradients_near_a_zero_log_density():
    check_reduction(
        "prod",
        3,
        True,
        np.array([-0.5 * np.log(2 * np.pi), -0.8, 0.2], dtype=np.float32),
        offset=np.array([0.0, 0.3, 0.3], dtype=np.float32),
    )
