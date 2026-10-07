"""Check both TRPO KL helpers against scalar density expectations and derivatives."""

from typing import Any

import math
import unittest
from types import ModuleType
import gymnasium

import numpy as np
import torch

from skrl.agents.torch.trpo import trpo, trpo_rnn
from skrl.models.torch import GaussianMixin, Model


class GaussianPolicy(GaussianMixin, Model):
    def __init__(self, mean: torch.Tensor, log_std: torch.Tensor) -> None:
        Model.__init__(
            self,
            observation_space=gymnasium.spaces.Box(-1.0, 1.0, shape=(2,)),
            action_space=gymnasium.spaces.Box(-np.inf, np.inf, shape=(2,)),
            device="cpu",
        )
        GaussianMixin.__init__(self, clip_actions=False, clip_mean_actions=False, clip_log_std=False)
        self.mean = torch.nn.Parameter(mean.clone().to(torch.float64))
        self.log_std = torch.nn.Parameter(log_std.clone().to(torch.float64))

    def compute(self, inputs: dict[str, Any], role: str) -> tuple[torch.Tensor, dict[str, Any]]:
        return self.mean.expand(inputs["observations"].shape[0], -1), {"log_std": self.log_std}


class RecurrentGaussianPolicy(GaussianPolicy):
    def __init__(self, mean: torch.Tensor, log_std: torch.Tensor) -> None:
        super().__init__(mean, log_std)
        self.recurrent = torch.nn.GRU(2, 2, batch_first=True, dtype=torch.float64)
        with torch.no_grad():
            for index, parameter in enumerate(self.recurrent.parameters()):
                parameter.fill_(0.05 * (index + 1))

    def compute(self, inputs: dict[str, Any], role: str) -> tuple[torch.Tensor, dict[str, Any]]:
        recurrent, _ = self.recurrent(inputs["observations"].unsqueeze(1), inputs["rnn_states"][0])
        return self.mean + recurrent.squeeze(1), {"log_std": self.log_std}


def density_expectation(
    first_mean: torch.Tensor,
    first_log_std: torch.Tensor,
    second_mean: torch.Tensor,
    second_log_std: torch.Tensor,
) -> float:
    """Integrate log p - log q under p with Gauss-Hermite quadrature."""
    nodes, weights = np.polynomial.hermite.hermgauss(32)
    mu_1, s_1, mu_2, s_2 = [
        value.detach().numpy() for value in (first_mean, first_log_std, second_mean, second_log_std)
    ]
    points = mu_1[:, None, :] + math.sqrt(2) * np.exp(s_1)[:, None, :] * nodes[None, :, None]
    log_p = (
        -0.5 * ((points - mu_1[:, None, :]) / np.exp(s_1)[:, None, :]) ** 2
        - s_1[:, None, :]
        - 0.5 * math.log(2 * math.pi)
    )
    log_q = (
        -0.5 * ((points - mu_2[:, None, :]) / np.exp(s_2)[:, None, :]) ** 2
        - s_2[:, None, :]
        - 0.5 * math.log(2 * math.pi)
    )
    return float((((log_p - log_q).sum(-1) * weights).sum(-1) / math.sqrt(math.pi)).mean())


class TRPOKLTest(unittest.TestCase):
    def call(
        self,
        module: ModuleType,
        first: GaussianPolicy,
        second: GaussianPolicy,
        observations: torch.Tensor,
        rnn: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        arguments = {"policy_1": first, "policy_2": second, "observations": observations, "states": observations}
        if module is trpo_rnn:
            arguments["rnn"] = {} if rnn is None else rnn
        return module.kl_divergence(**arguments)

    def policies(
        self,
        mean_1: list[float],
        scales_1: list[float],
        mean_2: list[float],
        scales_2: list[float] | list[list[float]],
    ) -> tuple[GaussianPolicy, GaussianPolicy]:
        first = GaussianPolicy(
            torch.tensor(mean_1, dtype=torch.float64), torch.tensor(scales_1, dtype=torch.float64).log()
        )
        second = GaussianPolicy(
            torch.tensor(mean_2, dtype=torch.float64), torch.tensor(scales_2, dtype=torch.float64).log()
        )
        return first, second

    def test_directional_kl_matches_independent_density_integrals(self) -> None:
        examples = (
            ([0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [2.0, 2.0]),
            ([0.0, 0.0], [2.0, 2.0], [0.0, 0.0], [1.0, 1.0]),
            ([0.2, -0.3], [0.5, 3.0], [-0.4, 0.8], [1.5, 0.75]),
            ([0.2, -0.3], [0.5, 3.0], [-0.4, 0.8], [0.5, 3.0]),
        )
        observations = torch.zeros(3, 2, dtype=torch.float64)
        for module in (trpo, trpo_rnn):
            for example in examples:
                with self.subTest(module=module.__name__, example=example):
                    first, second = self.policies(*example)
                    actual = self.call(module, first, second, observations)
                    mean_1, s_1 = first.mean.expand(3, -1), first.log_std.expand(3, -1)
                    mean_2, s_2 = second.mean.expand(3, -1), second.log_std.expand(3, -1)
                    expected = density_expectation(mean_1, s_1, mean_2, s_2)
                    self.assertGreaterEqual(actual.item(), 0)
                    torch.testing.assert_close(
                        actual, torch.tensor(expected, dtype=torch.float64), rtol=2e-12, atol=2e-12
                    )
                    native = (
                        torch.distributions.kl_divergence(
                            torch.distributions.Normal(mean_1, s_1.exp()), torch.distributions.Normal(mean_2, s_2.exp())
                        )
                        .sum(-1)
                        .mean()
                    )
                    torch.testing.assert_close(actual, native, rtol=2e-12, atol=2e-12)

    def test_reference_is_detached_and_candidate_gradients_match_scalar_normals(self) -> None:
        for module in (trpo, trpo_rnn):
            for state_dependent in (False, True):
                with self.subTest(module=module.__name__, state_dependent=state_dependent):
                    scales_2 = [[1.5, 0.75], [0.25, 2.0], [1.0, 0.5]] if state_dependent else [1.5, 0.75]
                    first, second = self.policies([0.2, -0.3], [0.5, 3.0], [-0.4, 0.8], scales_2)
                    actual = self.call(module, first, second, torch.zeros(3, 2, dtype=torch.float64))
                    actual.backward()
                    self.assertIsNone(first.mean.grad)
                    self.assertIsNone(first.log_std.grad)
                    delta = second.mean.detach() - first.mean.detach()
                    first_variance = first.log_std.detach().mul(2).exp()
                    second_variance = second.log_std.detach().mul(2).exp()
                    expected_mean = (delta / second_variance).mean(0) if state_dependent else delta / second_variance
                    expected_scale = 1 - (first_variance + delta.square()) / second_variance
                    if state_dependent:
                        expected_scale /= 3
                    torch.testing.assert_close(second.mean.grad, expected_mean, rtol=2e-12, atol=2e-12)
                    torch.testing.assert_close(second.log_std.grad, expected_scale, rtol=2e-12, atol=2e-12)

    def test_identical_policies_have_zero_kl_and_stationary_scale_gradients(self) -> None:
        for module in (trpo, trpo_rnn):
            with self.subTest(module=module.__name__):
                first, second = self.policies([0.2, -0.3], [0.5, 3.0], [0.2, -0.3], [0.5, 3.0])
                actual = self.call(module, first, second, torch.zeros(3, 2, dtype=torch.float64))
                torch.testing.assert_close(actual, torch.zeros((), dtype=torch.float64), rtol=0, atol=1e-14)
                actual.backward()
                torch.testing.assert_close(second.mean.grad, torch.zeros(2, dtype=torch.float64), rtol=0, atol=1e-14)
                torch.testing.assert_close(second.log_std.grad, torch.zeros(2, dtype=torch.float64), rtol=0, atol=1e-14)

    def test_kl_hessian_preserves_the_diagonal_fisher_metric(self) -> None:
        for module in (trpo, trpo_rnn):
            with self.subTest(module=module.__name__):
                first, second = self.policies([0.2, -0.3], [0.5, 3.0], [0.2, -0.3], [0.5, 3.0])
                actual = self.call(module, first, second, torch.zeros(3, 2, dtype=torch.float64))
                gradients = torch.autograd.grad(actual, (second.mean, second.log_std), create_graph=True)
                mean_direction = torch.tensor([0.7, -0.2], dtype=torch.float64)
                scale_direction = torch.tensor([-0.3, 0.9], dtype=torch.float64)
                directional_gradient = (gradients[0] * mean_direction).sum() + (gradients[1] * scale_direction).sum()
                actual_hessian = torch.autograd.grad(directional_gradient, (second.mean, second.log_std))
                torch.testing.assert_close(
                    actual_hessian[0], mean_direction / second.log_std.detach().mul(2).exp(), rtol=2e-12, atol=2e-12
                )
                torch.testing.assert_close(actual_hessian[1], 2 * scale_direction, rtol=2e-12, atol=2e-12)

    def test_scale_only_candidate_exceeds_the_trust_region_budget(self) -> None:
        for module in (trpo, trpo_rnn):
            with self.subTest(module=module.__name__):
                first, second = self.policies([0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [2.0, 2.0])
                actual = self.call(module, first, second, torch.zeros(3, 2, dtype=torch.float64))
                expected = torch.tensor(2 * (math.log(2) - 0.375), dtype=torch.float64)
                torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)
                self.assertGreater(actual.item(), 0.1)

    def test_recurrent_helper_uses_real_gru_hidden_states_in_density_expectation(self) -> None:
        observations = torch.tensor([[0.1, 0.2], [0.3, -0.4], [0.5, 0.6]], dtype=torch.float64)
        rnn = {"rnn_states": [torch.full((1, 3, 2), 0.3, dtype=torch.float64)]}
        first = RecurrentGaussianPolicy(torch.tensor([0.2, -0.3]), torch.tensor([math.log(0.5), math.log(3.0)]))
        second = RecurrentGaussianPolicy(torch.tensor([-0.4, 0.8]), torch.tensor([math.log(1.5), math.log(0.75)]))
        with torch.no_grad():
            for index, parameter in enumerate(second.recurrent.parameters()):
                parameter.fill_(0.09 * (index + 1))
        actual = self.call(trpo_rnn, first, second, observations, rnn)
        inputs = {"observations": observations, "states": observations, **rnn}
        _, first_outputs = first.act(inputs, role="policy")
        _, second_outputs = second.act(inputs, role="policy")
        expected = density_expectation(
            first_outputs["mean_actions"],
            first_outputs["log_std"].expand(3, -1),
            second_outputs["mean_actions"],
            second_outputs["log_std"].expand(3, -1),
        )
        torch.testing.assert_close(actual, torch.tensor(expected, dtype=torch.float64), rtol=2e-12, atol=2e-12)
        zero_rnn = {"rnn_states": [torch.zeros(1, 3, 2, dtype=torch.float64)]}
        zero_kl = self.call(trpo_rnn, first, second, observations, zero_rnn)
        self.assertFalse(torch.allclose(actual, zero_kl))
        _, zero_outputs = first.act({"observations": observations, **zero_rnn}, role="policy")
        self.assertFalse(torch.allclose(first_outputs["mean_actions"], zero_outputs["mean_actions"]))


if __name__ == "__main__":
    unittest.main()
