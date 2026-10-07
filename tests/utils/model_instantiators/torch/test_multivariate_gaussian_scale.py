"""Check multivariate policy scales against independent scalar Gaussian formulas."""

from typing import Any

import math
import unittest
import gymnasium

import numpy as np
import torch

from skrl.models.torch import Model, MultivariateGaussianMixin


class DiagonalPolicy(MultivariateGaussianMixin, Model):
    def __init__(self, log_std: torch.Tensor, clip_log_std: bool = False) -> None:
        Model.__init__(
            self,
            observation_space=gymnasium.spaces.Box(-1.0, 1.0, shape=(2,)),
            action_space=gymnasium.spaces.Box(-np.inf, np.inf, shape=(2,)),
            device="cpu",
        )
        MultivariateGaussianMixin.__init__(self, clip_log_std=clip_log_std, min_log_std=-0.5, max_log_std=0.5)
        self.mean = torch.nn.Parameter(torch.tensor([0.2, -0.3], dtype=torch.float64))
        self.log_std = torch.nn.Parameter(log_std.clone().to(torch.float64))

    def compute(self, inputs: dict[str, Any], role: str) -> tuple[torch.Tensor, dict[str, Any]]:
        return self.mean.expand(inputs["observations"].shape[0], -1), {"log_std": self.log_std}


class MultivariateScaleTest(unittest.TestCase):
    def inputs(self, batch: int, taken_actions: torch.Tensor | None = None) -> dict[str, Any]:
        inputs = {"observations": torch.zeros(batch, 2, dtype=torch.float64)}
        if taken_actions is not None:
            inputs["taken_actions"] = taken_actions
        return inputs

    def test_nonunit_scales_give_intended_covariance_and_density(self) -> None:
        policy = DiagonalPolicy(torch.tensor([math.log(0.5), math.log(3.0)], dtype=torch.float64))
        taken = torch.tensor([[0.7, 0.7], [-0.3, -0.8]], dtype=torch.float64)
        _, outputs = policy.act(self.inputs(2, taken))
        expected_variance = torch.tensor([0.25, 9.0], dtype=torch.float64).expand(2, -1)
        torch.testing.assert_close(policy.distribution().variance, expected_variance, rtol=1e-12, atol=1e-12)
        expected = (
            -0.5 * ((taken - policy.mean) / policy.log_std.exp()).square()
            - policy.log_std
            - 0.5 * math.log(2 * math.pi)
        ).sum(-1, keepdim=True)
        torch.testing.assert_close(outputs["log_prob"], expected, rtol=1e-12, atol=1e-12)
        entropy = (policy.log_std + 0.5 * math.log(2 * math.pi * math.e)).sum().expand(2, 1)
        torch.testing.assert_close(policy.get_entropy(), entropy, rtol=1e-12, atol=1e-12)

    def test_state_dependent_scales_preserve_each_batch_covariance(self) -> None:
        log_std = torch.tensor([[math.log(0.5), math.log(2.0)], [math.log(3.0), math.log(0.25)]], dtype=torch.float64)
        policy = DiagonalPolicy(log_std)
        taken = torch.tensor([[0.7, 0.7], [-0.3, -0.8]], dtype=torch.float64)
        actions, outputs = policy.act(self.inputs(2, taken))
        self.assertEqual(tuple(actions.shape), (2, 2))
        self.assertEqual(tuple(outputs["log_prob"].shape), (2, 1))
        torch.testing.assert_close(policy.distribution().variance, log_std.mul(2).exp(), rtol=1e-12, atol=1e-12)
        residual = taken - policy.mean.detach()
        variance = log_std.mul(2).exp()
        expected = (-0.5 * residual.square() / variance - log_std - 0.5 * math.log(2 * math.pi)).sum(-1, keepdim=True)
        torch.testing.assert_close(outputs["log_prob"], expected, rtol=1e-12, atol=1e-12)
        outputs["log_prob"].sum().backward()
        torch.testing.assert_close(policy.mean.grad, (residual / variance).sum(0), rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(policy.log_std.grad, residual.square() / variance - 1, rtol=1e-12, atol=1e-12)

    def test_rsample_uses_standard_deviation_and_has_correct_path_gradient(self) -> None:
        policy = DiagonalPolicy(torch.tensor([math.log(0.5), math.log(3.0)], dtype=torch.float64))
        torch.manual_seed(17)
        expected_noise = torch.randn(3, 2, dtype=torch.float64)
        torch.manual_seed(17)
        actions, _ = policy.act(self.inputs(3))
        expected = policy.mean + expected_noise * policy.log_std.exp()
        torch.testing.assert_close(actions, expected, rtol=1e-12, atol=1e-12)
        actions.sum().backward()
        torch.testing.assert_close(policy.mean.grad, torch.full((2,), 3.0, dtype=torch.float64), rtol=0, atol=0)
        torch.testing.assert_close(
            policy.log_std.grad, (expected_noise * policy.log_std.exp()).sum(0), rtol=1e-12, atol=1e-12
        )

    def test_log_density_parameter_gradients_match_scalar_normals(self) -> None:
        policy = DiagonalPolicy(torch.tensor([math.log(0.5), math.log(3.0)], dtype=torch.float64))
        taken = torch.tensor([[0.7, 0.7], [-0.3, -0.8]], dtype=torch.float64)
        _, outputs = policy.act(self.inputs(2, taken))
        outputs["log_prob"].sum().backward()
        residual = taken - policy.mean.detach()
        variance = policy.log_std.detach().mul(2).exp()
        torch.testing.assert_close(policy.mean.grad, (residual / variance).sum(0), rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(
            policy.log_std.grad, (residual.square() / variance - 1).sum(0), rtol=1e-12, atol=1e-12
        )

    def test_log_std_clipping_and_unit_scale_control(self) -> None:
        for log_std, clip in (([-1.0, 1.0], True), ([0.0, 0.0], False)):
            with self.subTest(log_std=log_std, clip=clip):
                policy = DiagonalPolicy(torch.tensor(log_std, dtype=torch.float64), clip_log_std=clip)
                _, outputs = policy.act(self.inputs(2))
                expected_std = (
                    torch.tensor(log_std, dtype=torch.float64).clamp(-0.5, 0.5)
                    if clip
                    else torch.zeros(2, dtype=torch.float64)
                )
                torch.testing.assert_close(outputs["log_std"], expected_std, rtol=0, atol=0)
                torch.testing.assert_close(
                    policy.distribution().variance, expected_std.mul(2).exp().expand(2, -1), rtol=1e-12, atol=1e-12
                )

    def test_entropy_and_gradients_follow_clipped_standard_deviations(self) -> None:
        batch = 3
        shared = [math.log(0.5), math.log(3.0)]
        examples = (
            (torch.tensor(shared, dtype=torch.float64), False),
            (torch.tensor([shared], dtype=torch.float64), False),
            (
                torch.tensor(
                    [shared, [math.log(2.0), math.log(4.0)], [math.log(0.75), math.log(0.25)]], dtype=torch.float64
                ),
                False,
            ),
            (torch.tensor([0.2, 1.0], dtype=torch.float64), True),
            (torch.tensor([-1.0, -0.2], dtype=torch.float64), True),
            (torch.zeros(2, dtype=torch.float64), False),
        )
        for log_std, clip in examples:
            with self.subTest(shape=tuple(log_std.shape), values=log_std.tolist(), clip=clip):
                policy = DiagonalPolicy(log_std, clip_log_std=clip)
                policy.act(self.inputs(batch))
                clipped = log_std.clamp(-0.5, 0.5) if clip else log_std
                expected_entropy = (clipped + 0.5 * math.log(2 * math.pi * math.e)).sum(-1).expand(batch).unsqueeze(-1)
                torch.testing.assert_close(policy.get_entropy(), expected_entropy, rtol=1e-12, atol=1e-12)
                policy.get_entropy().sum().backward()
                expected_gradient = torch.ones_like(log_std)
                if log_std.ndim == 1 or log_std.shape[0] == 1:
                    expected_gradient *= batch
                if clip:
                    expected_gradient *= ((log_std >= -0.5) & (log_std <= 0.5)).to(log_std.dtype)
                torch.testing.assert_close(policy.log_std.grad, expected_gradient, rtol=1e-12, atol=1e-12)

    def test_shared_row_scales_broadcast_density_and_parameter_gradients(self) -> None:
        log_std = torch.tensor([[math.log(0.5), math.log(3.0)]], dtype=torch.float64)
        policy = DiagonalPolicy(log_std)
        taken = torch.tensor([[0.7, 0.7], [-0.3, -0.8], [1.2, 0.2]], dtype=torch.float64)
        actions, outputs = policy.act(self.inputs(3, taken))
        self.assertEqual(tuple(actions.shape), (3, 2))
        self.assertEqual(tuple(outputs["log_prob"].shape), (3, 1))
        variance = log_std.mul(2).exp()
        torch.testing.assert_close(policy.distribution().variance, variance.expand(3, -1), rtol=1e-12, atol=1e-12)
        residual = taken - policy.mean.detach()
        expected = (-0.5 * residual.square() / variance - log_std - 0.5 * math.log(2 * math.pi)).sum(-1, keepdim=True)
        torch.testing.assert_close(outputs["log_prob"], expected, rtol=1e-12, atol=1e-12)
        outputs["log_prob"].sum().backward()
        torch.testing.assert_close(policy.mean.grad, (residual / variance).sum(0), rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(
            policy.log_std.grad, (residual.square() / variance - 1).sum(0, keepdim=True), rtol=1e-12, atol=1e-12
        )

    def test_kl_and_parameter_gradients_agree_with_scalar_gaussians(self) -> None:
        batch = 3
        first = DiagonalPolicy(torch.tensor([math.log(0.5), math.log(3.0)], dtype=torch.float64))
        second = DiagonalPolicy(torch.tensor([math.log(1.5), math.log(0.75)], dtype=torch.float64))
        with torch.no_grad():
            second.mean.copy_(torch.tensor([-0.4, 0.8], dtype=torch.float64))
        first.act(self.inputs(batch))
        second.act(self.inputs(batch))
        actual = torch.distributions.kl_divergence(first.distribution(), second.distribution())
        delta = second.mean.detach() - first.mean.detach()
        first_variance = first.log_std.detach().mul(2).exp()
        second_variance = second.log_std.detach().mul(2).exp()
        expected = (
            (
                second.log_std.detach()
                - first.log_std.detach()
                + 0.5 * ((first_variance + delta.square()) / second_variance - 1)
            )
            .sum()
            .expand(batch)
        )
        self.assertEqual(tuple(actual.shape), (batch,))
        self.assertTrue(bool(torch.all(actual > 0)))
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        actual.sum().backward()
        torch.testing.assert_close(first.mean.grad, -batch * delta / second_variance, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(second.mean.grad, batch * delta / second_variance, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(
            first.log_std.grad, batch * (first_variance / second_variance - 1), rtol=1e-12, atol=1e-12
        )
        torch.testing.assert_close(
            second.log_std.grad,
            batch * (1 - (first_variance + delta.square()) / second_variance),
            rtol=1e-12,
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
