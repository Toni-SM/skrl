import unittest
from fractions import Fraction

import numpy as np
import warp as wp

from skrl.resources.noises.warp import OrnsteinUhlenbeckNoise


INITIAL = np.array([[-3.0, 2.0], [1.0, -4.0], [0.5, 8.0]], dtype=np.float32)


def as_backend(array: np.ndarray) -> wp.array:
    return wp.array(array, dtype=wp.float32, device="cpu")


def as_numpy(array: wp.array) -> np.ndarray:
    return array.numpy().copy()


class TestMeanReversion(unittest.TestCase):
    def make_noise(self, theta: float, base_scale: float = 1.0) -> OrnsteinUhlenbeckNoise:
        """Sigma zero isolates the drift from the independently incorrect innovation scale."""
        noise = OrnsteinUhlenbeckNoise(theta=theta, sigma=0.0, base_scale=base_scale, device="cpu")
        noise.state = as_backend(INITIAL)
        return noise

    def test_zero_reversion_preserves_existing_state(self) -> None:
        """Zero mean-reversion rate preserves state when the innovation is zero."""
        noise = self.make_noise(0.0)
        for _ in range(6):
            np.testing.assert_array_equal(as_numpy(noise.sample(INITIAL.shape)), INITIAL)
            np.testing.assert_array_equal(as_numpy(noise.state), INITIAL)

    def test_unit_reversion_removes_existing_state(self) -> None:
        """A unit drift step removes the full old state without innovation."""
        noise = self.make_noise(1.0)
        for _ in range(4):
            np.testing.assert_array_equal(as_numpy(noise.sample(INITIAL.shape)), np.zeros_like(INITIAL))
            np.testing.assert_array_equal(as_numpy(noise.state), np.zeros_like(INITIAL))

    def test_decay_matches_independent_exact_linear_flow(self) -> None:
        """The unit Euler step has exact discrete solution x[n]=(1-theta)**n*x[0]."""
        for theta in (Fraction(0), Fraction(3, 20), Fraction(1, 2), Fraction(4, 5), Fraction(1), Fraction(5, 4)):
            for base_scale in (0.0, 0.25, 2.0):
                with self.subTest(theta=theta, base_scale=base_scale):
                    noise = self.make_noise(float(theta), base_scale)
                    for step in range(1, 8):
                        sampled = as_numpy(noise.sample(INITIAL.shape))
                        expected_state = INITIAL.astype(np.float64) * float((1 - theta) ** step)
                        np.testing.assert_allclose(as_numpy(noise.state), expected_state, rtol=3e-6, atol=3e-6)
                        np.testing.assert_allclose(sampled, base_scale * expected_state, rtol=3e-6, atol=3e-6)

    def test_output_scale_does_not_rescale_internal_dynamics(self) -> None:
        """The returned action scale does not change the recurrence's internal state."""
        unit = self.make_noise(0.15, 1.0)
        scaled = self.make_noise(0.15, 3.0)
        for step in range(1, 6):
            unit_sample = as_numpy(unit.sample(INITIAL.shape))
            scaled_sample = as_numpy(scaled.sample(INITIAL.shape))
            expected_state = INITIAL.astype(np.float64) * float(Fraction(17, 20) ** step)
            np.testing.assert_allclose(unit_sample, expected_state, rtol=3e-6, atol=3e-6)
            np.testing.assert_allclose(as_numpy(unit.state), as_numpy(scaled.state), rtol=0, atol=0)
            np.testing.assert_allclose(scaled_sample, 3.0 * expected_state, rtol=3e-6, atol=3e-6)

    def test_shape_change_keeps_the_existing_zero_reset_contract(self) -> None:
        """Shape changes discard old state; zero innovation preserves zero."""
        noise = self.make_noise(0.15)
        for size in ((4, 1), (2, 3), (4, 1)):
            sampled = as_numpy(noise.sample(size))
            self.assertEqual(sampled.shape, size)
            np.testing.assert_array_equal(sampled, np.zeros(size, dtype=np.float32))
            np.testing.assert_array_equal(as_numpy(noise.state), np.zeros(size, dtype=np.float32))

    def test_real_stochastic_samples_have_the_correct_conditional_slope(self) -> None:
        """Independent innovations imply E[x2|x1]=(1-theta)*x1, regardless of their scale."""
        noise = OrnsteinUhlenbeckNoise(theta=0.15, sigma=0.4, base_scale=1.0, mean=0.0, std=1.0, device="cpu")
        previous = as_numpy(noise.sample((16384, 1))).astype(np.float64).copy()
        following = as_numpy(noise.sample((16384, 1))).astype(np.float64).copy()
        denominator = float(np.sum(previous * previous))
        self.assertGreater(denominator, 0.0)
        slope = float(np.sum(previous * following)) / denominator
        self.assertLess(abs(slope - 0.85), 0.08)


if __name__ == "__main__":
    unittest.main()
