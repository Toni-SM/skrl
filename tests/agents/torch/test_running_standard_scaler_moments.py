import unittest
from fractions import Fraction

import numpy as np
import torch

from skrl.resources.preprocessors.torch import RunningStandardScaler


VALUES = ((-3, 6), (1, 1), (4, -1), (0, 7), (2, 4), (-2, 1))


def raw_moments(rows: tuple[tuple[int, ...], ...]) -> tuple[int, np.ndarray, np.ndarray]:
    """Compute E[x] and E[x**2] with the scaler's existing unit prior."""
    count = len(rows) + 1
    sums = [sum(row[j] for row in rows) for j in range(len(rows[0]))]
    squares = [1 + sum(row[j] ** 2 for row in rows) for j in range(len(rows[0]))]
    means = [Fraction(value, count) for value in sums]
    variances = [Fraction(second, count) - mean**2 for second, mean in zip(squares, means)]
    return count, np.array([float(x) for x in means]), np.array([float(x) for x in variances])


def to_backend(array: np.ndarray) -> torch.Tensor:
    """Transfer test observations through the genuine CPU tensor API."""
    return torch.tensor(array, dtype=torch.float32, device="cpu")


def as_numpy(array: torch.Tensor) -> np.ndarray:
    """Read actual public tensor values."""
    return array.detach().cpu().numpy()


def read_count(scaler: RunningStandardScaler) -> float:
    return scaler.current_count.item()


def restore(destination: RunningStandardScaler, source: RunningStandardScaler) -> None:
    """Use the standard PyTorch checkpoint API with independent tensor storage."""
    destination.load_state_dict({key: value.clone() for key, value in source.state_dict().items()})


class TestPopulationMoments(unittest.TestCase):
    def check_statistics(self, scaler: RunningStandardScaler, rows: tuple[tuple[int, ...], ...]) -> None:
        """Compare with independent exact raw moments, never a variance API."""
        count, mean, variance = raw_moments(rows)
        self.assertEqual(read_count(scaler), count)
        np.testing.assert_allclose(as_numpy(scaler.running_mean), mean, rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(as_numpy(scaler.running_variance), variance, rtol=2e-6, atol=2e-6)

    def test_every_partition_recovers_raw_moments(self) -> None:
        """Changing rollout boundaries must not change stream statistics."""
        for temporal in (False, True):
            for partition in ((6,), (1, 2, 3), (1, 1, 1, 1, 1, 1), (2, 4), (3, 3)):
                with self.subTest(temporal=temporal, partition=partition):
                    scaler = RunningStandardScaler(2, clip_threshold=100.0, device="cpu")
                    offset = 0
                    for size in partition:
                        rows = VALUES[offset : offset + size]
                        array = np.array(rows, dtype=np.float32)
                        if temporal:
                            array = array.reshape(1, size, 2)
                        result = as_numpy(scaler(to_backend(array), train=True))
                        offset += size
                        self.check_statistics(scaler, VALUES[:offset])
                        self.assertTrue(np.isfinite(result).all())
                        _, mean, variance = raw_moments(VALUES[:offset])
                        expected = (array - mean.astype(np.float32)) / (
                            np.sqrt(variance.astype(np.float32)) + scaler.epsilon
                        )
                        np.testing.assert_allclose(result, expected, rtol=4e-6, atol=4e-6)

    def test_singleton_updates_remain_finite(self) -> None:
        """One observation has zero within-batch population central moment."""
        rows = ((-4,), (-4,), (2,), (3,), (0,))
        for temporal in (False, True):
            with self.subTest(temporal=temporal):
                scaler = RunningStandardScaler(1, device="cpu")
                for index, row in enumerate(rows):
                    array = np.array([row], dtype=np.float32)
                    if temporal:
                        array = array.reshape(1, 1, 1)
                    result = as_numpy(scaler(to_backend(array), train=True))
                    self.assertTrue(np.isfinite(result).all())
                    self.check_statistics(scaler, rows[: index + 1])

    def test_temporal_axes_match_flattened_observations(self) -> None:
        """Both rollout axes represent observations in the same empirical law."""
        for shape in ((6, 2), (1, 6, 2), (2, 3, 2), (3, 2, 2)):
            with self.subTest(shape=shape):
                scaler = RunningStandardScaler(2, device="cpu")
                scaler(to_backend(np.array(VALUES, dtype=np.float32).reshape(shape)), train=True)
                self.check_statistics(scaler, VALUES)

    def test_checkpoint_resume_preserves_stream_moments(self) -> None:
        """Restored moments merge subsequent observations as one uninterrupted stream."""
        original = RunningStandardScaler(2, device="cpu")
        original(to_backend(np.array(VALUES[:2], dtype=np.float32)), train=True)
        resumed = RunningStandardScaler(2, device="cpu")
        restore(resumed, original)
        for row in VALUES[2:]:
            batch = to_backend(np.array([row], dtype=np.float32))
            original(batch, train=True)
            resumed(batch, train=True)
        self.check_statistics(original, VALUES)
        self.check_statistics(resumed, VALUES)
        np.testing.assert_array_equal(as_numpy(resumed.running_mean), as_numpy(original.running_mean))
        np.testing.assert_array_equal(as_numpy(resumed.running_variance), as_numpy(original.running_variance))

    def test_evaluation_inverse_and_clipping_preserve_state(self) -> None:
        """Inference transforms use fixed moments and preserve the existing clip contract."""
        scaler = RunningStandardScaler(2, clip_threshold=2.0, device="cpu")
        scaler(to_backend(np.array(VALUES, dtype=np.float32)), train=True)
        before = (read_count(scaler), as_numpy(scaler.running_mean).copy(), as_numpy(scaler.running_variance).copy())
        probe = np.array([[1000.0, -1000.0], [0.75, 1.25]], dtype=np.float32)
        _, mean, variance = raw_moments(VALUES)
        expected = np.clip((probe - mean) / (np.sqrt(variance) + scaler.epsilon), -2.0, 2.0)
        result = as_numpy(scaler(to_backend(probe), train=False))
        np.testing.assert_allclose(result, expected, rtol=3e-6, atol=3e-6)
        inverse_input = np.array([[10.0, -10.0], [0.25, 0.5]], dtype=np.float32)
        inverse = as_numpy(scaler(to_backend(inverse_input), inverse=True))
        np.testing.assert_allclose(
            inverse, np.sqrt(variance) * np.clip(inverse_input, -2.0, 2.0) + mean, rtol=3e-6, atol=3e-6
        )
        self.assertIsNone(scaler(None, train=True))
        self.assertEqual(read_count(scaler), before[0])
        np.testing.assert_array_equal(as_numpy(scaler.running_mean), before[1])
        np.testing.assert_array_equal(as_numpy(scaler.running_variance), before[2])

    def test_constant_batch_is_an_unchanged_variance_control(self) -> None:
        """Constant non-singleton batches have zero variance under either estimator."""
        rows = ((-2, 3),) * 5
        for shape in ((5, 2), (1, 5, 2)):
            with self.subTest(shape=shape):
                scaler = RunningStandardScaler(2, device="cpu")
                scaler(to_backend(np.array(rows, dtype=np.float32).reshape(shape)), train=True)
                self.check_statistics(scaler, rows)
