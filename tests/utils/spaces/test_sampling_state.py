from typing import Any

import pytest

from collections.abc import Callable, Iterator
from importlib import import_module
import gymnasium

import numpy as np


SPACE_FACTORIES = [
    lambda: gymnasium.spaces.Box(-1, 1, (3,)),
    lambda: gymnasium.spaces.Discrete(17),
    lambda: gymnasium.spaces.MultiDiscrete([17, 23]),
    lambda: gymnasium.spaces.Dict(
        {"box": gymnasium.spaces.Box(-1, 1, (3,)), "discrete": gymnasium.spaces.Discrete(17)}
    ),
]


def _arrays(sample: Any) -> Iterator[np.ndarray]:
    if isinstance(sample, dict):
        for value in sample.values():
            yield from _arrays(value)
    elif isinstance(sample, tuple):
        for value in sample:
            yield from _arrays(value)
    else:
        yield sample.numpy() if hasattr(sample, "numpy") else np.asarray(sample)


@pytest.mark.parametrize("framework", ["torch", "jax", "warp"])
@pytest.mark.parametrize("backend", ["numpy", "native"])
@pytest.mark.parametrize("space_factory", SPACE_FACTORIES, ids=["box", "discrete", "multidiscrete", "dict"])
def test_sampling_advances_state_and_reseeding_replays_batches(
    framework: str, backend: str, space_factory: Callable[[], gymnasium.Space]
) -> None:
    pytest.importorskip(framework)
    sample_space = import_module(f"skrl.utils.spaces.{framework}").sample_space
    space = space_factory()
    space.seed(123)
    generator = space.np_random
    first = list(_arrays(sample_space(space, batch_size=7, backend=backend, device="cpu")))
    second = list(_arrays(sample_space(space, batch_size=7, backend=backend, device="cpu")))
    assert space.np_random is generator
    assert any(not np.array_equal(a, b) for a, b in zip(first, second))

    space.seed(123)
    for expected in (first, second):
        replay = list(_arrays(sample_space(space, batch_size=7, backend=backend, device="cpu")))
        for actual, original in zip(replay, expected):
            np.testing.assert_array_equal(actual, original)
