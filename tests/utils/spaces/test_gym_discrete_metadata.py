import pytest

from importlib import import_module
import gym

import numpy as np


@pytest.mark.parametrize("framework", ["torch", "jax", "warp"])
@pytest.mark.parametrize("nested", [False, True])
def test_convert_gym_discrete_preserves_start(framework: str, nested: bool) -> None:
    pytest.importorskip(framework)
    original = gym.spaces.Discrete(4, start=8)
    space = gym.spaces.Tuple((original,)) if nested else original
    converted = import_module(f"skrl.utils.spaces.{framework}").convert_gym_space(space)
    result = converted[0] if nested else converted
    assert result.start == original.start
    assert result.n == original.n
    assert original.contains(result.sample())


@pytest.mark.parametrize("framework", ["torch", "jax", "warp"])
def test_convert_gym_multidiscrete_preserves_dtype(framework: str) -> None:
    pytest.importorskip(framework)
    original = gym.spaces.MultiDiscrete([3, 4], dtype=np.int32)
    converted = import_module(f"skrl.utils.spaces.{framework}").convert_gym_space(original)
    assert converted.dtype == original.dtype
    np.testing.assert_array_equal(converted.nvec, original.nvec)
