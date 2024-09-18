import hypothesis
import hypothesis.strategies as st
import pytest

import gymnasium as gym

import numpy as np
import torch

from skrl.utils.spaces.torch import compute_space_size, flatten_tensorized_space, tensorize_space


@st.composite
def space_stategy(draw, space_type: str = "", remaining_iterations: int = 5) -> gym.spaces.Space:
    if not space_type:
        space_type = draw(st.sampled_from(["Box", "Discrete", "MultiDiscrete", "Dict", "Tuple"]))
    # recursion base case
    if remaining_iterations <= 0 and space_type == "Dict":
        space_type = "Box"

    if space_type == "Box":
        shape = draw(st.lists(st.integers(min_value=1, max_value=5), min_size=1, max_size=5))
        return gym.spaces.Box(low=-1, high=1, shape=shape)
    elif space_type == "Discrete":
        n = draw(st.integers(min_value=1, max_value=5))
        return gym.spaces.Discrete(n)
    elif space_type == "MultiDiscrete":
        nvec = draw(st.lists(st.integers(min_value=1, max_value=5), min_size=1, max_size=5))
        return gym.spaces.MultiDiscrete(nvec)
    elif space_type == "Dict":
        remaining_iterations -= 1
        keys = draw(st.lists(st.text(st.characters(codec="ascii"), min_size=1, max_size=5), min_size=1, max_size=3))
        spaces = {key: draw(space_stategy(remaining_iterations=remaining_iterations)) for key in keys}
        return gym.spaces.Dict(spaces)
    elif space_type == "Tuple":
        remaining_iterations -= 1
        spaces = draw(st.lists(space_stategy(remaining_iterations=remaining_iterations), min_size=1, max_size=3))
        return gym.spaces.Tuple(spaces)
    else:
        raise ValueError(f"Invalid space type: {space_type}")


@hypothesis.given(space=space_stategy())
@hypothesis.settings(suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture], deadline=None)
def test_compute_space_size(capsys, space: gym.spaces.Space):
    def occupied_size(s):
        if isinstance(s, gym.spaces.Discrete):
            return 1
        elif isinstance(s, gym.spaces.MultiDiscrete):
            return s.nvec.shape[0]
        elif isinstance(s, gym.spaces.Dict):
            return sum([occupied_size(_s) for _s in s.values()])
        elif isinstance(s, gym.spaces.Tuple):
            return sum([occupied_size(_s) for _s in s])
        return gym.spaces.flatdim(s)

    space_size = compute_space_size(space, occupied_size=False)
    assert space_size == gym.spaces.flatdim(space)

    space_size = compute_space_size(space, occupied_size=True)
    assert space_size == occupied_size(space)

@hypothesis.given(space=space_stategy())
@hypothesis.settings(suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture], deadline=None)
def test_tensorize_space(capsys, space: gym.spaces.Space):
    def check_tensorized_space(s, x):
        if isinstance(s, gym.spaces.Box):
            assert x.shape == torch.Size([1, *s.shape])
        elif isinstance(s, gym.spaces.Discrete):
            assert x.ndim == 2 and x.shape[1] == 1
        elif isinstance(s, gym.spaces.MultiDiscrete):
            assert x.ndim == 2 and x.shape[1] == s.nvec.shape[0]
        elif isinstance(s, gym.spaces.Dict):
            list(map(check_tensorized_space, s.values(), x.values()))
        elif isinstance(s, gym.spaces.Tuple):
            list(map(check_tensorized_space, s, x))
        else:
            raise ValueError(f"Invalid space type: {type(s)}")

    tensorized_space = tensorize_space(space, space.sample())
    check_tensorized_space(space, tensorized_space)

    tensorized_space = tensorize_space(space, tensorized_space)
    check_tensorized_space(space, tensorized_space)

@hypothesis.given(space=space_stategy())
@hypothesis.settings(suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture], deadline=None)
def test_flatten_tensorized_space(capsys, space: gym.spaces.Space):
    tensorized_space = tensorize_space(space, space.sample())
    space_size = compute_space_size(space, occupied_size=True)

    flattened_space = flatten_tensorized_space(tensorized_space)
    with capsys.disabled():
        print(space, flattened_space.shape)
    assert flattened_space.shape == torch.Size([1, space_size])
