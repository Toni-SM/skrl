import hypothesis
import hypothesis.strategies as st

import gymnasium as gym

import numpy as np
import torch

from skrl.utils.spaces.torch import (
    compute_space_size,
    flatten_tensorized_space,
    sample_space,
    tensorize_space,
    unflatten_tensorized_space
)


def _check_backend(x, backend):
    if backend == "torch":
        assert isinstance(x, torch.Tensor)
    elif backend == "numpy":
        assert isinstance(x, np.ndarray)
    else:
        raise ValueError(f"Invalid backend type: {backend}")

def check_sampled_space(space, x, n, backend):
    if isinstance(space, gym.spaces.Box):
        _check_backend(x, backend)
        assert x.shape == (n, *space.shape)
    elif isinstance(space, gym.spaces.Discrete):
        _check_backend(x, backend)
        assert x.shape == (n, 1)
    elif isinstance(space, gym.spaces.MultiDiscrete):
        assert x.shape == (n, *space.nvec.shape)
    elif isinstance(space, gym.spaces.Dict):
        list(map(check_sampled_space, space.values(), x.values(), [n] * len(space), [backend] * len(space)))
    elif isinstance(space, gym.spaces.Tuple):
        list(map(check_sampled_space, space, x, [n] * len(space), [backend] * len(space)))
    else:
        raise ValueError(f"Invalid space type: {type(space)}")

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
    def check_tensorized_space(s, x, n):
        if isinstance(s, gym.spaces.Box):
            assert isinstance(x, torch.Tensor) and x.shape == (n, *s.shape)
        elif isinstance(s, gym.spaces.Discrete):
            assert isinstance(x, torch.Tensor) and x.shape == (n, 1)
        elif isinstance(s, gym.spaces.MultiDiscrete):
            assert isinstance(x, torch.Tensor) and x.shape == (n, *s.nvec.shape)
        elif isinstance(s, gym.spaces.Dict):
            list(map(check_tensorized_space, s.values(), x.values(), [n] * len(s)))
        elif isinstance(s, gym.spaces.Tuple):
            list(map(check_tensorized_space, s, x, [n] * len(s)))
        else:
            raise ValueError(f"Invalid space type: {type(s)}")

    tensorized_space = tensorize_space(space, space.sample())
    check_tensorized_space(space, tensorized_space, 1)

    tensorized_space = tensorize_space(space, tensorized_space)
    check_tensorized_space(space, tensorized_space, 1)

    sampled_space = sample_space(space, 5, backend="numpy")
    tensorized_space = tensorize_space(space, sampled_space)
    check_tensorized_space(space, tensorized_space, 5)

    sampled_space = sample_space(space, 5, backend="torch")
    tensorized_space = tensorize_space(space, sampled_space)
    check_tensorized_space(space, tensorized_space, 5)

@hypothesis.given(space=space_stategy(), batch_size=st.integers(min_value=1, max_value=10))
@hypothesis.settings(suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture], deadline=None)
def test_sample_space(capsys, space: gym.spaces.Space, batch_size: int):

    sampled_space = sample_space(space, batch_size, backend="numpy")
    check_sampled_space(space, sampled_space, batch_size, backend="numpy")

    sampled_space = sample_space(space, batch_size, backend="torch")
    check_sampled_space(space, sampled_space, batch_size, backend="torch")

@hypothesis.given(space=space_stategy())
@hypothesis.settings(suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture], deadline=None)
def test_flatten_tensorized_space(capsys, space: gym.spaces.Space):
    space_size = compute_space_size(space, occupied_size=True)

    tensorized_space = tensorize_space(space, space.sample())
    flattened_space = flatten_tensorized_space(tensorized_space)
    assert flattened_space.shape == (1, space_size)

    tensorized_space = sample_space(space, batch_size=5, backend="torch")
    flattened_space = flatten_tensorized_space(tensorized_space)
    assert flattened_space.shape == (5, space_size)

@hypothesis.given(space=space_stategy())
@hypothesis.settings(suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture], deadline=None)
def test_unflatten_tensorized_space(capsys, space: gym.spaces.Space):
    tensorized_space = tensorize_space(space, space.sample())
    flattened_space = flatten_tensorized_space(tensorized_space)
    unflattened_space = unflatten_tensorized_space(space, flattened_space)
    check_sampled_space(space, unflattened_space, 1, backend="torch")

    tensorized_space = sample_space(space, batch_size=5, backend="torch")
    flattened_space = flatten_tensorized_space(tensorized_space)
    unflattened_space = unflatten_tensorized_space(space, flattened_space)
    check_sampled_space(space, unflattened_space, 5, backend="torch")
