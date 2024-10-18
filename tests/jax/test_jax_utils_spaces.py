import hypothesis
import hypothesis.strategies as st

import gym
import gymnasium

import jax
import jax.numpy as jnp
import numpy as np

from skrl.utils.spaces.jax import (
    compute_space_size,
    convert_gym_space,
    flatten_tensorized_space,
    sample_space,
    tensorize_space,
    unflatten_tensorized_space,
    untensorize_space
)

from ..stategies import gym_space_stategy, gymnasium_space_stategy


def _check_backend(x, backend):
    if backend == "jax":
        assert isinstance(x, jax.Array)
    elif backend == "numpy":
        assert isinstance(x, np.ndarray)
    else:
        raise ValueError(f"Invalid backend type: {backend}")

def check_sampled_space(space, x, n, backend):
    if isinstance(space, gymnasium.spaces.Box):
        _check_backend(x, backend)
        assert x.shape == (n, *space.shape)
    elif isinstance(space, gymnasium.spaces.Discrete):
        _check_backend(x, backend)
        assert x.shape == (n, 1)
    elif isinstance(space, gymnasium.spaces.MultiDiscrete):
        assert x.shape == (n, *space.nvec.shape)
    elif isinstance(space, gymnasium.spaces.Dict):
        list(map(check_sampled_space, space.values(), x.values(), [n] * len(space), [backend] * len(space)))
    elif isinstance(space, gymnasium.spaces.Tuple):
        list(map(check_sampled_space, space, x, [n] * len(space), [backend] * len(space)))
    else:
        raise ValueError(f"Invalid space type: {type(space)}")


@hypothesis.given(space=gymnasium_space_stategy())
@hypothesis.settings(suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture], deadline=None)
def test_compute_space_size(capsys, space: gymnasium.spaces.Space):
    def occupied_size(s):
        if isinstance(s, gymnasium.spaces.Discrete):
            return 1
        elif isinstance(s, gymnasium.spaces.MultiDiscrete):
            return s.nvec.shape[0]
        elif isinstance(s, gymnasium.spaces.Dict):
            return sum([occupied_size(_s) for _s in s.values()])
        elif isinstance(s, gymnasium.spaces.Tuple):
            return sum([occupied_size(_s) for _s in s])
        return gymnasium.spaces.flatdim(s)

    space_size = compute_space_size(space, occupied_size=False)
    assert space_size == gymnasium.spaces.flatdim(space)

    space_size = compute_space_size(space, occupied_size=True)
    assert space_size == occupied_size(space)

@hypothesis.given(space=gymnasium_space_stategy())
@hypothesis.settings(suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture], deadline=None)
def test_tensorize_space(capsys, space: gymnasium.spaces.Space):
    def check_tensorized_space(s, x, n):
        if isinstance(s, gymnasium.spaces.Box):
            assert isinstance(x, jax.Array) and x.shape == (n, *s.shape)
        elif isinstance(s, gymnasium.spaces.Discrete):
            assert isinstance(x, jax.Array) and x.shape == (n, 1)
        elif isinstance(s, gymnasium.spaces.MultiDiscrete):
            assert isinstance(x, jax.Array) and x.shape == (n, *s.nvec.shape)
        elif isinstance(s, gymnasium.spaces.Dict):
            list(map(check_tensorized_space, s.values(), x.values(), [n] * len(s)))
        elif isinstance(s, gymnasium.spaces.Tuple):
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

    sampled_space = sample_space(space, 5, backend="jax")
    tensorized_space = tensorize_space(space, sampled_space)
    check_tensorized_space(space, tensorized_space, 5)

@hypothesis.given(space=gymnasium_space_stategy())
@hypothesis.settings(suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture], deadline=None)
def test_untensorize_space(capsys, space: gymnasium.spaces.Space):
    def check_untensorized_space(s, x, squeeze_batch_dimension):
        if isinstance(s, gymnasium.spaces.Box):
            assert isinstance(x, np.ndarray)
            assert x.shape == s.shape if squeeze_batch_dimension else (1, *s.shape)
        elif isinstance(s, gymnasium.spaces.Discrete):
            assert isinstance(x, (np.ndarray, int))
            assert isinstance(x, int) if squeeze_batch_dimension else x.shape == (1, 1)
        elif isinstance(s, gymnasium.spaces.MultiDiscrete):
            assert isinstance(x, np.ndarray) and x.shape == s.nvec.shape if squeeze_batch_dimension else (1, *s.nvec.shape)
        elif isinstance(s, gymnasium.spaces.Dict):
            list(map(check_untensorized_space, s.values(), x.values(), [squeeze_batch_dimension] * len(s)))
        elif isinstance(s, gymnasium.spaces.Tuple):
            list(map(check_untensorized_space, s, x, [squeeze_batch_dimension] * len(s)))
        else:
            raise ValueError(f"Invalid space type: {type(s)}")

    tensorized_space = tensorize_space(space, space.sample())

    untensorized_space = untensorize_space(space, tensorized_space, squeeze_batch_dimension=False)
    check_untensorized_space(space, untensorized_space, squeeze_batch_dimension=False)

    untensorized_space = untensorize_space(space, tensorized_space, squeeze_batch_dimension=True)
    check_untensorized_space(space, untensorized_space, squeeze_batch_dimension=True)

@hypothesis.given(space=gymnasium_space_stategy(), batch_size=st.integers(min_value=1, max_value=10))
@hypothesis.settings(suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture], deadline=None)
def test_sample_space(capsys, space: gymnasium.spaces.Space, batch_size: int):

    sampled_space = sample_space(space, batch_size, backend="numpy")
    check_sampled_space(space, sampled_space, batch_size, backend="numpy")

    sampled_space = sample_space(space, batch_size, backend="jax")
    check_sampled_space(space, sampled_space, batch_size, backend="jax")

@hypothesis.given(space=gymnasium_space_stategy())
@hypothesis.settings(suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture], deadline=None)
def test_flatten_tensorized_space(capsys, space: gymnasium.spaces.Space):
    space_size = compute_space_size(space, occupied_size=True)

    tensorized_space = tensorize_space(space, space.sample())
    flattened_space = flatten_tensorized_space(tensorized_space)
    assert flattened_space.shape == (1, space_size)

    tensorized_space = sample_space(space, batch_size=5, backend="jax")
    flattened_space = flatten_tensorized_space(tensorized_space)
    assert flattened_space.shape == (5, space_size)

@hypothesis.given(space=gymnasium_space_stategy())
@hypothesis.settings(suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture], deadline=None)
def test_unflatten_tensorized_space(capsys, space: gymnasium.spaces.Space):
    tensorized_space = tensorize_space(space, space.sample())
    flattened_space = flatten_tensorized_space(tensorized_space)
    unflattened_space = unflatten_tensorized_space(space, flattened_space)
    check_sampled_space(space, unflattened_space, 1, backend="jax")

    tensorized_space = sample_space(space, batch_size=5, backend="jax")
    flattened_space = flatten_tensorized_space(tensorized_space)
    unflattened_space = unflatten_tensorized_space(space, flattened_space)
    check_sampled_space(space, unflattened_space, 5, backend="jax")

@hypothesis.given(space=gym_space_stategy())
@hypothesis.settings(suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture], deadline=None)
def test_convert_gym_space(capsys, space: gym.spaces.Space):
    def check_converted_space(gym_space, gymnasium_space):
        if isinstance(gym_space, gym.spaces.Box):
            assert isinstance(gymnasium_space, gymnasium.spaces.Box)
            assert np.all(gym_space.low == gymnasium_space.low)
            assert np.all(gym_space.high == gymnasium_space.high)
            assert gym_space.shape == gymnasium_space.shape
            assert gym_space.dtype == gymnasium_space.dtype
        elif isinstance(gym_space, gym.spaces.Discrete):
            assert isinstance(gymnasium_space, gymnasium.spaces.Discrete)
            assert gym_space.n == gymnasium_space.n
        elif isinstance(gym_space, gym.spaces.MultiDiscrete):
            assert isinstance(gymnasium_space, gymnasium.spaces.MultiDiscrete)
            assert np.all(gym_space.nvec) == np.all(gymnasium_space.nvec)
        elif isinstance(gym_space, gym.spaces.Tuple):
            assert isinstance(gymnasium_space, gymnasium.spaces.Tuple)
            assert len(gym_space) == len(gymnasium_space)
            list(map(check_converted_space, gym_space, gymnasium_space))
        elif isinstance(gym_space, gym.spaces.Dict):
            assert isinstance(gymnasium_space, gymnasium.spaces.Dict)
            assert sorted(list(gym_space.keys())) == sorted(list(gymnasium_space.keys()))
            for k in gym_space.keys():
                check_converted_space(gym_space[k], gymnasium_space[k])
        else:
            raise ValueError(f"Invalid space type: {type(gym_space)}")

    check_converted_space(space, convert_gym_space(space))
