import hypothesis
import hypothesis.strategies as st
import pytest

import gym
import gymnasium

import numpy as np
import torch

from skrl.models.torch import Model
from skrl.utils.model_instantiators.torch.common import one_hot_encoding
from skrl.utils.spaces.torch import (
    compute_space_limits,
    compute_space_size,
    convert_gym_space,
    flatten_tensorized_space,
    sample_space,
    tensorize_space,
    unflatten_tensorized_space,
    untensorize_space,
)

from ....strategies import gym_space_stategy, gymnasium_space_stategy


def _check_backend(x, backend):
    if backend == "native":
        assert isinstance(x, torch.Tensor)
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
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture],
    deadline=None,
    phases=[hypothesis.Phase.explicit, hypothesis.Phase.reuse, hypothesis.Phase.generate],
)
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

    assert compute_space_size(None) == 0

    space_size = compute_space_size(space, occupied_size=False)
    assert space_size == gymnasium.spaces.flatdim(space)

    space_size = compute_space_size(space, occupied_size=True)
    assert space_size == occupied_size(space)


def test_compute_space_limits(capsys):
    space = gymnasium.spaces.Dict(
        {
            "a": gymnasium.spaces.Box(low=0, high=1, shape=(2,)),
            "b": gymnasium.spaces.Discrete(n=2),
            "c": gymnasium.spaces.MultiDiscrete(nvec=[2, 3]),
            "d": gymnasium.spaces.Tuple(
                (
                    gymnasium.spaces.Box(low=-2, high=-1, shape=(3,)),
                    gymnasium.spaces.Discrete(n=2),
                    gymnasium.spaces.MultiDiscrete(nvec=[2, 3]),
                )
            ),
            "e": gymnasium.spaces.Dict({"a": gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(2,))}),
        }
    )

    # fmt: off
    expected_low = np.array([ 0, 0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -2, -2, -2, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf])
    expected_high = np.array([1, 1,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf, -1, -1, -1,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf])
    # fmt: on
    low, high = compute_space_limits(space, occupied_size=False)
    assert (
        low.cpu().numpy() == expected_low
    ).all(), f"Expected {expected_low.tolist()}, got {low.cpu().numpy().tolist()}"
    assert (
        high.cpu().numpy() == expected_high
    ).all(), f"Expected {expected_high.tolist()}, got {high.cpu().numpy().tolist()}"

    for none_if_unbounded in ["both", "below", "above", "any"]:
        low, high = compute_space_limits(space, occupied_size=False, none_if_unbounded=none_if_unbounded)
        assert low is not None and high is not None

    # fmt: off
    expected_low = np.array([ 0, 0, -np.inf, -np.inf, -np.inf, -2, -2, -2, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf])
    expected_high = np.array([1, 1,  np.inf,  np.inf,  np.inf, -1, -1, -1,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf])
    # fmt: on
    low, high = compute_space_limits(space, occupied_size=True)
    assert (
        low.cpu().numpy() == expected_low
    ).all(), f"Expected {expected_low.tolist()}, got {low.cpu().numpy().tolist()}"
    assert (
        high.cpu().numpy() == expected_high
    ).all(), f"Expected {expected_high.tolist()}, got {high.cpu().numpy().tolist()}"

    for none_if_unbounded in ["both", "below", "above", "any"]:
        low, high = compute_space_limits(space, occupied_size=True, none_if_unbounded=none_if_unbounded)
        assert low is not None and high is not None

    space = gymnasium.spaces.Tuple(
        (
            gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
            gymnasium.spaces.Discrete(n=2),
            gymnasium.spaces.MultiDiscrete(nvec=[2, 3]),
        )
    )
    low, high = compute_space_limits(space, none_if_unbounded="both")
    assert low is None and high is None
    low, high = compute_space_limits(space, none_if_unbounded="below")
    assert low is None and np.isinf(high.cpu().numpy()).all()
    low, high = compute_space_limits(space, none_if_unbounded="above")
    assert np.isinf(low.cpu().numpy()).all() and high is None
    low, high = compute_space_limits(space, none_if_unbounded="any")
    assert low is None and high is None


@hypothesis.given(space=gymnasium_space_stategy())
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture],
    deadline=None,
    phases=[hypothesis.Phase.explicit, hypothesis.Phase.reuse, hypothesis.Phase.generate],
)
def test_tensorize_space(capsys, space: gymnasium.spaces.Space):
    def check_tensorized_space(s, x, n):
        if isinstance(s, gymnasium.spaces.Box):
            assert isinstance(x, torch.Tensor) and x.shape == (n, *s.shape)
        elif isinstance(s, gymnasium.spaces.Discrete):
            assert isinstance(x, torch.Tensor) and x.shape == (n, 1)
        elif isinstance(s, gymnasium.spaces.MultiDiscrete):
            assert isinstance(x, torch.Tensor) and x.shape == (n, *s.nvec.shape)
        elif isinstance(s, gymnasium.spaces.Dict):
            list(map(check_tensorized_space, s.values(), x.values(), [n] * len(s)))
        elif isinstance(s, gymnasium.spaces.Tuple):
            list(map(check_tensorized_space, s, x, [n] * len(s)))
        else:
            raise ValueError(f"Invalid space type: {type(s)}")

    assert tensorize_space(None, None) is None
    assert tensorize_space(space, None) is None

    tensorized_space = tensorize_space(space, space.sample())
    check_tensorized_space(space, tensorized_space, 1)

    tensorized_space = tensorize_space(space, tensorized_space)
    check_tensorized_space(space, tensorized_space, 1)

    sampled_space = sample_space(space, batch_size=5, backend="numpy")
    tensorized_space = tensorize_space(space, sampled_space)
    check_tensorized_space(space, tensorized_space, 5)

    sampled_space = sample_space(space, batch_size=5, backend="native")
    tensorized_space = tensorize_space(space, sampled_space)
    check_tensorized_space(space, tensorized_space, 5)


@hypothesis.given(space=gymnasium_space_stategy())
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture],
    deadline=None,
    phases=[hypothesis.Phase.explicit, hypothesis.Phase.reuse, hypothesis.Phase.generate],
)
def test_untensorize_space(capsys, space: gymnasium.spaces.Space):
    def check_untensorized_space(s, x, squeeze_batch_dimension):
        if isinstance(s, gymnasium.spaces.Box):
            assert isinstance(x, np.ndarray)
            assert x.shape == s.shape if squeeze_batch_dimension else (1, *s.shape)
        elif isinstance(s, gymnasium.spaces.Discrete):
            assert isinstance(x, (np.ndarray, int))
            assert isinstance(x, int) if squeeze_batch_dimension else x.shape == (1, 1)
        elif isinstance(s, gymnasium.spaces.MultiDiscrete):
            assert (
                isinstance(x, np.ndarray) and x.shape == s.nvec.shape if squeeze_batch_dimension else (1, *s.nvec.shape)
            )
        elif isinstance(s, gymnasium.spaces.Dict):
            list(map(check_untensorized_space, s.values(), x.values(), [squeeze_batch_dimension] * len(s)))
        elif isinstance(s, gymnasium.spaces.Tuple):
            list(map(check_untensorized_space, s, x, [squeeze_batch_dimension] * len(s)))
        else:
            raise ValueError(f"Invalid space type: {type(s)}")

    tensorized_space = tensorize_space(space, space.sample())

    assert untensorize_space(None, None) is None
    assert untensorize_space(space, None) is None

    untensorized_space = untensorize_space(space, tensorized_space, squeeze_batch_dimension=False)
    check_untensorized_space(space, untensorized_space, squeeze_batch_dimension=False)

    untensorized_space = untensorize_space(space, tensorized_space, squeeze_batch_dimension=True)
    check_untensorized_space(space, untensorized_space, squeeze_batch_dimension=True)


@hypothesis.given(space=gymnasium_space_stategy(), batch_size=st.integers(min_value=1, max_value=10))
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture],
    deadline=None,
    phases=[hypothesis.Phase.explicit, hypothesis.Phase.reuse, hypothesis.Phase.generate],
)
def test_sample_space(capsys, space: gymnasium.spaces.Space, batch_size: int):

    assert sample_space(None) is None

    sampled_space = sample_space(space, batch_size=batch_size, backend="numpy")
    check_sampled_space(space, sampled_space, batch_size, backend="numpy")

    sampled_space = sample_space(space, batch_size=batch_size, backend="native")
    check_sampled_space(space, sampled_space, batch_size, backend="native")


@hypothesis.given(space=gymnasium_space_stategy())
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture],
    deadline=None,
    phases=[hypothesis.Phase.explicit, hypothesis.Phase.reuse, hypothesis.Phase.generate],
)
def test_flatten_tensorized_space(capsys, space: gymnasium.spaces.Space):
    space_size = compute_space_size(space, occupied_size=True)

    assert flatten_tensorized_space(None) is None

    tensorized_space = tensorize_space(space, space.sample())
    flattened_space = flatten_tensorized_space(tensorized_space)
    assert flattened_space.shape == (1, space_size)

    tensorized_space = sample_space(space, batch_size=5, backend="native")
    flattened_space = flatten_tensorized_space(tensorized_space)
    assert flattened_space.shape == (5, space_size)


@hypothesis.given(space=gymnasium_space_stategy())
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture],
    deadline=None,
    phases=[hypothesis.Phase.explicit, hypothesis.Phase.reuse, hypothesis.Phase.generate],
)
def test_unflatten_tensorized_space(capsys, space: gymnasium.spaces.Space):
    assert unflatten_tensorized_space(None, None) is None
    assert unflatten_tensorized_space(space, None) is None

    tensorized_space = tensorize_space(space, space.sample())
    flattened_space = flatten_tensorized_space(tensorized_space)
    unflattened_space = unflatten_tensorized_space(space, flattened_space)
    check_sampled_space(space, unflattened_space, 1, backend="native")

    tensorized_space = sample_space(space, batch_size=5, backend="native")
    flattened_space = flatten_tensorized_space(tensorized_space)
    unflattened_space = unflatten_tensorized_space(space, flattened_space)
    check_sampled_space(space, unflattened_space, 5, backend="native")


@hypothesis.given(space=gym_space_stategy())
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture],
    deadline=None,
    phases=[hypothesis.Phase.explicit, hypothesis.Phase.reuse, hypothesis.Phase.generate],
)
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
            assert gym_space.start == gymnasium_space.start
        elif isinstance(gym_space, gym.spaces.MultiDiscrete):
            assert isinstance(gymnasium_space, gymnasium.spaces.MultiDiscrete)
            assert np.all(gym_space.nvec) == np.all(gymnasium_space.nvec)
            assert gym_space.dtype == gymnasium_space.dtype
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

    assert convert_gym_space(None) is None
    check_converted_space(space, convert_gym_space(space))


@pytest.mark.parametrize(
    "space",
    [
        gymnasium.spaces.Discrete(3, start=1),
        gymnasium.spaces.Discrete(4, start=-2),
        gymnasium.spaces.MultiDiscrete([3, 2], start=[1, 5]),
        convert_gym_space(gym.spaces.Discrete(3, start=1)),
    ],
)
def test_discrete_spaces_with_start(capsys, space: gymnasium.spaces.Space):
    nvec = torch.tensor(
        np.array(space.nvec if isinstance(space, gymnasium.spaces.MultiDiscrete) else space.n).reshape(-1)
    )

    def in_range(x):
        x = torch.as_tensor(x)
        return bool(((x >= 0) & (x < nvec)).all())

    # environment values are in [start, start + n): tensorized values must be in [0, n)
    values = np.array([space.sample() for _ in range(64)])
    tensorized_space = tensorize_space(space, values)
    assert in_range(tensorized_space)
    # untensorizing must recover the original environment values
    untensorized_space = untensorize_space(space, tensorized_space, squeeze_batch_dimension=False)
    assert np.array_equal(untensorized_space.reshape(values.shape), values)
    # single (non-batched) value round trip
    value = space.sample()
    tensorized_space = tensorize_space(space, value.tolist() if isinstance(value, np.ndarray) else int(value))
    assert in_range(tensorized_space)
    assert np.array_equal(untensorize_space(space, tensorized_space, squeeze_batch_dimension=True), value)

    # sampled values (e.g.: random actions) are in [0, n), as the tensorized ones
    for backend in ["numpy", "native"]:
        assert in_range(sample_space(space, batch_size=64, backend=backend))


def test_discrete_space_with_start_random_act_one_hot_encoding(capsys):
    class _Model(Model):
        def compute(self, inputs, *, role=""):
            return inputs["observations"], {}

        def act(self, inputs, *, role=""):
            return self.random_act(inputs, role=role)

    action_space = convert_gym_space(gym.spaces.Discrete(3, start=1))
    observation_space = gymnasium.spaces.Box(low=-1, high=1, shape=(2,))
    model = _Model(observation_space=observation_space, action_space=action_space, device="cpu")

    actions = model.random_act({"observations": torch.zeros((256, 2))})[0]
    assert ((actions >= 0) & (actions < action_space.n)).all()
    # random actions can be one-hot encoded and used to index (e.g.: DQN's Q-values)
    one_hot = one_hot_encoding(action_space, actions)
    assert one_hot.shape == (256, action_space.n) and (one_hot.sum(dim=1) == 1).all()
    torch.gather(torch.zeros((256, action_space.n)), dim=1, index=actions.long())
    # random actions are valid environment actions once untensorized
    env_actions = untensorize_space(action_space, actions, squeeze_batch_dimension=False)
    assert all(action_space.contains(int(a)) for a in env_actions.reshape(-1))


@pytest.mark.parametrize("space", [gymnasium.spaces.Discrete(3), gymnasium.spaces.MultiDiscrete([3, 2])])
def test_discrete_spaces_without_start_are_unchanged(capsys, space: gymnasium.spaces.Space):
    values = np.array([space.sample() for _ in range(16)])
    tensorized_space = tensorize_space(space, values)
    assert np.array_equal(tensorized_space.numpy().reshape(values.shape), values)
    untensorized_space = untensorize_space(space, tensorized_space, squeeze_batch_dimension=False)
    assert np.array_equal(untensorized_space.reshape(values.shape), values)
