import hypothesis
import hypothesis.strategies as st
import pytest

import gymnasium as gym

import jax
import jax.numpy as jnp
import numpy as np

from skrl.utils.model_instantiators.jax import Shape, categorical_model, deterministic_model, gaussian_model


@hypothesis.given(observation_space_size=st.integers(min_value=1, max_value=10),
                  action_space_size=st.integers(min_value=1, max_value=10))
@hypothesis.settings(suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture], deadline=None)
@pytest.mark.parametrize("device", [None, "cpu", "cuda:0"])
def test_categorical_model(capsys, observation_space_size, action_space_size, device):
    observation_space = gym.spaces.Box(np.array([-1] * observation_space_size), np.array([1] * observation_space_size))
    action_space = gym.spaces.Discrete(action_space_size)
    # TODO: randomize all parameters
    model = categorical_model(observation_space=observation_space,
                              action_space=action_space,
                              device=device,
                              unnormalized_log_prob=True,
                              input_shape=Shape.STATES,
                              hiddens=[256, 256],
                              hidden_activation=["relu", "relu"],
                              output_shape=Shape.ACTIONS,
                              output_activation=None)
    model.init_state_dict("model")

    with jax.default_device(model.device):
        observations = jnp.ones((10, model.num_observations))
    output = model.act({"states": observations})
    assert output[0].shape == (10, 1)

@hypothesis.given(observation_space_size=st.integers(min_value=1, max_value=10),
                  action_space_size=st.integers(min_value=1, max_value=10))
@hypothesis.settings(suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture], deadline=None)
@pytest.mark.parametrize("device", [None, "cpu", "cuda:0"])
def test_deterministic_model(capsys, observation_space_size, action_space_size, device):
    observation_space = gym.spaces.Box(np.array([-1] * observation_space_size), np.array([1] * observation_space_size))
    action_space = gym.spaces.Box(np.array([-1] * action_space_size), np.array([1] * action_space_size))
    # TODO: randomize all parameters
    model = deterministic_model(observation_space=observation_space,
                                action_space=action_space,
                                device=device,
                                clip_actions=False,
                                input_shape=Shape.STATES,
                                hiddens=[256, 256],
                                hidden_activation=["relu", "relu"],
                                output_shape=Shape.ACTIONS,
                                output_activation=None,
                                output_scale=1)
    model.init_state_dict("model")

    with jax.default_device(model.device):
        observations = jnp.ones((10, model.num_observations))
    output = model.act({"states": observations})
    assert output[0].shape == (10, model.num_actions)

@hypothesis.given(observation_space_size=st.integers(min_value=1, max_value=10),
                  action_space_size=st.integers(min_value=1, max_value=10))
@hypothesis.settings(suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture], deadline=None)
@pytest.mark.parametrize("device", [None, "cpu", "cuda:0"])
def test_gaussian_model(capsys, observation_space_size, action_space_size, device):
    observation_space = gym.spaces.Box(np.array([-1] * observation_space_size), np.array([1] * observation_space_size))
    action_space = gym.spaces.Box(np.array([-1] * action_space_size), np.array([1] * action_space_size))
    # TODO: randomize all parameters
    model = gaussian_model(observation_space=observation_space,
                           action_space=action_space,
                           device=device,
                           clip_actions=False,
                           clip_log_std=True,
                           min_log_std=-20,
                           max_log_std=2,
                           initial_log_std=0,
                           input_shape=Shape.STATES,
                           hiddens=[256, 256],
                           hidden_activation=["relu", "relu"],
                           output_shape=Shape.ACTIONS,
                           output_activation=None,
                           output_scale=1)
    model.init_state_dict("model")

    with jax.default_device(model.device):
        observations = jnp.ones((10, model.num_observations))
    output = model.act({"states": observations})
    assert output[0].shape == (10, model.num_actions)
