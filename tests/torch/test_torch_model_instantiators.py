import hypothesis
import hypothesis.strategies as st
import pytest

import gymnasium as gym

import numpy as np
import torch

from skrl.utils.model_instantiators.torch import (
    Shape,
    categorical_model,
    deterministic_model,
    gaussian_model,
    multivariate_gaussian_model,
    shared_model
)


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
    model.to(device=device)

    observations = torch.ones((10, model.num_observations), device=device)
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
    model.to(device=device)

    observations = torch.ones((10, model.num_observations), device=device)
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
    model.to(device=device)

    observations = torch.ones((10, model.num_observations), device=device)
    output = model.act({"states": observations})
    assert output[0].shape == (10, model.num_actions)

@hypothesis.given(observation_space_size=st.integers(min_value=1, max_value=10),
                  action_space_size=st.integers(min_value=1, max_value=10))
@hypothesis.settings(suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture], deadline=None)
@pytest.mark.parametrize("device", [None, "cpu", "cuda:0"])
def test_multivariate_gaussian_model(capsys, observation_space_size, action_space_size, device):
    observation_space = gym.spaces.Box(np.array([-1] * observation_space_size), np.array([1] * observation_space_size))
    action_space = gym.spaces.Box(np.array([-1] * action_space_size), np.array([1] * action_space_size))
    # TODO: randomize all parameters
    model = multivariate_gaussian_model(observation_space=observation_space,
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
    model.to(device=device)

    observations = torch.ones((10, model.num_observations), device=device)
    output = model.act({"states": observations})
    assert output[0].shape == (10, model.num_actions)

@hypothesis.given(observation_space_size=st.integers(min_value=1, max_value=10),
                  action_space_size=st.integers(min_value=1, max_value=10))
@hypothesis.settings(suppress_health_check=[hypothesis.HealthCheck.function_scoped_fixture], deadline=None)
@pytest.mark.parametrize("device", [None, "cpu", "cuda:0"])
def test_shared_model(capsys, observation_space_size, action_space_size, device):
    observation_space = gym.spaces.Box(np.array([-1] * observation_space_size), np.array([1] * observation_space_size))
    action_space = gym.spaces.Box(np.array([-1] * action_space_size), np.array([1] * action_space_size))
    # TODO: randomize all parameters
    model = shared_model(observation_space=observation_space,
                         action_space=action_space,
                         device=device,
                         structure="",
                         roles=["policy", "value"],
                         parameters=[
                            {
                                "clip_actions": False,
                                "clip_log_std": True,
                                "min_log_std": -20,
                                "max_log_std": 2,
                                "initial_log_std": 0,
                                "input_shape": Shape.STATES,
                                "hiddens": [256, 256],
                                "hidden_activation": ["relu", "relu"],
                                "output_shape": Shape.ACTIONS,
                                "output_activation": None,
                                "output_scale": 1,
                            },
                            {
                                "clip_actions": False,
                                "input_shape": Shape.STATES,
                                "hiddens": [256, 256],
                                "hidden_activation": ["relu", "relu"],
                                "output_shape": Shape.ONE,
                                "output_activation": None,
                                "output_scale": 1,
                            }
                         ],
                         single_forward_pass=True)
    model.to(device=device)

    observations = torch.ones((10, model.num_observations), device=device)
    output = model.act({"states": observations}, "policy")
    assert output[0].shape == (10, model.num_actions)
    output = model.act({"states": observations}, "value")
    assert output[0].shape == (10, 1)
