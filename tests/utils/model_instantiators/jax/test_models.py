import pytest

import yaml
from gymnasium import spaces

from skrl.utils.model_instantiators.jax import (
    categorical_model,
    deterministic_model,
    gaussian_model,
    multicategorical_model,
)
from skrl.utils.spaces.jax import flatten_tensorized_space, sample_space


NETWORK_SPEC_OBSERVATION = {
    spaces.Box: (
        r"""
    network:
      - name: net
        input: OBSERVATIONS
        layers: [32, 32, 32]
        activations: elu
    """,
        spaces.Box(low=-1, high=1, shape=(2,)),
    ),
    spaces.Discrete: r"""
    network:
      - name: net
        input: OBSERVATIONS
        layers: [32, 32, 32]
        activations: elu
    """,
    spaces.MultiDiscrete: r"""
    network:
      - name: net
        input: OBSERVATIONS
        layers: [32, 32, 32]
        activations: elu
    """,
    spaces.Tuple: (
        r"""
    network:
      - name: net_0
        input: OBSERVATIONS[0]
        layers: [32, 32, 32]
        activations: elu
      - name: net_1
        input: OBSERVATIONS[1]
        layers: [32, 32, 32]
        activations: elu
      - name: net
        input: net_0 + net_1
        layers: [32, 32, 32]
        activations: elu
    """,
        spaces.Tuple((spaces.Box(low=-1, high=1, shape=(2,)), spaces.Box(low=-1, high=1, shape=(3,)))),
    ),
    spaces.Dict: (
        r"""
    network:
      - name: net_0
        input: OBSERVATIONS["0"]
        layers: [32, 32, 32]
        activations: elu
      - name: net_1
        input: OBSERVATIONS["1"]
        layers: [32, 32, 32]
        activations: elu
      - name: net
        input: net_0 + net_1
        layers: [32, 32, 32]
        activations: elu
    """,
        spaces.Dict({"0": spaces.Box(low=-1, high=1, shape=(2,)), "1": spaces.Box(low=-1, high=1, shape=(3,))}),
    ),
}


@pytest.mark.parametrize("device", [None, "cpu", "cuda:0"])
def test_categorical_model(capsys, device):
    # observation
    action_space = spaces.Discrete(2)
    for observation_space_type in [spaces.Box, spaces.Tuple, spaces.Dict]:
        observation_space = NETWORK_SPEC_OBSERVATION[observation_space_type][1]
        model = categorical_model(
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            unnormalized_log_prob=True,
            network=yaml.safe_load(NETWORK_SPEC_OBSERVATION[observation_space_type][0])["network"],
            output="ACTIONS",
        )
        model.init_state_dict()

        output = model.act(
            {
                "observations": flatten_tensorized_space(
                    sample_space(observation_space, batch_size=10, backend="native", device=device)
                )
            }
        )
        assert output[0].shape == (10, 1)


@pytest.mark.parametrize("device", [None, "cpu", "cuda:0"])
def test_multicategorical_model(capsys, device):
    # observation
    action_space = spaces.MultiDiscrete([3, 4])
    for observation_space_type in [spaces.Box, spaces.Tuple, spaces.Dict]:
        observation_space = NETWORK_SPEC_OBSERVATION[observation_space_type][1]
        model = multicategorical_model(
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            unnormalized_log_prob=True,
            network=yaml.safe_load(NETWORK_SPEC_OBSERVATION[observation_space_type][0])["network"],
            output="ACTIONS",
        )
        model.init_state_dict()

        output = model.act(
            {
                "observations": flatten_tensorized_space(
                    sample_space(observation_space, batch_size=10, backend="native", device=device)
                )
            }
        )
        assert output[0].shape == (10, 2)


@pytest.mark.parametrize("device", [None, "cpu", "cuda:0"])
def test_deterministic_model(capsys, device):
    # observation
    action_space = spaces.Box(low=-1, high=1, shape=(2,))
    for observation_space_type in [spaces.Box, spaces.Tuple, spaces.Dict]:
        observation_space = NETWORK_SPEC_OBSERVATION[observation_space_type][1]
        model = deterministic_model(
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            clip_actions=False,
            network=yaml.safe_load(NETWORK_SPEC_OBSERVATION[observation_space_type][0])["network"],
            output="ACTIONS",
        )
        model.init_state_dict()

        output = model.act(
            {
                "observations": flatten_tensorized_space(
                    sample_space(observation_space, batch_size=10, backend="native", device=device)
                )
            }
        )
        assert output[0].shape == (10, 2)


@pytest.mark.parametrize("device", [None, "cpu", "cuda:0"])
def test_gaussian_model(capsys, device):
    # observation
    action_space = spaces.Box(low=-1, high=1, shape=(2,))
    for observation_space_type in [spaces.Box, spaces.Tuple, spaces.Dict]:
        observation_space = NETWORK_SPEC_OBSERVATION[observation_space_type][1]
        model = gaussian_model(
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            clip_actions=False,
            clip_log_std=True,
            min_log_std=-20,
            max_log_std=2,
            initial_log_std=0,
            network=yaml.safe_load(NETWORK_SPEC_OBSERVATION[observation_space_type][0])["network"],
            output="ACTIONS",
        )
        model.init_state_dict()

        output = model.act(
            {
                "observations": flatten_tensorized_space(
                    sample_space(observation_space, batch_size=10, backend="native", device=device)
                )
            }
        )
        assert output[0].shape == (10, 2)
