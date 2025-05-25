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


NETWORK_SPEC = [
    (
        r"""
    network:
      - name: net
        input: PLACEHOLDER
        layers: [32, 32, 32]
        activations: elu
    """,
        spaces.Box(low=-1, high=1, shape=(2,)),
    ),
    (
        r"""
    network:
      - name: net
        input: PLACEHOLDER
        layers: [32, 32, 32]
        activations: elu
    """,
        spaces.Discrete(2),
    ),
    (
        r"""
    network:
      - name: net
        input: PLACEHOLDER
        layers: [32, 32, 32]
        activations: elu
    """,
        spaces.MultiDiscrete([2, 3]),
    ),
    (
        r"""
    network:
      - name: net_0
        input: PLACEHOLDER[0]
        layers: [32, 32, 32]
        activations: elu
      - name: net_1
        input: PLACEHOLDER[1]
        layers: [32, 32, 32]
        activations: elu
      - name: net
        input: net_0 + net_1
        layers: [32, 32, 32]
        activations: elu
    """,
        spaces.Tuple((spaces.Box(low=-1, high=1, shape=(2,)), spaces.Box(low=-1, high=1, shape=(3,)))),
    ),
    (
        r"""
    network:
      - name: net_0
        input: PLACEHOLDER["0"]
        layers: [32, 32, 32]
        activations: elu
      - name: net_1
        input: PLACEHOLDER["1"]
        layers: [32, 32, 32]
        activations: elu
      - name: net
        input: net_0 + net_1
        layers: [32, 32, 32]
        activations: elu
    """,
        spaces.Dict({"0": spaces.Box(low=-1, high=1, shape=(2,)), "1": spaces.Box(low=-1, high=1, shape=(3,))}),
    ),
]


def _sample_inputs(token, space, device):
    sample = flatten_tensorized_space(sample_space(space, batch_size=10, backend="native", device=device))
    return {{"OBSERVATIONS": "observations", "STATES": "states", "ACTIONS": "taken_actions"}[token]: sample}


def _define_space_arg(token, space):
    if token == "ACTIONS":
        return {}
    return {{"OBSERVATIONS": "observation_space", "STATES": "state_space"}[token]: space}


@pytest.mark.parametrize("device", [None, "cpu", "cuda:0"])
def test_categorical_model(capsys, device):
    action_space = spaces.Discrete(2)
    for network_spec, input_space in NETWORK_SPEC:
        for token in ["OBSERVATIONS", "STATES", "ACTIONS"]:
            if token == "ACTIONS":
                if type(action_space) == type(input_space):
                    input_space = action_space
                else:
                    continue
            model = categorical_model(
                **_define_space_arg(token, input_space),
                action_space=action_space,
                device=device,
                unnormalized_log_prob=True,
                network=yaml.safe_load(network_spec.replace("PLACEHOLDER", token))["network"],
            )
            model.init_state_dict()

            output = model.act(_sample_inputs(token, input_space, device))
            assert len(output) == 2
            assert output[0].shape == (10, 1)


@pytest.mark.parametrize("device", [None, "cpu", "cuda:0"])
def test_multicategorical_model(capsys, device):
    action_space = spaces.MultiDiscrete([3, 4])
    for network_spec, input_space in NETWORK_SPEC:
        for token in ["OBSERVATIONS", "STATES", "ACTIONS"]:
            if token == "ACTIONS":
                if type(action_space) == type(input_space):
                    input_space = action_space
                else:
                    continue
            model = multicategorical_model(
                **_define_space_arg(token, input_space),
                action_space=action_space,
                device=device,
                unnormalized_log_prob=True,
                network=yaml.safe_load(network_spec.replace("PLACEHOLDER", token))["network"],
                output="ACTIONS",
            )
            model.init_state_dict()

            output = model.act(_sample_inputs(token, input_space, device))
            assert len(output) == 2
            assert output[0].shape == (10, 2)


@pytest.mark.parametrize("device", [None, "cpu", "cuda:0"])
def test_deterministic_model(capsys, device):
    action_space = spaces.Box(low=-1, high=1, shape=(2,))
    for network_spec, input_space in NETWORK_SPEC:
        for token in ["OBSERVATIONS", "STATES", "ACTIONS"]:
            if token == "ACTIONS":
                if type(action_space) == type(input_space):
                    input_space = action_space
                else:
                    continue
            model = deterministic_model(
                **_define_space_arg(token, input_space),
                action_space=action_space,
                device=device,
                clip_actions=False,
                network=yaml.safe_load(network_spec.replace("PLACEHOLDER", token))["network"],
                output="ACTIONS",
            )
            model.init_state_dict()

            output = model.act(_sample_inputs(token, input_space, device))
            assert len(output) == 2
            assert output[0].shape == (10, 2)


@pytest.mark.parametrize("device", [None, "cpu", "cuda:0"])
def test_gaussian_model(capsys, device):
    action_space = spaces.Box(low=-1, high=1, shape=(2,))
    for network_spec, input_space in NETWORK_SPEC:
        for token in ["OBSERVATIONS", "STATES", "ACTIONS"]:
            if token == "ACTIONS":
                if type(action_space) == type(input_space):
                    input_space = action_space
                else:
                    continue
            model = gaussian_model(
                **_define_space_arg(token, input_space),
                action_space=action_space,
                device=device,
                clip_actions=False,
                clip_log_std=True,
                min_log_std=-20,
                max_log_std=2,
                initial_log_std=0,
                network=yaml.safe_load(network_spec.replace("PLACEHOLDER", token))["network"],
                output="ACTIONS",
            )
            model.init_state_dict()

            output = model.act(_sample_inputs(token, input_space, device))
            assert len(output) == 2
            assert output[0].shape == (10, 2)
