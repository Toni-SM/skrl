import hypothesis
import hypothesis.strategies as st
import pytest

import gymnasium as gym
import yaml

import flax
import jax.numpy as jnp
import numpy as np

from skrl.utils.model_instantiators.jax import (
    Shape,
    categorical_model,
    deterministic_model,
    gaussian_model,
    multicategorical_model,
)
from skrl.utils.model_instantiators.jax.common import _generate_modules, _get_activation_function, _parse_input


def test_get_activation_function(capsys):
    _globals = {"nn": flax.linen, "x": jnp.ones((1, 1))}

    activations = ["relu", "tanh", "sigmoid", "leaky_relu", "elu", "softplus", "softsign", "selu", "softmax"]
    for item in activations:
        activation = _get_activation_function(item)
        assert activation is not None, f"{item} -> None"
        exec(f"{activation}(x)", _globals, {})


def test_parse_input(capsys):
    # check for Shape enum (compatibility with prior versions)
    for input in [Shape.STATES, Shape.OBSERVATIONS, Shape.ACTIONS, Shape.STATES_ACTIONS]:
        # Shape enum with/without class
        output = _parse_input(str(input))
        output_1 = _parse_input(str(input).replace("Shape.", ""))
        assert output == output_1, f"'{output}' != '{output_1}'"
        # Shape is not in output
        for item in ["Shape", "STATES", "OBSERVATIONS", "ACTIONS", "STATES_ACTIONS"]:
            assert item not in output, f"'{item}' in '{output}'"
    # check for tokens
    for input in ["STATES", "OBSERVATIONS", "ACTIONS", "STATES_ACTIONS"]:
        output = _parse_input(input)
        for item in ["STATES", "OBSERVATIONS", "ACTIONS", "STATES_ACTIONS"]:
            assert item not in output, f"'{item}' in '{output}'"
    # Mixed operation
    input = 'OBSERVATIONS["joint"] + concatenate([net * ACTIONS[:, -3:]])'
    statement = 'states["joint"] + jnp.concatenate([net * taken_actions[:, -3:]], axis=-1)'
    output = _parse_input(str(input))
    assert output.replace("'", '"') == statement, f"'{output}' != '{statement}'"


def test_generate_modules(capsys):
    _globals = {"nn": flax.linen}

    # activation functions
    content = r"""
    activations: [relu, tanh, sigmoid, leaky_relu, elu, softplus, softsign, selu, softmax]
    """
    content = yaml.safe_load(content)
    modules = _generate_modules([1] * len(content["activations"]), content["activations"])
    _locals = {}
    exec(f'container = nn.Sequential([{", ".join(modules)}])', _globals, _locals)
    container = _locals["container"]
    with capsys.disabled():
        print("\nactivations:", container)
    assert isinstance(container, flax.linen.Sequential)
    assert len(container.layers) == len(content["activations"]) * 2

    # linear
    content = r"""
    layers:
    - 8
    - linear: 16
    - linear: [32, True]
    - linear: {out_features: 64, bias: True}
    - linear: {features: 64, use_bias: False}
    activations: elu
    """
    content = yaml.safe_load(content)
    modules = _generate_modules(content["layers"], content["activations"])
    _locals = {}
    exec(f'container = nn.Sequential([{", ".join(modules)}])', _globals, _locals)
    container = _locals["container"]
    with capsys.disabled():
        print("\nlinear:", container)
    assert isinstance(container, flax.linen.Sequential)
    assert len(container.layers) == 10

    # conv2d
    content = r"""
    layers:
    - conv2d: [2, 4, [8, 16]]
    - conv2d: {out_channels: 16, kernel_size: 8, stride: [4, 2], padding: valid, bias: True}
    - conv2d: {features: 16, kernel_size: 8, strides: [4, 2], padding: VALID, use_bias: False}
    activations:
    - elu
    """
    content = yaml.safe_load(content)
    modules = _generate_modules(content["layers"], content["activations"])
    _locals = {}
    exec(f'container = nn.Sequential([{", ".join(modules)}])', _globals, _locals)
    container = _locals["container"]
    with capsys.disabled():
        print("\nconv2d:", container)
    assert isinstance(container, flax.linen.Sequential)
    assert len(container.layers) == 6

    # flatten
    content = r"""
    layers:
    - flatten
    - flatten: [2, -2]
    - flatten: {start_dim: 3, end_dim: -3}
    activations:
    - elu
    - elu
    - elu
    """
    content = yaml.safe_load(content)
    modules = _generate_modules(content["layers"], content["activations"])
    _locals = {}
    exec(f'container = nn.Sequential([{", ".join(modules)}])', _globals, _locals)
    container = _locals["container"]
    with capsys.disabled():
        print("\nflatten:", container)
    assert isinstance(container, flax.linen.Sequential)
    assert len(container.layers) == 3

    # non-lazy layers
    content = r"""
    layers:
    - linear: {in_features: STATES, out_features: ONE}
    - conv2d: {in_channels: 3, out_channels: 16, kernel_size: 8}
    """
    content = yaml.safe_load(content)
    modules = _generate_modules(content["layers"], content.get("activations", []))
    _locals = {}
    _globals["self"] = lambda: None
    _globals["self"].num_observations = 5
    exec(f'container = nn.Sequential([{", ".join(modules)}])', _globals, _locals)
    container = _locals["container"]
    with capsys.disabled():
        print("\nnon-lazy:", container)
    assert isinstance(container, flax.linen.Sequential)
    assert len(container.layers) == 2


def test_gaussian_model(capsys):
    device = "cpu"
    observation_space = gym.spaces.Box(np.array([-1] * 5), np.array([1] * 5))
    action_space = gym.spaces.Discrete(2)

    content = r"""
    clip_actions: False
    clip_log_std: True
    initial_log_std: 0
    min_log_std: -20.0
    max_log_std: 2.0
    network:
      - name: net
        input: OBSERVATIONS
        layers:
          - linear: 32
          - linear: [32]
          - linear: {out_features: 32}
        activations: elu
    output: 2 * tanh(ACTIONS)
    """
    content = yaml.safe_load(content)
    # source
    model = gaussian_model(
        observation_space=observation_space, action_space=action_space, device=device, return_source=True, **content
    )
    with capsys.disabled():
        print(model)
    # instance
    model = gaussian_model(
        observation_space=observation_space, action_space=action_space, device=device, return_source=False, **content
    )
    model.init_state_dict("model")
    with capsys.disabled():
        print(model)

    observations = jnp.ones((10, model.num_observations))
    output = model.act({"states": observations})
    assert output[0].shape == (10, 2)


def test_deterministic_model(capsys):
    device = "cpu"
    observation_space = gym.spaces.Box(np.array([-1] * 5), np.array([1] * 5))
    action_space = gym.spaces.Box(np.array([-1] * 3), np.array([1] * 3))

    content = r"""
    clip_actions: True
    network:
      - name: net
        input: OBSERVATIONS
        layers:
          - linear: 32
          - linear: [32]
          - linear: {out_features: 32}
          - linear: {out_features: ACTIONS}
        activations: elu
    output: net / 10
    """
    content = yaml.safe_load(content)
    # source
    model = deterministic_model(
        observation_space=observation_space, action_space=action_space, device=device, return_source=True, **content
    )
    with capsys.disabled():
        print(model)
    # instance
    model = deterministic_model(
        observation_space=observation_space, action_space=action_space, device=device, return_source=False, **content
    )
    model.init_state_dict("model")
    with capsys.disabled():
        print(model)

    observations = jnp.ones((10, model.num_observations))
    output = model.act({"states": observations})
    assert output[0].shape == (10, 3)


def test_categorical_model(capsys):
    device = "cpu"
    observation_space = gym.spaces.Box(np.array([-1] * 5), np.array([1] * 5))
    action_space = gym.spaces.Discrete(2)

    content = r"""
    unnormalized_log_prob: True
    network:
      - name: net
        input: OBSERVATIONS
        layers:
          - linear: 32
          - linear: [32]
          - linear: {out_features: 32}
        activations: elu
    output: ACTIONS
    """
    content = yaml.safe_load(content)
    # source
    model = categorical_model(
        observation_space=observation_space, action_space=action_space, device=device, return_source=True, **content
    )
    with capsys.disabled():
        print(model)
    # instance
    model = categorical_model(
        observation_space=observation_space, action_space=action_space, device=device, return_source=False, **content
    )
    model.init_state_dict("model")
    with capsys.disabled():
        print(model)

    observations = jnp.ones((10, model.num_observations))
    output = model.act({"states": observations})
    assert output[0].shape == (10, 1)


def test_multicategorical_model(capsys):
    device = "cpu"
    observation_space = gym.spaces.Box(np.array([-1] * 5), np.array([1] * 5))
    action_space = gym.spaces.MultiDiscrete([2, 3])

    content = r"""
    unnormalized_log_prob: True
    network:
      - name: net
        input: OBSERVATIONS
        layers:
          - linear: 32
          - linear: [32]
          - linear: {out_features: 32}
        activations: elu
    output: ACTIONS
    """
    content = yaml.safe_load(content)
    # source
    model = multicategorical_model(
        observation_space=observation_space, action_space=action_space, device=device, return_source=True, **content
    )
    with capsys.disabled():
        print(model)
    # instance
    model = multicategorical_model(
        observation_space=observation_space, action_space=action_space, device=device, return_source=False, **content
    )
    model.init_state_dict("model")
    with capsys.disabled():
        print(model)

    observations = jnp.ones((10, model.num_observations))
    output = model.act({"states": observations})
    assert output[0].shape == (10, 2)
