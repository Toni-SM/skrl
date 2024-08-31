import hypothesis
import hypothesis.strategies as st
import pytest

import gymnasium as gym
import yaml

import numpy as np
import torch

from skrl.utils.model_instantiators.torch.common import Shape, _generate_sequential, _parse_input


def test_parse_input(capsys):
    # check for Shape enum
    for input in [Shape.STATES, Shape.OBSERVATIONS, Shape.ACTIONS, Shape.STATES_ACTIONS, Shape.OBSERVATIONS_ACTIONS]:
        # Shape enum with/without class
        output = _parse_input(str(input))
        output_1 = _parse_input(str(input).replace("Shape.", ""))
        assert output == output_1, f"'{output}' != '{output_1}'"
        # Shape is not in output
        for item in ["Shape", "STATES", "OBSERVATIONS", "ACTIONS", "STATES_ACTIONS", "OBSERVATIONS_ACTIONS"]:
            assert item not in output, f"'{item}' in '{output}'"
    # Mixed operation
    input = 'Shape.OBSERVATIONS["joint"] + concatenate([net * ACTIONS[:, -3:]])'
    statement = 'inputs["states"]["joint"] + torch.cat([net * inputs["taken_actions"][:, -3:]], dim=1)'
    output = _parse_input(str(input))
    assert output.replace("'", '"') == statement, f"'{output}' != '{statement}'"

def test_generate_sequential(capsys):
    _globals = {"nn": torch.nn}

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
    modules = _generate_sequential(content["layers"], content["activations"])
    _locals = {}
    exec(f'container = nn.Sequential({", ".join(modules)})', _globals, _locals)
    container = _locals["container"]
    with capsys.disabled():
        print("\nlinear:", container)
    assert isinstance(container, torch.nn.Sequential)
    assert len(container) == 10

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
    modules = _generate_sequential(content["layers"], content["activations"])
    _locals = {}
    exec(f'container = nn.Sequential({", ".join(modules)})', _globals, _locals)
    container = _locals["container"]
    with capsys.disabled():
        print("\nconv2d:", container)
    assert isinstance(container, torch.nn.Sequential)
    assert len(container) == 6

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
    modules = _generate_sequential(content["layers"], content["activations"])
    _locals = {}
    exec(f'container = nn.Sequential({", ".join(modules)})', _globals, _locals)
    container = _locals["container"]
    with capsys.disabled():
        print("\nflatten:", container)
    assert isinstance(container, torch.nn.Sequential)
    assert len(container) == 3

    # non-lazy layers
    content = r"""
layers:
  - linear: {in_features: STATES, out_features: Shape.ONE}
  - conv2d: {in_channels: 3, out_channels: 16, kernel_size: 8}
"""
    content = yaml.safe_load(content)
    modules = _generate_sequential(content["layers"], content.get("activations", []))
    _locals = {}
    _globals["self"] = lambda: None
    _globals["self"].num_observations = 5
    exec(f'container = nn.Sequential({", ".join(modules)})', _globals, _locals)
    container = _locals["container"]
    with capsys.disabled():
        print("\nnon-lazy:", container)
    assert isinstance(container, torch.nn.Sequential)
    assert len(container) == 2

# def test_gaussian_model(capsys):
#     device = "cpu"
#     observation_space = gym.spaces.Box(np.array([-1] * 5), np.array([1] * 5))
#     action_space = gym.spaces.Discrete(2)

#     cfg = r"""
# clip_actions: True
# clip_log_std: True
# initial_log_std: 0
# min_log_std: -20.0
# max_log_std: 2.0
# network:
#   - name: net
#     input: Shape.OBSERVATIONS
#     layers:
#       - linear: 32
#       - linear: [32]
#       - linear: {out_features: 32}
#     activations: elu
# output_shape: "Shape.ACTIONS"
# output_activation: "tanh"
# output_scale: 1.0
# """

#     values = yaml.safe_load(cfg)
#     with capsys.disabled():
#         import pprint
#         pprint.pprint(values)

#         # TODO: randomize all parameters
#         model = gaussian_model(observation_space=observation_space,
#                             action_space=action_space,
#                             device=device,
#                             return_source=True,
#                             **values)
#     # model.to(device=device)
#     # with capsys.disabled():
#     #     print(model)

#     # observations = torch.ones((10, model.num_observations), device=device)
#     # output = model.act({"states": observations})
#     # # assert output[0].shape == (10, 1)
