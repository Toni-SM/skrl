import hypothesis
import hypothesis.strategies as st
import pytest

import yaml
import gymnasium as gym

import numpy as np
import torch


from skrl.utils.model_instantiators.torch import Shape 
from skrl.utils.model_instantiators.torch.common import _parse_input, _generate_sequential


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
    content = r"""
layers:
  - linear: 32
  - linear: [32]
  - linear: {out_features: 32}
activations: elu
"""
    content = yaml.safe_load(content)
    with capsys.disabled():
        print()
        print(content)
        _generate_sequential(content["layers"], content["activations"])


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
