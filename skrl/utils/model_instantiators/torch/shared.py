from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import textwrap
import gym
import gymnasium

import torch
import torch.nn as nn  # noqa

from skrl.models.torch import Model  # noqa
from skrl.models.torch import DeterministicMixin, GaussianMixin  # noqa
from skrl.utils.model_instantiators.torch.common import convert_deprecated_parameters, generate_containers


def shared_model(observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                 action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                 device: Optional[Union[str, torch.device]] = None,
                 structure: str = "",
                 roles: Sequence[str] = [],
                 parameters: Sequence[Mapping[str, Any]] = [],
                 single_forward_pass: bool = True,
                 return_source: bool = False) -> Union[Model, str]:
    """Instantiate a shared model

    :param observation_space: Observation/state space or shape (default: None).
                              If it is not None, the num_observations property will contain the size of that space
    :type observation_space: int, tuple or list of integers, gym.Space, gymnasium.Space or None, optional
    :param action_space: Action space or shape (default: None).
                         If it is not None, the num_actions property will contain the size of that space
    :type action_space: int, tuple or list of integers, gym.Space, gymnasium.Space or None, optional
    :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                   If None, the device will be either ``"cuda"`` if available or ``"cpu"``
    :type device: str or torch.device, optional
    :param structure: Shared model structure (default: ``""``).
                      Note: this parameter is ignored for the moment
    :type structure: str, optional
    :param roles: Organized list of model roles (default: ``[]``)
    :type roles: sequence of strings, optional
    :param parameters: Organized list of model instantiator parameters (default: ``[]``)
    :type parameters: sequence of dict, optional
    :param single_forward_pass: Whether to perform a single forward-pass for the shared layers/network (default: ``True``)
    :type single_forward_pass: bool
    :param return_source: Whether to return the source string containing the model class used to
                          instantiate the model rather than the model instance (default: False).
    :type return_source: bool, optional

    :return: Shared model instance or definition source
    :rtype: Model
    """
    # compatibility with versions prior to 1.3.0
    if not "network" in parameters[0]:
        parameters[0]["network"], parameters[0]["output"] = convert_deprecated_parameters(parameters[0])
        parameters[1]["network"], parameters[1]["output"] = convert_deprecated_parameters(parameters[1])
        # delete deprecated parameters
        for parameter in ["input_shape", "hiddens", "hidden_activation", "output_shape", "output_activation", "output_scale"]:
            if parameter in parameters[0]:
                del parameters[0][parameter]
            if parameter in parameters[1]:
                del parameters[1][parameter]

    # parse model definitions
    containers_gaussian, output_gaussian = generate_containers(parameters[0]["network"], parameters[0]["output"], embed_output=False, indent=1)
    containers_deterministic, output_deterministic = generate_containers(parameters[1]["network"], parameters[1]["output"], embed_output=False, indent=1)

    # network definitions
    networks_common = []
    forward_common = []
    for container in containers_gaussian:
        networks_common.append(f'self.{container["name"]}_container = {container["sequential"]}')
        forward_common.append(f'{container["name"]} = self.{container["name"]}_container({container["input"]})')

    # process output
    networks_gaussian = []
    forward_gaussian = []
    if output_gaussian["modules"]:
        networks_gaussian.append(f'self.{roles[0]}_layer = {output_gaussian["modules"][0]}')
        forward_gaussian.append(f'output = self.{roles[0]}_layer({container["name"]})')
    if output_gaussian["output"]:
        forward_gaussian.append(f'output = {output_gaussian["output"]}')
    else:
        forward_gaussian[-1] = forward_gaussian[-1].replace(f'{container["name"]} =', "output =", 1)

    networks_deterministic = []
    forward_deterministic = []
    if output_deterministic["modules"]:
        networks_deterministic.append(f'self.{roles[1]}_layer = {output_deterministic["modules"][0]}')
        forward_deterministic.append(f'output = self.{roles[1]}_layer({"shared_output" if single_forward_pass else container["name"]})')
    if output_deterministic["output"]:
        forward_deterministic.append(f'output = {output_deterministic["output"]}')
    else:
        forward_deterministic[-1] = forward_deterministic[-1].replace(f'{container["name"]} =', "output =", 1)

    # build substitutions and indent content
    networks_common = textwrap.indent("\n".join(networks_common), prefix=" " * 8)[8:]
    networks_gaussian = textwrap.indent("\n".join(networks_gaussian), prefix=" " * 8)[8:]
    networks_deterministic = textwrap.indent("\n".join(networks_deterministic), prefix=" " * 8)[8:]

    if single_forward_pass:
        forward_deterministic = [
            "if self._shared_output is None:",
        ] + ["    " + item for item in forward_common] + [
            f'    shared_output = {container["name"]}',
            "else:",
            "    shared_output = self._shared_output",
            "self._shared_output = None",
        ] + forward_deterministic
        forward_common.append(f'self._shared_output = {container["name"]}')
        forward_common = textwrap.indent("\n".join(forward_common), prefix=" " * 12)[12:]
    else:
        forward_common = textwrap.indent("\n".join(forward_common), prefix=" " * 8)[8:]

    forward_gaussian = textwrap.indent("\n".join(forward_gaussian), prefix=" " * 12)[12:]
    forward_deterministic = textwrap.indent("\n".join(forward_deterministic), prefix=" " * 12)[12:]

    template = f"""class GaussianDeterministicModel(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self,
                               clip_actions={parameters[0]["clip_actions"]},
                               clip_log_std={parameters[0]["clip_log_std"]},
                               min_log_std={parameters[0]["min_log_std"]},
                               max_log_std={parameters[0]["max_log_std"]},
                               role="{roles[0]}")
        DeterministicMixin.__init__(self, clip_actions={parameters[1]["clip_actions"]}, role="{roles[1]}")

        {networks_common}
        {networks_gaussian}
        {networks_deterministic}
        self.log_std_parameter = nn.Parameter({parameters[0]["initial_log_std"]} * torch.ones({output_gaussian["size"]}))

    def act(self, inputs, role):
        if role == "{roles[0]}":
            return GaussianMixin.act(self, inputs, role)
        elif role == "{roles[1]}":
            return DeterministicMixin.act(self, inputs, role)
    """
    if single_forward_pass:
        template +=f"""
    def compute(self, inputs, role=""):
        if role == "{roles[0]}":
            {forward_common}
            {forward_gaussian}
            return output, self.log_std_parameter, {{}}
        elif role == "{roles[1]}":
            {forward_deterministic}
            return output, {{}}
    """
    else:
        template +=f"""
    def compute(self, inputs, role=""):
        {forward_common}
        if role == "{roles[0]}":
            {forward_gaussian}
            return output, self.log_std_parameter, {{}}
        elif role == "{roles[1]}":
            {forward_deterministic}
            return output, {{}}
    """
    # return source
    if return_source:
        return template

    # instantiate model
    _locals = {}
    exec(template, globals(), _locals)
    return _locals["GaussianDeterministicModel"](observation_space=observation_space,
                                                 action_space=action_space,
                                                 device=device)
