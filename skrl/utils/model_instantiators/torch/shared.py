from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import textwrap
import gymnasium

import torch
import torch.nn as nn  # noqa

from skrl.models.torch import Model  # noqa
from skrl.models.torch import CategoricalMixin, DeterministicMixin, GaussianMixin  # noqa
from skrl.utils.model_instantiators.torch.common import convert_deprecated_parameters, generate_containers
from skrl.utils.spaces.torch import unflatten_tensorized_space  # noqa


def shared_model(
    observation_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
    action_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
    device: Optional[Union[str, torch.device]] = None,
    structure: Sequence[str] = ["GaussianMixin", "DeterministicMixin"],
    roles: Sequence[str] = [],
    parameters: Sequence[Mapping[str, Any]] = [],
    single_forward_pass: bool = True,
    return_source: bool = False,
) -> Union[Model, str]:
    """Instantiate a shared model

    :param observation_space: Observation/state space or shape (default: None).
                              If it is not None, the num_observations property will contain the size of that space
    :type observation_space: int, tuple or list of integers, gymnasium.Space or None, optional
    :param action_space: Action space or shape (default: None).
                         If it is not None, the num_actions property will contain the size of that space
    :type action_space: int, tuple or list of integers, gymnasium.Space or None, optional
    :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                   If None, the device will be either ``"cuda"`` if available or ``"cpu"``
    :type device: str or torch.device, optional
    :param structure: Shared model structure (default: Gaussian-Deterministic).
    :type structure: sequence of strings, optional
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
        for parameter in [
            "input_shape",
            "hiddens",
            "hidden_activation",
            "output_shape",
            "output_activation",
            "output_scale",
        ]:
            if parameter in parameters[0]:
                del parameters[0][parameter]
            if parameter in parameters[1]:
                del parameters[1][parameter]

    # checking
    assert (
        len(structure) == len(roles) == len(parameters)
    ), f"Invalid configuration: structures ({len(structure)}), roles ({len(roles)}) and parameters ({len(parameters)}) have different lengths"

    models = [{"class": item} for item in structure]

    # parse model definitions
    for i, model in enumerate(models):
        model["forward"] = []
        model["networks"] = []
        model["containers"], model["output"] = generate_containers(
            parameters[i]["network"], parameters[i]["output"], embed_output=False, indent=1
        )

    # network definitions
    networks_common = []
    forward_common = []
    for container in models[0]["containers"]:
        networks_common.append(f'self.{container["name"]}_container = {container["sequential"]}')
        forward_common.append(f'{container["name"]} = self.{container["name"]}_container({container["input"]})')
    forward_common.insert(
        0, 'taken_actions = unflatten_tensorized_space(self.action_space, inputs.get("taken_actions"))'
    )
    forward_common.insert(0, 'states = unflatten_tensorized_space(self.observation_space, inputs.get("states"))')

    # process output
    if models[0]["output"]["modules"]:
        models[0]["networks"].append(f'self.{roles[0]}_layer = {models[0]["output"]["modules"][0]}')
        models[0]["forward"].append(f'output = self.{roles[0]}_layer({container["name"]})')
    if models[0]["output"]["output"]:
        models[0]["forward"].append(f'output = {models[0]["output"]["output"]}')
    else:
        models[0]["forward"][-1] = models[0]["forward"][-1].replace(f'{container["name"]} =', "output =", 1)

    if models[1]["output"]["modules"]:
        models[1]["networks"].append(f'self.{roles[1]}_layer = {models[1]["output"]["modules"][0]}')
        models[1]["forward"].append(
            f'output = self.{roles[1]}_layer({"shared_output" if single_forward_pass else container["name"]})'
        )
    if models[1]["output"]["output"]:
        models[1]["forward"].append(f'output = {models[1]["output"]["output"]}')
    else:
        models[1]["forward"][-1] = models[1]["forward"][-1].replace(f'{container["name"]} =', "output =", 1)

    # build substitutions and indent content
    networks_common = textwrap.indent("\n".join(networks_common), prefix=" " * 8)[8:]
    models[0]["networks"] = textwrap.indent("\n".join(models[0]["networks"]), prefix=" " * 8)[8:]
    models[1]["networks"] = textwrap.indent("\n".join(models[1]["networks"]), prefix=" " * 8)[8:]

    if single_forward_pass:
        models[1]["forward"] = (
            [
                "if self._shared_output is None:",
            ]
            + ["    " + item for item in forward_common]
            + [
                f'    shared_output = {container["name"]}',
                "else:",
                "    shared_output = self._shared_output",
                "self._shared_output = None",
            ]
            + models[1]["forward"]
        )
        forward_common.append(f'self._shared_output = {container["name"]}')
        forward_common = textwrap.indent("\n".join(forward_common), prefix=" " * 12)[12:]
    else:
        forward_common = textwrap.indent("\n".join(forward_common), prefix=" " * 8)[8:]

    models[0]["forward"] = textwrap.indent("\n".join(models[0]["forward"]), prefix=" " * 12)[12:]
    models[1]["forward"] = textwrap.indent("\n".join(models[1]["forward"]), prefix=" " * 12)[12:]

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
        {models[0]["networks"]}
        {models[1]["networks"]}
        self.log_std_parameter = nn.Parameter({parameters[0]["initial_log_std"]} * torch.ones({models[0]["output"]["size"]}))

    def act(self, inputs, role):
        if role == "{roles[0]}":
            return GaussianMixin.act(self, inputs, role)
        elif role == "{roles[1]}":
            return DeterministicMixin.act(self, inputs, role)
    """
    if single_forward_pass:
        template += f"""
    def compute(self, inputs, role=""):
        if role == "{roles[0]}":
            {forward_common}
            {models[0]["forward"]}
            return output, self.log_std_parameter, {{}}
        elif role == "{roles[1]}":
            {models[1]["forward"]}
            return output, {{}}
    """
    else:
        template += f"""
    def compute(self, inputs, role=""):
        {forward_common}
        if role == "{roles[0]}":
            {models[0]["forward"]}
            return output, self.log_std_parameter, {{}}
        elif role == "{roles[1]}":
            {models[1]["forward"]}
            return output, {{}}
    """
    # return source
    if return_source:
        return template

    # instantiate model
    _locals = {}
    exec(template, globals(), _locals)
    return _locals["GaussianDeterministicModel"](
        observation_space=observation_space, action_space=action_space, device=device
    )
