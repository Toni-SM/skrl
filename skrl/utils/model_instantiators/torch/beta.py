from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import textwrap
import gymnasium

import torch
import torch.nn as nn  # noqa

from skrl.models.torch import BetaMixin  # noqa
from skrl.models.torch import Model
from skrl.utils.model_instantiators.torch.common import one_hot_encoding  # noqa
from skrl.utils.model_instantiators.torch.common import convert_deprecated_parameters, generate_containers
from skrl.utils.spaces.torch import unflatten_tensorized_space  # noqa


def beta_model(
    observation_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
    action_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
    device: Optional[Union[str, torch.device]] = None,
    reduction: str = "sum",
    network: Sequence[Mapping[str, Any]] = [],
    output: Union[str, Sequence[str]] = "",
    return_source: bool = False,
    *args,
    **kwargs,
) -> Union[Model, str]:
    """Instantiate a Beta model

    :param observation_space: Observation/state space or shape (default: None).
                              If it is not None, the num_observations property will contain the size of that space
    :type observation_space: int, tuple or list of integers, gymnasium.Space or None, optional
    :param action_space: Action space or shape (default: None).
                         If it is not None, the num_actions property will contain the size of that space
    :type action_space: int, tuple or list of integers, gymnasium.Space or None, optional
    :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                   If None, the device will be either ``"cuda"`` if available or ``"cpu"``
    :type device: str or torch.device, optional
    :param reduction: Reduction method for returning the log probability density function: (default: ``"sum"``).
                      Supported values are ``"mean"``, ``"sum"``, ``"prod"`` and ``"none"``. If "``none"``, the log probability density
                      function is returned as a tensor of shape ``(num_samples, num_actions)`` instead of ``(num_samples, 1)``
    :type reduction: str, optional
    :param network: Network definition (default: [])
    :type network: list of dict, optional
    :param output: Output expression (default: "")
    :type output: list or str, optional
    :param return_source: Whether to return the source string containing the model class used to
                          instantiate the model rather than the model instance (default: False).
    :type return_source: bool, optional

    :return: Beta model instance or definition source
    :rtype: Model
    """
    # compatibility with versions prior to 1.3.0
    if not network and kwargs:
        network, output = convert_deprecated_parameters(kwargs)

    # parse model definition
    containers, output = generate_containers(network, output, embed_output=True, indent=1)
    # network definitions
    networks = []
    forward: list[str] = []
    for container in containers:
        networks.append(f'self.{container["name"]}_container = {container["sequential"]}')
        forward.append(f'{container["name"]} = self.{container["name"]}_container({container["input"]})')
    # process output
    networks.append(f'self.alpha_layer = nn.LazyLinear(out_features={output["size"]})')
    networks.append(f'self.beta_layer = nn.LazyLinear(out_features={output["size"]})')
    networks.append('self.alpha_activation = torch.nn.Softplus()')
    networks.append('self.beta_activation = torch.nn.Softplus()')
    if output["modules"]:
        networks.append(f'self.custom_output = {output["modules"][0]}')
        forward.append(f'custom_output = self.custom_output({container["name"]})')
        forward.append('alpha = self.alpha_activation(self.alpha_layer(custom_output)) + 1')
        forward.append('beta = self.beta_activation(self.beta_layer(custom_output)) + 1')
    if output["output"]:
        forward.append(f'alpha = self.alpha_activation(self.alpha_layer({container["name"]})) + 1')
        forward.append(f'beta = self.beta_activation(self.beta_layer({container["name"]})) + 1')
    else:
        forward.append(f'alpha = self.alpha_activation(self.alpha_layer({container["name"]})) + 1')
        forward.append(f'beta = self.beta_activation(self.beta_layer({container["name"]})) + 1')

    # build substitutions and indent content
    networks = textwrap.indent("\n".join(networks), prefix=" " * 8)[8:]
    forward = textwrap.indent("\n".join(forward), prefix=" " * 8)[8:]

    template = f"""class BetaModel(BetaMixin, Model):
    def __init__(self, observation_space, action_space, device, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        BetaMixin.__init__(self, reduction)

        {networks}

    def compute(self, inputs, role=""):
        states = unflatten_tensorized_space(self.observation_space, inputs.get("states"))
        taken_actions = unflatten_tensorized_space(self.action_space, inputs.get("taken_actions"))
        {forward}
        return alpha, beta, {{}}
    """
    # return source
    if return_source:
        return template

    # instantiate model
    _locals = {}
    exec(template, globals(), _locals)
    return _locals["BetaModel"](
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        reduction=reduction,
    )
