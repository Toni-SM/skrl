from typing import Any, Mapping, Optional, Sequence, Tuple, Union

from enum import Enum
import gym
import gymnasium

import torch
import torch.nn as nn

from skrl.models.torch import Model  # noqa
from skrl.models.torch import CategoricalMixin, DeterministicMixin, GaussianMixin, MultivariateGaussianMixin  # noqa
from skrl.utils.model_instantiators.torch.categorical import categorical_model
from skrl.utils.model_instantiators.torch.deterministic import deterministic_model
from skrl.utils.model_instantiators.torch.gaussian import gaussian_model
from skrl.utils.model_instantiators.torch.multivariate_gaussian import multivariate_gaussian_model


class Shape(Enum):
    """
    Enum to select the shape of the model's inputs and outputs
    """
    ONE = 1
    STATES = 0
    OBSERVATIONS = 0
    ACTIONS = -1
    STATES_ACTIONS = -2


def _get_activation_function(activation: str, as_string: bool = False) -> Union[nn.Module, str]:
    """Get the activation function

    Supported activation functions:

    - "elu"
    - "leaky_relu"
    - "relu"
    - "selu"
    - "sigmoid"
    - "softmax"
    - "softplus"
    - "softsign"
    - "tanh"

    :param activation: activation function name.
                       If activation is an empty string, a placeholder will be returned (``torch.nn.Identity()``)
    :type activation: str
    :param as_string: Whether to return the activation function as string.
    :type as_string: bool

    :raises: ValueError if activation is not a valid activation function

    :return: activation function
    :rtype: nn.Module
    """
    if not activation:
        return "torch.nn.Identity()" if as_string else torch.nn.Identity()
    elif activation == "relu":
        return "torch.nn.ReLU()" if as_string else torch.nn.ReLU()
    elif activation == "tanh":
        return "torch.nn.Tanh()" if as_string else torch.nn.Tanh()
    elif activation == "sigmoid":
        return "torch.nn.Sigmoid()" if as_string else torch.nn.Sigmoid()
    elif activation == "leaky_relu":
        return "torch.nn.LeakyReLU()" if as_string else torch.nn.LeakyReLU()
    elif activation == "elu":
        return "torch.nn.ELU()" if as_string else torch.nn.ELU()
    elif activation == "softplus":
        return "torch.nn.Softplus()" if as_string else torch.nn.Softplus()
    elif activation == "softsign":
        return "torch.nn.Softsign()" if as_string else torch.nn.Softsign()
    elif activation == "selu":
        return "torch.nn.SELU()" if as_string else torch.nn.SELU()
    elif activation == "softmax":
        return "torch.nn.Softmax()" if as_string else torch.nn.Softmax()
    else:
        raise ValueError(f"Unknown activation function: {activation}")

def _get_num_units_by_shape(model: Model, shape: Shape, as_string: bool = False) -> Union[int, str]:
    """Get the number of units in a layer by shape

    :param model: Model to get the number of units for
    :type model: Model
    :param shape: Shape of the layer
    :type shape: Shape or int
    :param as_string: Whether to return the activation function as string.
    :type as_string: bool

    :return: Number of units in the layer
    :rtype: int
    """
    num_units = {Shape.ONE: "1" if as_string else 1,
                 Shape.STATES: "self.num_observations" if as_string else model.num_observations,
                 Shape.ACTIONS: "self.num_actions" if as_string else model.num_actions,
                 Shape.STATES_ACTIONS: "self.num_observations + self.num_actions" if as_string else model.num_observations + model.num_actions}
    try:
        return num_units[shape]
    except:
        return shape

def _generate_sequential(model: Model,
                         input_shape: Shape = Shape.STATES,
                         hiddens: list = [256, 256],
                         hidden_activation: list = ["relu", "relu"],
                         output_shape: Shape = Shape.ACTIONS,
                         output_activation: Union[str, None] = "tanh",
                         output_scale: Optional[int] = None) -> nn.Sequential:
    """Generate a sequential model

    :param model: model to generate sequential model for
    :type model: Model
    :param input_shape: Shape of the input (default: Shape.STATES)
    :type input_shape: Shape, optional
    :param hiddens: Number of hidden units in each hidden layer
    :type hiddens: int or list of ints
    :param hidden_activation: Activation function for each hidden layer (default: "relu").
    :type hidden_activation: list of strings
    :param output_shape: Shape of the output (default: Shape.ACTIONS)
    :type output_shape: Shape, optional
    :param output_activation: Activation function for the output layer (default: "tanh")
    :type output_activation: str or None, optional
    :param output_scale: Scale of the output layer (default: None).
                            If None, the output layer will not be scaled
    :type output_scale: int, optional

    :return: sequential model
    :rtype: nn.Sequential
    """
    modules = []
    if not hiddens:
        modules.append(f"nn.Linear({_get_num_units_by_shape(None, input_shape, as_string=True)}, {_get_num_units_by_shape(None, output_shape, as_string=True)})")
        if output_activation:
            modules.append(_get_activation_function(output_activation, as_string=True))
    for i in range(len(hiddens)):
        # first layer
        if not i:
            modules.append(f"nn.Linear({_get_num_units_by_shape(None, input_shape, as_string=True)}, {hiddens[i]})")
            if hidden_activation[i]:
                modules.append(_get_activation_function(hidden_activation[i], as_string=True))
        # last layer
        if i == len(hiddens) - 1:
            modules.append(f"nn.Linear({hiddens[i]}, {_get_num_units_by_shape(None, output_shape, as_string=True)})")
            if output_activation:
                modules.append(_get_activation_function(output_activation, as_string=True))
        # hidden layers
        else:
            modules.append(f"nn.Linear({hiddens[i]}, {hiddens[i + 1]})")
            if hidden_activation[i]:
                modules.append(_get_activation_function(hidden_activation[i], as_string=True))
    return f'nn.Sequential({", ".join(modules)})'

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
    # network
    net = _generate_sequential(None,
                               parameters[0]["input_shape"],
                               parameters[0]["hiddens"][:-1],
                               parameters[0]["hidden_activation"][:-1],
                               parameters[0]["hiddens"][-1],
                               parameters[0]["hidden_activation"][-1])
    policy_net = _generate_sequential(None,
                                      parameters[0]["hiddens"][-1],
                                      [],
                                      [],
                                      parameters[0]["output_shape"],
                                      parameters[0]["output_activation"])
    value_net = _generate_sequential(None,
                                     parameters[1]["hiddens"][-1],
                                     [],
                                     [],
                                     parameters[1]["output_shape"],
                                     parameters[1]["output_activation"])

    # compute
    if parameters[0]["input_shape"] == Shape.OBSERVATIONS:
        forward = 'self.net(inputs["states"])'
    elif parameters[0]["input_shape"] == Shape.ACTIONS:
        forward = 'self.net(inputs["taken_actions"])'
    elif parameters[0]["input_shape"] == Shape.STATES_ACTIONS:
        forward = 'self.net(torch.cat((inputs["states"], inputs["taken_actions"]), dim=1))'

    if single_forward_pass:
        policy_return = f"""self._shared_output = {forward}
            return <SCALE>self.policy_net(self._shared_output), self.log_std_parameter, {{}}"""
        value_return = f"""shared_output = {forward} if self._shared_output is None else self._shared_output
            self._shared_output = None
            return <SCALE>self.value_net(shared_output), {{}}"""
    else:
        policy_return = f'return <SCALE>self.policy_net{forward}), self.log_std_parameter, {{}}'
        value_return = f'return <SCALE>self.value_net{forward}), {{}}'

    policy_scale = parameters[0]["output_scale"]
    value_scale = parameters[1]["output_scale"]
    policy_return = policy_return.replace("<SCALE>", policy_scale if policy_scale != 1 else "")
    value_return = value_return.replace("<SCALE>", value_scale if value_scale != 1 else "")

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

        self.net = {net}
        self.policy_net = {policy_net}
        self.log_std_parameter = nn.Parameter({parameters[0]["initial_log_std"]} * torch.ones({_get_num_units_by_shape(None, parameters[0]["output_shape"], as_string=True)}))
        self.value_net = {value_net}

    def act(self, inputs, role):
        if role == "{roles[0]}":
            return GaussianMixin.act(self, inputs, role)
        elif role == "{roles[1]}":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role=""):
        if role == "{roles[0]}":
            {policy_return}
        elif role == "{roles[1]}":
            {value_return}
    """
    if return_source:
        return template
    _locals = {}
    exec(template, globals(), _locals)
    return _locals["GaussianDeterministicModel"](observation_space=observation_space,
                                                 action_space=action_space,
                                                 device=device)
