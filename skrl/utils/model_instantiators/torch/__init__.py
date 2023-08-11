from typing import Any, Mapping, Optional, Sequence, Tuple, Union

from enum import Enum
import gym
import gymnasium

import torch
import torch.nn as nn

from skrl.models.torch import Model  # noqa
from skrl.models.torch import CategoricalMixin, DeterministicMixin, GaussianMixin, MultivariateGaussianMixin  # noqa


__all__ = ["categorical_model", "deterministic_model", "gaussian_model", "multivariate_gaussian_model", "Shape"]


class Shape(Enum):
    """
    Enum to select the shape of the model's inputs and outputs
    """
    ONE = 1
    STATES = 0
    OBSERVATIONS = 0
    ACTIONS = -1
    STATES_ACTIONS = -2


def _get_activation_function(activation: str) -> nn.Module:
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

    :raises: ValueError if activation is not a valid activation function

    :return: activation function
    :rtype: nn.Module
    """
    if not activation:
        return torch.nn.Identity()
    elif activation == "relu":
        return torch.nn.ReLU()
    elif activation == "tanh":
        return torch.nn.Tanh()
    elif activation == "sigmoid":
        return torch.nn.Sigmoid()
    elif activation == "leaky_relu":
        return torch.nn.LeakyReLU()
    elif activation == "elu":
        return torch.nn.ELU()
    elif activation == "softplus":
        return torch.nn.Softplus()
    elif activation == "softsign":
        return torch.nn.Softsign()
    elif activation == "selu":
        return torch.nn.SELU()
    elif activation == "softmax":
        return torch.nn.Softmax()
    else:
        raise ValueError(f"Unknown activation function: {activation}")

def _get_num_units_by_shape(model: Model, shape: Shape) -> int:
    """Get the number of units in a layer by shape

    :param model: Model to get the number of units for
    :type model: Model
    :param shape: Shape of the layer
    :type shape: Shape or int

    :return: Number of units in the layer
    :rtype: int
    """
    num_units = {Shape.ONE: 1,
                 Shape.STATES: model.num_observations,
                 Shape.ACTIONS: model.num_actions,
                 Shape.STATES_ACTIONS: model.num_observations + model.num_actions}
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
    # input layer
    input_layer = [nn.Linear(_get_num_units_by_shape(model, input_shape), hiddens[0])]
    # hidden layers
    hidden_layers = []
    for i in range(len(hiddens) - 1):
        hidden_layers.append(_get_activation_function(hidden_activation[i]))
        hidden_layers.append(nn.Linear(hiddens[i], hiddens[i + 1]))
    hidden_layers.append(_get_activation_function(hidden_activation[-1]))
    # output layer
    output_layer = [nn.Linear(hiddens[-1], _get_num_units_by_shape(model, output_shape))]
    if output_activation is not None:
        output_layer.append(_get_activation_function(output_activation))

    return nn.Sequential(*input_layer, *hidden_layers, *output_layer)

def gaussian_model(observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                   action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                   device: Optional[Union[str, torch.device]] = None,
                   clip_actions: bool = False,
                   clip_log_std: bool = True,
                   min_log_std: float = -20,
                   max_log_std: float = 2,
                   initial_log_std: float = 0,
                   input_shape: Shape = Shape.STATES,
                   hiddens: list = [256, 256],
                   hidden_activation: list = ["relu", "relu"],
                   output_shape: Shape = Shape.ACTIONS,
                   output_activation: Optional[str] = "tanh",
                   output_scale: float = 1.0) -> Model:
    """Instantiate a Gaussian model

    :param observation_space: Observation/state space or shape (default: None).
                              If it is not None, the num_observations property will contain the size of that space
    :type observation_space: int, tuple or list of integers, gym.Space, gymnasium.Space or None, optional
    :param action_space: Action space or shape (default: None).
                         If it is not None, the num_actions property will contain the size of that space
    :type action_space: int, tuple or list of integers, gym.Space, gymnasium.Space or None, optional
    :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                   If None, the device will be either ``"cuda"`` if available or ``"cpu"``
    :type device: str or torch.device, optional
    :param clip_actions: Flag to indicate whether the actions should be clipped (default: False)
    :type clip_actions: bool, optional
    :param clip_log_std: Flag to indicate whether the log standard deviations should be clipped (default: True)
    :type clip_log_std: bool, optional
    :param min_log_std: Minimum value of the log standard deviation (default: -20)
    :type min_log_std: float, optional
    :param max_log_std: Maximum value of the log standard deviation (default: 2)
    :type max_log_std: float, optional
    :param initial_log_std: Initial value for the log standard deviation (default: 0)
    :type initial_log_std: float, optional
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
    :param output_scale: Scale of the output layer (default: 1.0).
                         If None, the output layer will not be scaled
    :type output_scale: float, optional

    :return: Gaussian model instance
    :rtype: Model
    """
    class GaussianModel(GaussianMixin, Model):
        def __init__(self, observation_space, action_space, device, clip_actions,
                     clip_log_std, min_log_std, max_log_std, reduction="sum"):
            Model.__init__(self, observation_space, action_space, device)
            GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

            self.instantiator_output_scale = metadata["output_scale"]
            self.instantiator_input_type = metadata["input_shape"].value

            self.net = _generate_sequential(model=self,
                                            input_shape=metadata["input_shape"],
                                            hiddens=metadata["hiddens"],
                                            hidden_activation=metadata["hidden_activation"],
                                            output_shape=metadata["output_shape"],
                                            output_activation=metadata["output_activation"],
                                            output_scale=metadata["output_scale"])
            self.log_std_parameter = nn.Parameter(metadata["initial_log_std"] \
                                                  * torch.ones(_get_num_units_by_shape(self, metadata["output_shape"])))

        def compute(self, inputs, role=""):
            if self.instantiator_input_type == 0:
                output = self.net(inputs["states"])
            elif self.instantiator_input_type == -1:
                output = self.net(inputs["taken_actions"])
            elif self.instantiator_input_type == -2:
                output = self.net(torch.cat((inputs["states"], inputs["taken_actions"]), dim=1))

            return output * self.instantiator_output_scale, self.log_std_parameter, {}

    metadata = {"input_shape": input_shape,
                "hiddens": hiddens,
                "hidden_activation": hidden_activation,
                "output_shape": output_shape,
                "output_activation": output_activation,
                "output_scale": output_scale,
                "initial_log_std": initial_log_std}

    return GaussianModel(observation_space=observation_space,
                         action_space=action_space,
                         device=device,
                         clip_actions=clip_actions,
                         clip_log_std=clip_log_std,
                         min_log_std=min_log_std,
                         max_log_std=max_log_std)

def multivariate_gaussian_model(observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                                action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                                device: Optional[Union[str, torch.device]] = None,
                                clip_actions: bool = False,
                                clip_log_std: bool = True,
                                min_log_std: float = -20,
                                max_log_std: float = 2,
                                initial_log_std: float = 0,
                                input_shape: Shape = Shape.STATES,
                                hiddens: list = [256, 256],
                                hidden_activation: list = ["relu", "relu"],
                                output_shape: Shape = Shape.ACTIONS,
                                output_activation: Optional[str] = "tanh",
                                output_scale: float = 1.0) -> Model:
    """Instantiate a multivariate Gaussian model

    :param observation_space: Observation/state space or shape (default: None).
                              If it is not None, the num_observations property will contain the size of that space
    :type observation_space: int, tuple or list of integers, gym.Space, gymnasium.Space or None, optional
    :param action_space: Action space or shape (default: None).
                         If it is not None, the num_actions property will contain the size of that space
    :type action_space: int, tuple or list of integers, gym.Space, gymnasium.Space or None, optional
    :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                   If None, the device will be either ``"cuda"`` if available or ``"cpu"``
    :type device: str or torch.device, optional
    :param clip_actions: Flag to indicate whether the actions should be clipped (default: False)
    :type clip_actions: bool, optional
    :param clip_log_std: Flag to indicate whether the log standard deviations should be clipped (default: True)
    :type clip_log_std: bool, optional
    :param min_log_std: Minimum value of the log standard deviation (default: -20)
    :type min_log_std: float, optional
    :param max_log_std: Maximum value of the log standard deviation (default: 2)
    :type max_log_std: float, optional
    :param initial_log_std: Initial value for the log standard deviation (default: 0)
    :type initial_log_std: float, optional
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
    :param output_scale: Scale of the output layer (default: 1.0).
                         If None, the output layer will not be scaled
    :type output_scale: float, optional

    :return: Multivariate Gaussian model instance
    :rtype: Model
    """
    class MultivariateGaussianModel(MultivariateGaussianMixin, Model):
        def __init__(self, observation_space, action_space, device, clip_actions,
                     clip_log_std, min_log_std, max_log_std):
            Model.__init__(self, observation_space, action_space, device)
            MultivariateGaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

            self.instantiator_output_scale = metadata["output_scale"]
            self.instantiator_input_type = metadata["input_shape"].value

            self.net = _generate_sequential(model=self,
                                            input_shape=metadata["input_shape"],
                                            hiddens=metadata["hiddens"],
                                            hidden_activation=metadata["hidden_activation"],
                                            output_shape=metadata["output_shape"],
                                            output_activation=metadata["output_activation"],
                                            output_scale=metadata["output_scale"])
            self.log_std_parameter = nn.Parameter(metadata["initial_log_std"] \
                                                  * torch.ones(_get_num_units_by_shape(self, metadata["output_shape"])))

        def compute(self, inputs, role=""):
            if self.instantiator_input_type == 0:
                output = self.net(inputs["states"])
            elif self.instantiator_input_type == -1:
                output = self.net(inputs["taken_actions"])
            elif self.instantiator_input_type == -2:
                output = self.net(torch.cat((inputs["states"], inputs["taken_actions"]), dim=1))

            return output * self.instantiator_output_scale, self.log_std_parameter, {}

    metadata = {"input_shape": input_shape,
                "hiddens": hiddens,
                "hidden_activation": hidden_activation,
                "output_shape": output_shape,
                "output_activation": output_activation,
                "output_scale": output_scale,
                "initial_log_std": initial_log_std}

    return MultivariateGaussianModel(observation_space=observation_space,
                                     action_space=action_space,
                                     device=device,
                                     clip_actions=clip_actions,
                                     clip_log_std=clip_log_std,
                                     min_log_std=min_log_std,
                                     max_log_std=max_log_std)

def deterministic_model(observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                        action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                        device: Optional[Union[str, torch.device]] = None,
                        clip_actions: bool = False,
                        input_shape: Shape = Shape.STATES,
                        hiddens: list = [256, 256],
                        hidden_activation: list = ["relu", "relu"],
                        output_shape: Shape = Shape.ACTIONS,
                        output_activation: Optional[str] = "tanh",
                        output_scale: float = 1.0) -> Model:
    """Instantiate a deterministic model

    :param observation_space: Observation/state space or shape (default: None).
                              If it is not None, the num_observations property will contain the size of that space
    :type observation_space: int, tuple or list of integers, gym.Space, gymnasium.Space or None, optional
    :param action_space: Action space or shape (default: None).
                         If it is not None, the num_actions property will contain the size of that space
    :type action_space: int, tuple or list of integers, gym.Space, gymnasium.Space or None, optional
    :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                   If None, the device will be either ``"cuda"`` if available or ``"cpu"``
    :type device: str or torch.device, optional
    :param clip_actions: Flag to indicate whether the actions should be clipped to the action space (default: False)
    :type clip_actions: bool, optional
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
    :param output_scale: Scale of the output layer (default: 1.0).
                         If None, the output layer will not be scaled
    :type output_scale: float, optional

    :return: Deterministic model instance
    :rtype: Model
    """
    class DeterministicModel(DeterministicMixin, Model):
        def __init__(self, observation_space, action_space, device, clip_actions):
            Model.__init__(self, observation_space, action_space, device)
            DeterministicMixin.__init__(self, clip_actions)

            self.instantiator_output_scale = metadata["output_scale"]
            self.instantiator_input_type = metadata["input_shape"].value

            self.net = _generate_sequential(model=self,
                                            input_shape=metadata["input_shape"],
                                            hiddens=metadata["hiddens"],
                                            hidden_activation=metadata["hidden_activation"],
                                            output_shape=metadata["output_shape"],
                                            output_activation=metadata["output_activation"],
                                            output_scale=metadata["output_scale"])

        def compute(self, inputs, role=""):
            if self.instantiator_input_type == 0:
                output = self.net(inputs["states"])
            elif self.instantiator_input_type == -1:
                output = self.net(inputs["taken_actions"])
            elif self.instantiator_input_type == -2:
                output = self.net(torch.cat((inputs["states"], inputs["taken_actions"]), dim=1))

            return output * self.instantiator_output_scale, {}

    metadata = {"input_shape": input_shape,
                "hiddens": hiddens,
                "hidden_activation": hidden_activation,
                "output_shape": output_shape,
                "output_activation": output_activation,
                "output_scale": output_scale}

    return DeterministicModel(observation_space=observation_space,
                              action_space=action_space,
                              device=device,
                              clip_actions=clip_actions)

def categorical_model(observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                      action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                      device: Optional[Union[str, torch.device]] = None,
                      unnormalized_log_prob: bool = True,
                      input_shape: Shape = Shape.STATES,
                      hiddens: list = [256, 256],
                      hidden_activation: list = ["relu", "relu"],
                      output_shape: Shape = Shape.ACTIONS,
                      output_activation: Optional[str] = None) -> Model:
    """Instantiate a categorical model

    :param observation_space: Observation/state space or shape (default: None).
                              If it is not None, the num_observations property will contain the size of that space
    :type observation_space: int, tuple or list of integers, gym.Space, gymnasium.Space or None, optional
    :param action_space: Action space or shape (default: None).
                         If it is not None, the num_actions property will contain the size of that space
    :type action_space: int, tuple or list of integers, gym.Space, gymnasium.Space or None, optional
    :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                   If None, the device will be either ``"cuda"`` if available or ``"cpu"``
    :type device: str or torch.device, optional
    :param unnormalized_log_prob: Flag to indicate how to be interpreted the model's output (default: True).
                                  If True, the model's output is interpreted as unnormalized log probabilities
                                  (it can be any real number), otherwise as normalized probabilities
                                  (the output must be non-negative, finite and have a non-zero sum)
    :type unnormalized_log_prob: bool, optional
    :param input_shape: Shape of the input (default: Shape.STATES)
    :type input_shape: Shape, optional
    :param hiddens: Number of hidden units in each hidden layer
    :type hiddens: int or list of ints
    :param hidden_activation: Activation function for each hidden layer (default: "relu").
    :type hidden_activation: list of strings
    :param output_shape: Shape of the output (default: Shape.ACTIONS)
    :type output_shape: Shape, optional
    :param output_activation: Activation function for the output layer (default: None)
    :type output_activation: str or None, optional

    :return: Categorical model instance
    :rtype: Model
    """
    class CategoricalModel(CategoricalMixin, Model):
        def __init__(self, observation_space, action_space, device, unnormalized_log_prob):
            Model.__init__(self, observation_space, action_space, device)
            CategoricalMixin.__init__(self, unnormalized_log_prob)

            self.instantiator_input_type = metadata["input_shape"].value

            self.net = _generate_sequential(model=self,
                                            input_shape=metadata["input_shape"],
                                            hiddens=metadata["hiddens"],
                                            hidden_activation=metadata["hidden_activation"],
                                            output_shape=metadata["output_shape"],
                                            output_activation=metadata["output_activation"])

        def compute(self, inputs, role=""):
            if self.instantiator_input_type == 0:
                output = self.net(inputs["states"])
            elif self.instantiator_input_type == -1:
                output = self.net(inputs["taken_actions"])
            elif self.instantiator_input_type == -2:
                output = self.net(torch.cat((inputs["states"], inputs["taken_actions"]), dim=1))

            return output, {}

    metadata = {"input_shape": input_shape,
                "hiddens": hiddens,
                "hidden_activation": hidden_activation,
                "output_shape": output_shape,
                "output_activation": output_activation}

    return CategoricalModel(observation_space=observation_space,
                            action_space=action_space,
                            device=device,
                            unnormalized_log_prob=unnormalized_log_prob)

def shared_model(observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                 action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                 device: Optional[Union[str, torch.device]] = None,
                 structure: str = "",
                 roles: Sequence[str] = [],
                 parameters: Sequence[Mapping[str, Any]] = []) -> Model:
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

    :return: Shared model instance
    :rtype: Model
    """
    class GaussianDeterministicModel(GaussianMixin, DeterministicMixin, Model):
        def __init__(self, observation_space, action_space, device, roles, metadata):
            Model.__init__(self, observation_space, action_space, device)
            GaussianMixin.__init__(self,
                                   clip_actions=metadata[0]["clip_actions"],
                                   clip_log_std=metadata[0]["clip_log_std"],
                                   min_log_std=metadata[0]["min_log_std"],
                                   max_log_std=metadata[0]["max_log_std"],
                                   role=roles[0])
            DeterministicMixin.__init__(self, clip_actions=metadata[1]["clip_actions"], role=roles[1])

            self._roles = roles
            self.instantiator_input_type = metadata[0]["input_shape"].value
            self.instantiator_output_scales = [m["output_scale"] for m in metadata]

            # shared layers/network
            self.net = _generate_sequential(model=self,
                                            input_shape=metadata[0]["input_shape"],
                                            hiddens=metadata[0]["hiddens"][:-1],
                                            hidden_activation=metadata[0]["hidden_activation"][:-1],
                                            output_shape=metadata[0]["hiddens"][-1],
                                            output_activation=metadata[0]["hidden_activation"][-1])

            # separated layers ("policy")
            mean_layers = [nn.Linear(metadata[0]["hiddens"][-1], _get_num_units_by_shape(self, metadata[0]["output_shape"]))]
            if metadata[0]["output_activation"] is not None:
                mean_layers.append(_get_activation_function(metadata[0]["output_activation"]))
            self.mean_net = nn.Sequential(*mean_layers)

            self.log_std_parameter = nn.Parameter(metadata[0]["initial_log_std"] \
                                                  * torch.ones(_get_num_units_by_shape(self, metadata[0]["output_shape"])))

            # separated layer ("value")
            value_layers = [nn.Linear(metadata[1]["hiddens"][-1], _get_num_units_by_shape(self, metadata[1]["output_shape"]))]
            if metadata[1]["output_activation"] is not None:
                value_layers.append(_get_activation_function(metadata[1]["output_activation"]))
            self.value_net = nn.Sequential(*value_layers)

        def act(self, inputs, role):
            if role == self._roles[0]:
                return GaussianMixin.act(self, inputs, role)
            elif role == self._roles[1]:
                return DeterministicMixin.act(self, inputs, role)

        def compute(self, inputs, role):
            if self.instantiator_input_type == 0:
                output = self.net(inputs["states"])
            elif self.instantiator_input_type == -1:
                output = self.net(inputs["taken_actions"])
            elif self.instantiator_input_type == -2:
                output = self.net(torch.cat((inputs["states"], inputs["taken_actions"]), dim=1))

            if role == self._roles[0]:
                return self.instantiator_output_scales[0] * self.mean_net(output), self.log_std_parameter, {}
            elif role == self._roles[1]:
                return self.instantiator_output_scales[1] * self.value_net(output), {}

    # TODO: define the model using the specified structure

    return GaussianDeterministicModel(observation_space=observation_space,
                                        action_space=action_space,
                                        device=device,
                                        roles=roles,
                                        metadata=parameters)
