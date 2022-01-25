from typing import Union, Tuple

import gym
from enum import Enum

import torch
import torch.nn as nn

from ..models.torch import Model
from ..models.torch import GaussianModel
from ..models.torch import CategoricalModel
from ..models.torch import DeterministicModel


class Shape(Enum):
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

    :param activation: activation function name
    :type activation: str

    :raises: ValueError if activation is not a valid activation function

    :return: activation function
    :rtype: nn.Module
    """
    if activation == "relu":
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
        raise ValueError("Unknown activation function: {}".format(activation))

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
    return num_units[shape]

def _generate_sequential(model: Model, 
                         input_shape: Shape = Shape.STATES, 
                         hiddens: list = [256, 256], 
                         hidden_activation: list = ["relu", "relu"], 
                         output_shape: Shape = Shape.ACTIONS, 
                         output_activation: Union[str, None] = "tanh", 
                         output_scale: int = None,) -> nn.Sequential:
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
    for i in range(1, len(hiddens) - 1):
        hidden_layers.append(_get_activation_function(hidden_activation[i - 1]))
        hidden_layers.append(nn.Linear(hiddens[i], hiddens[i + 1]))
    hidden_layers.append(_get_activation_function(hidden_activation[-1]))
    # output layer
    output_layer = [nn.Linear(hiddens[-1], _get_num_units_by_shape(model, output_shape))]
    if output_activation is not None:
        output_layer.append(_get_activation_function(output_activation))
    
    return nn.Sequential(*input_layer, *hidden_layers, *output_layer)

def gaussian_model(observation_space: Union[int, Tuple[int], gym.Space, None] = None, 
                   action_space: Union[int, Tuple[int], gym.Space, None] = None,
                   device: Union[str, torch.device] = "cuda:0", 
                   clip_actions: bool = False, 
                   clip_log_std: bool = True, 
                   min_log_std: float = -20, 
                   max_log_std: float = 2, 
                   input_shape: Shape = Shape.STATES, 
                   hiddens: list = [256, 256], 
                   hidden_activation: list = ["relu", "relu"], 
                   output_shape: Shape = Shape.ACTIONS, 
                   output_activation: Union[str, None] = "tanh", 
                   output_scale: float = 1.0) -> GaussianModel: 
    """Instantiate a GaussianModel model

    :param observation_space: Observation/state space or shape (default: None).
                              If it is not None, the num_observations property will contain the size of that space
    :type observation_space: int, tuple or list of integers, gym.Space or None, optional
    :param action_space: Action space or shape (default: None).
                         If it is not None, the num_actions property will contain the size of that space
    :type action_space: int, tuple or list of integers, gym.Space or None, optional
    :param device: Device on which the model will be trained (default: "cuda:0")
    :type device: str or torch.device, optional
    :param clip_actions: Flag to indicate whether the actions should be clipped (default: False)
    :type clip_actions: bool, optional
    :param clip_log_std: Flag to indicate whether the log standard deviations should be clipped (default: True)
    :type clip_log_std: bool, optional
    :param min_log_std: Minimum value of the log standard deviation (default: -20)
    :type min_log_std: float, optional
    :param max_log_std: Maximum value of the log standard deviation (default: 2)
    :type max_log_std: float, optional
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

    :return: GaussianModel instance
    :rtype: GaussianModel
    """
    model = GaussianModel(observation_space=observation_space,
                          action_space=action_space, 
                          device=device, 
                          clip_actions=clip_actions, 
                          clip_log_std=clip_log_std, 
                          min_log_std=min_log_std,
                          max_log_std=max_log_std)
    
    model._instantiator_net = _generate_sequential(model=model,
                                                   input_shape=input_shape,
                                                   hiddens=hiddens,
                                                   hidden_activation=hidden_activation,
                                                   output_shape=output_shape,
                                                   output_activation=output_activation,
                                                   output_scale=output_scale)
    model._instantiator_output_scale = output_scale
    model._instantiator_input_type = input_shape.value
    model._instantiator_parameter = nn.Parameter(torch.zeros(_get_num_units_by_shape(model, output_shape)))

    return model
    
def deterministic_model(observation_space: Union[int, Tuple[int], gym.Space, None] = None, 
                        action_space: Union[int, Tuple[int], gym.Space, None] = None, 
                        device: Union[str, torch.device] = "cuda:0", 
                        clip_actions: bool = False, 
                        input_shape: Shape = Shape.STATES, 
                        hiddens: list = [256, 256], 
                        hidden_activation: list = ["relu", "relu"], 
                        output_shape: Shape = Shape.ACTIONS, 
                        output_activation: Union[str, None] = "tanh", 
                        output_scale: float = 1.0) -> DeterministicModel:
    """Instantiate a DeterministicModel model

    :param observation_space: Observation/state space or shape (default: None).
                              If it is not None, the num_observations property will contain the size of that space
    :type observation_space: int, tuple or list of integers, gym.Space or None, optional
    :param action_space: Action space or shape (default: None).
                            If it is not None, the num_actions property will contain the size of that space
    :type action_space: int, tuple or list of integers, gym.Space or None, optional
    :param device: Device on which a torch tensor is or will be allocated (default: "cuda:0")
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

    :return: DeterministicModel instance
    :rtype: DeterministicModel
    """
    model = DeterministicModel(observation_space=observation_space,
                               action_space=action_space, 
                               device=device, 
                               clip_actions=clip_actions)
    
    model._instantiator_net = _generate_sequential(model=model,
                                                   input_shape=input_shape,
                                                   hiddens=hiddens,
                                                   hidden_activation=hidden_activation,
                                                   output_shape=output_shape,
                                                   output_activation=output_activation,
                                                   output_scale=output_scale)
    model._instantiator_output_scale = output_scale
    model._instantiator_input_type = input_shape.value

    return model
    
def categorical_model(observation_space: Union[int, Tuple[int], gym.Space, None] = None, 
                      action_space: Union[int, Tuple[int], gym.Space, None] = None, 
                      device: Union[str, torch.device] = "cuda:0", 
                      unnormalized_log_prob: bool = False, 
                      input_shape: Shape = Shape.STATES, 
                      hiddens: list = [256, 256], 
                      hidden_activation: list = ["relu", "relu"], 
                      output_shape: Shape = Shape.ACTIONS, 
                      output_activation: Union[str, None] = None) -> CategoricalModel:
    """Instantiate a CategoricalModel model

    :param observation_space: Observation/state space or shape (default: None).
                              If it is not None, the num_observations property will contain the size of that space
    :type observation_space: int, tuple or list of integers, gym.Space or None, optional
    :param action_space: Action space or shape (default: None).
                            If it is not None, the num_actions property will contain the size of that space
    :type action_space: int, tuple or list of integers, gym.Space or None, optional
    :param device: Device on which a torch tensor is or will be allocated (default: "cuda:0")
    :type device: str or torch.device, optional
    :param unnormalized_log_prob: Flag to indicate how to be interpreted the network's output (default: True).
                                  If True, the network's output is interpreted as unnormalized log probabilities 
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

    :return: CategoricalModel instance
    :rtype: CategoricalModel
    """
    model = CategoricalModel(observation_space=observation_space,
                             action_space=action_space, 
                             device=device, 
                             unnormalized_log_prob=unnormalized_log_prob)
    
    model._instantiator_net = _generate_sequential(model=model,
                                                   input_shape=input_shape,
                                                   hiddens=hiddens,
                                                   hidden_activation=hidden_activation,
                                                   output_shape=output_shape,
                                                   output_activation=output_activation)
    model._instantiator_input_type = input_shape.value

    return model
    