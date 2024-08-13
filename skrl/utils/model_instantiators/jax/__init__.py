from typing import Optional, Tuple, Union

from enum import Enum
import gym
import gymnasium

import flax.linen as nn
import jax
import jax.numpy as jnp  # noqa

from skrl.models.jax import Model  # noqa
from skrl.models.jax import CategoricalMixin, DeterministicMixin, GaussianMixin  # noqa


__all__ = ["categorical_model", "deterministic_model", "gaussian_model", "Shape"]


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
                       If activation is an empty string, a placeholder will be returned (``lambda x: x``)
    :type activation: str
    :param as_string: Whether to return the activation function as string.
    :type as_string: bool

    :raises: ValueError if activation is not a valid activation function

    :return: activation function
    :rtype: nn.Module
    """
    if not activation:
        return None if as_string else lambda x: x
    elif activation == "relu":
        return "nn.relu" if as_string else nn.relu
    elif activation == "tanh":
        return "nn.tanh" if as_string else nn.tanh
    elif activation == "sigmoid":
        return "nn.sigmoid" if as_string else nn.sigmoid
    elif activation == "leaky_relu":
        return "nn.leaky_relu" if as_string else nn.leaky_relu
    elif activation == "elu":
        return "nn.elu" if as_string else nn.elu
    elif activation == "softplus":
        return "nn.softplus" if as_string else nn.softplus
    elif activation == "softsign":
        return "nn.soft_sign" if as_string else nn.soft_sign
    elif activation == "selu":
        return "nn.selu" if as_string else nn.selu
    elif activation == "softmax":
        return "nn.softmax" if as_string else nn.softmax
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
    for i in range(len(hiddens)):
        # first and middle layers
        modules.append(f"nn.Dense({hiddens[i]})")
        if hidden_activation[i]:
            modules.append(_get_activation_function(hidden_activation[i], as_string=True))
        # last layer
        if i == len(hiddens) - 1:
            modules.append(f"nn.Dense({_get_num_units_by_shape(None, output_shape, as_string=True)})")
            if output_activation:
                modules.append(_get_activation_function(output_activation, as_string=True))
    return f'nn.Sequential([{", ".join(modules)}])'

def gaussian_model(observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                   action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                   device: Optional[Union[str, jax.Device]] = None,
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
                   output_scale: float = 1.0,
                   return_source: bool = False) -> Model:
    """Instantiate a Gaussian model

    :param observation_space: Observation/state space or shape (default: None).
                              If it is not None, the num_observations property will contain the size of that space
    :type observation_space: int, tuple or list of integers, gym.Space, gymnasium.Space or None, optional
    :param action_space: Action space or shape (default: None).
                         If it is not None, the num_actions property will contain the size of that space
    :type action_space: int, tuple or list of integers, gym.Space, gymnasium.Space or None, optional
    :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                   If None, the device will be either ``"cuda"`` if available or ``"cpu"``
    :type device: str or jax.Device, optional
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
    :param return_source: Whether to return the source string containing the model class used to
                          instantiate the model rather than the model instance (default: False).
    :type return_source: bool, optional

    :return: Gaussian model instance
    :rtype: Model
    """
    # network
    net = _generate_sequential(None, input_shape, hiddens, hidden_activation, output_shape, output_activation)

    # compute
    if input_shape == Shape.OBSERVATIONS:
        forward = 'self.net(inputs["states"])'
    elif input_shape == Shape.ACTIONS:
        forward = 'self.net(inputs["taken_actions"])'
    elif input_shape == Shape.STATES_ACTIONS:
        forward = 'self.net(jnp.concatenate([inputs["states"], inputs["taken_actions"]], axis=-1))'
    if output_scale != 1:
        forward = f"{output_scale} * {forward}"

    template = f"""class GaussianModel(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                    clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum", **kwargs):
        Model.__init__(self, observation_space, action_space, device, **kwargs)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

    def setup(self):
        self.net = {net}
        self.log_std_parameter = self.param("log_std_parameter", lambda _: {initial_log_std} * jnp.ones({_get_num_units_by_shape(None, output_shape, as_string=True)}))

    def __call__(self, inputs, role):
        return {forward}, self.log_std_parameter, {{}}
    """
    if return_source:
        return template
    _locals = {}
    exec(template, globals(), _locals)
    return _locals["GaussianModel"](observation_space=observation_space,
                                    action_space=action_space,
                                    device=device,
                                    clip_actions=clip_actions,
                                    clip_log_std=clip_log_std,
                                    min_log_std=min_log_std,
                                    max_log_std=max_log_std)

def deterministic_model(observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                        action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                        device: Optional[Union[str, jax.Device]] = None,
                        clip_actions: bool = False,
                        input_shape: Shape = Shape.STATES,
                        hiddens: list = [256, 256],
                        hidden_activation: list = ["relu", "relu"],
                        output_shape: Shape = Shape.ACTIONS,
                        output_activation: Optional[str] = "tanh",
                        output_scale: float = 1.0,
                        return_source: bool = False) -> Model:
    """Instantiate a deterministic model

    :param observation_space: Observation/state space or shape (default: None).
                              If it is not None, the num_observations property will contain the size of that space
    :type observation_space: int, tuple or list of integers, gym.Space, gymnasium.Space or None, optional
    :param action_space: Action space or shape (default: None).
                         If it is not None, the num_actions property will contain the size of that space
    :type action_space: int, tuple or list of integers, gym.Space, gymnasium.Space or None, optional
    :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                   If None, the device will be either ``"cuda"`` if available or ``"cpu"``
    :type device: str or jax.Device, optional
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
    :param return_source: Whether to return the source string containing the model class used to
                          instantiate the model rather than the model instance (default: False).
    :type return_source: bool, optional

    :return: Deterministic model instance
    :rtype: Model
    """
    # network
    net = _generate_sequential(None, input_shape, hiddens, hidden_activation, output_shape, output_activation)

    # compute
    if input_shape == Shape.OBSERVATIONS:
        forward = 'self.net(inputs["states"])'
    elif input_shape == Shape.ACTIONS:
        forward = 'self.net(inputs["taken_actions"])'
    elif input_shape == Shape.STATES_ACTIONS:
        forward = 'self.net(jnp.concatenate([inputs["states"], inputs["taken_actions"]], axis=-1))'
    if output_scale != 1:
        forward = f"{output_scale} * {forward}"

    template = f"""class DeterministicModel(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False, **kwargs):
        Model.__init__(self, observation_space, action_space, device, **kwargs)
        DeterministicMixin.__init__(self, clip_actions)

    def setup(self):
        self.net = {net}

    def __call__(self, inputs, role):
        return {forward}, {{}}
    """
    if return_source:
        return template
    _locals = {}
    exec(template, globals(), _locals)
    return _locals["DeterministicModel"](observation_space=observation_space,
                                         action_space=action_space,
                                         device=device,
                                         clip_actions=clip_actions)

def categorical_model(observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                      action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                      device: Optional[Union[str, jax.Device]] = None,
                      unnormalized_log_prob: bool = True,
                      input_shape: Shape = Shape.STATES,
                      hiddens: list = [256, 256],
                      hidden_activation: list = ["relu", "relu"],
                      output_shape: Shape = Shape.ACTIONS,
                      output_activation: Optional[str] = None,
                      return_source: bool = False) -> Model:
    """Instantiate a categorical model

    :param observation_space: Observation/state space or shape (default: None).
                              If it is not None, the num_observations property will contain the size of that space
    :type observation_space: int, tuple or list of integers, gym.Space, gymnasium.Space or None, optional
    :param action_space: Action space or shape (default: None).
                         If it is not None, the num_actions property will contain the size of that space
    :type action_space: int, tuple or list of integers, gym.Space, gymnasium.Space or None, optional
    :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                   If None, the device will be either ``"cuda"`` if available or ``"cpu"``
    :type device: str or jax.Device, optional
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
    :param return_source: Whether to return the source string containing the model class used to
                          instantiate the model rather than the model instance (default: False).
    :type return_source: bool, optional

    :return: Categorical model instance
    :rtype: Model
    """
    # network
    net = _generate_sequential(None, input_shape, hiddens, hidden_activation, output_shape, output_activation)

    # compute
    if input_shape == Shape.OBSERVATIONS:
        forward = 'self.net(inputs["states"])'
    elif input_shape == Shape.ACTIONS:
        forward = 'self.net(inputs["taken_actions"])'
    elif input_shape == Shape.STATES_ACTIONS:
        forward = 'self.net(jnp.concatenate([inputs["states"], inputs["taken_actions"]], axis=-1))'

    template = f"""class CategoricalModel(CategoricalMixin, Model):
    def __init__(self, observation_space, action_space, device, unnormalized_log_prob=True, **kwargs):
        Model.__init__(self, observation_space, action_space, device, **kwargs)
        CategoricalMixin.__init__(self, unnormalized_log_prob)

    def setup(self):
        self.net = {net}

    def __call__(self, inputs, role):
        return {forward}, {{}}
    """
    if return_source:
        return template
    _locals = {}
    exec(template, globals(), _locals)
    return _locals["CategoricalModel"](observation_space=observation_space,
                                      action_space=action_space,
                                      device=device,
                                      unnormalized_log_prob=unnormalized_log_prob)
