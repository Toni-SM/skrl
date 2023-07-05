# [start-model-torch]
from typing import Optional, Union, Mapping, Sequence, Tuple, Any

import gym, gymnasium

import torch

from skrl.models.torch import Model


class CustomModel(Model):
    def __init__(self,
                 observation_space: Union[int, Sequence[int], gym.Space, gymnasium.Space],
                 action_space: Union[int, Sequence[int], gym.Space, gymnasium.Space],
                 device: Optional[Union[str, torch.device]] = None) -> None:
        """Custom model

        :param observation_space: Observation/state space or shape.
                                  The ``num_observations`` property will contain the size of that space
        :type observation_space: int, sequence of int, gym.Space, gymnasium.Space
        :param action_space: Action space or shape.
                             The ``num_actions`` property will contain the size of that space
        :type action_space: int, sequence of int, gym.Space, gymnasium.Space
        :param device: Device on which a torch tensor is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda:0"`` if available or ``"cpu"``
        :type device: str or torch.device, optional
        """
        super().__init__(observation_space, action_space, device)
        # =====================================
        # - define custom attributes and others
        # =====================================
        flax.linen.Module.__post_init__(self)

    def act(self,
            inputs: Mapping[str, Union[torch.Tensor, Any]],
            role: str = "") -> Tuple[torch.Tensor, Union[torch.Tensor, None], Mapping[str, Union[torch.Tensor, Any]]]:
        """Act according to the specified behavior

        :param inputs: Model inputs. The most common keys are:

                       - ``"states"``: state of the environment used to make the decision
                       - ``"taken_actions"``: actions taken by the policy for the given states
        :type inputs: dict where the values are typically torch.Tensor
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        :return: Model output. The first component is the action to be taken by the agent.
                 The second component is the log of the probability density function for stochastic models
                 or None for deterministic models. The third component is a dictionary containing extra output values
        :rtype: tuple of torch.Tensor, torch.Tensor or None, and dictionary
        """
        # ==============================
        # - act in response to the state
        # ==============================
# [end-model-torch]


# [start-model-jax]
from typing import Optional, Union, Mapping, Tuple, Any

import gym, gymnasium

import flax
import jaxlib
import jax.numpy as jnp

from skrl.models.jax import Model


class CustomModel(Model):
    def __init__(self,
                 observation_space: Union[int, Sequence[int], gym.Space, gymnasium.Space],
                 action_space: Union[int, Sequence[int], gym.Space, gymnasium.Space],
                 device: Optional[Union[str, jaxlib.xla_extension.Device]] = None,
                 parent: Optional[Any] = None,
                 name: Optional[str] = None) -> None:
        """Custom model

        :param observation_space: Observation/state space or shape.
                                  The ``num_observations`` property will contain the size of that space
        :type observation_space: int, sequence of int, gym.Space, gymnasium.Space
        :param action_space: Action space or shape.
                             The ``num_actions`` property will contain the size of that space
        :type action_space: int, sequence of int, gym.Space, gymnasium.Space
        :param device: Device on which a jax array is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda:0"`` if available or ``"cpu"``
        :type device: str or jaxlib.xla_extension.Device, optional
        :param parent: The parent Module of this Module (default: ``None``).
                       It is a Flax reserved attribute
        :type parent: str, optional
        :param name: The name of this Module (default: ``None``).
                     It is a Flax reserved attribute
        :type name: str, optional
        """
        Model.__init__(self, observation_space, action_space, device, parent, name)
        # =====================================
        # - define custom attributes and others
        # =====================================
        flax.linen.Module.__post_init__(self)

    def act(self,
            inputs: Mapping[str, Union[jnp.ndarray, Any]],
            role: str = "",
            params: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, Union[jnp.ndarray, None], Mapping[str, Union[jnp.ndarray, Any]]]:
        """Act according to the specified behavior

        :param inputs: Model inputs. The most common keys are:

                       - ``"states"``: state of the environment used to make the decision
                       - ``"taken_actions"``: actions taken by the policy for the given states
        :type inputs: dict where the values are typically jnp.ndarray
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional
        :param params: Parameters used to compute the output (default: ``None``).
                       If ``None``, internal parameters will be used
        :type params: jnp.array

        :return: Model output. The first component is the action to be taken by the agent.
                 The second component is the log of the probability density function.
                 The third component is a dictionary containing the mean actions ``"mean_actions"``
                 and extra output values
        :rtype: tuple of jnp.ndarray, jnp.ndarray or None, and dictionary
        """
        # ==============================
        # - act in response to the state
        # ==============================
# [end-model-jax]

# =============================================================================

# [start-mixin-torch]
from typing import Union, Mapping, Tuple, Any

import torch


class CustomMixin:
    def __init__(self, role: str = "") -> None:
        """Custom mixin

        :param role: Role play by the model (default: ``""``)
        :type role: str, optional
        """
        # =====================================
        # - define custom attributes and others
        # =====================================

    def act(self,
            inputs: Mapping[str, Union[torch.Tensor, Any]],
            role: str = "") -> Tuple[torch.Tensor, Union[torch.Tensor, None], Mapping[str, Union[torch.Tensor, Any]]]:
        """Act according to the specified behavior

        :param inputs: Model inputs. The most common keys are:

                       - ``"states"``: state of the environment used to make the decision
                       - ``"taken_actions"``: actions taken by the policy for the given states
        :type inputs: dict where the values are typically torch.Tensor
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        :return: Model output. The first component is the action to be taken by the agent.
                 The second component is the log of the probability density function for stochastic models
                 or None for deterministic models. The third component is a dictionary containing extra output values
        :rtype: tuple of torch.Tensor, torch.Tensor or None, and dictionary
        """
        # ==============================
        # - act in response to the state
        # ==============================
# [end-mixin-torch]


# [start-mixin-jax]
from typing import Optional, Union, Mapping, Tuple, Any

import flax
import jax.numpy as jnp


class CustomMixin:
    def __init__(self, role: str = "") -> None:
        """Custom mixin

        :param role: Role play by the model (default: ``""``)
        :type role: str, optional
        """
        # =====================================
        # - define custom attributes and others
        # =====================================
        flax.linen.Module.__post_init__(self)

    def act(self,
            inputs: Mapping[str, Union[jnp.ndarray, Any]],
            role: str = "",
            params: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, Union[jnp.ndarray, None], Mapping[str, Union[jnp.ndarray, Any]]]:
        """Act according to the specified behavior

        :param inputs: Model inputs. The most common keys are:

                       - ``"states"``: state of the environment used to make the decision
                       - ``"taken_actions"``: actions taken by the policy for the given states
        :type inputs: dict where the values are typically jnp.ndarray
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional
        :param params: Parameters used to compute the output (default: ``None``).
                       If ``None``, internal parameters will be used
        :type params: jnp.array

        :return: Model output. The first component is the action to be taken by the agent.
                 The second component is the log of the probability density function.
                 The third component is a dictionary containing the mean actions ``"mean_actions"``
                 and extra output values
        :rtype: tuple of jnp.ndarray, jnp.ndarray or None, and dictionary
        """
        # ==============================
        # - act in response to the state
        # ==============================
# [end-mixin-jax]
