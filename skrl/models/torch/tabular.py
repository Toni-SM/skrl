from typing import Union, Tuple

import gym

import torch

from . import Model


class TabularModel(Model):
    def __init__(self, 
                 observation_space: Union[int, Tuple[int], gym.Space, None] = None, 
                 action_space: Union[int, Tuple[int], gym.Space, None] = None, 
                 device: Union[str, torch.device] = "cuda:0",
                 num_envs: int = 1) -> None:
        """Tabular model

        :param observation_space: Observation/state space or shape (default: None).
                                  If it is not None, the num_observations property will contain the size of that space
        :type observation_space: int, tuple or list of integers, gym.Space or None, optional
        :param action_space: Action space or shape (default: None).
                             If it is not None, the num_actions property will contain the size of that space
        :type action_space: int, tuple or list of integers, gym.Space or None, optional
        :param device: Device on which a torch tensor is or will be allocated (default: "cuda:0")
        :type device: str or torch.device, optional
        :param num_envs: Number of environments (default: 1)
        :type num_envs: int, optional
        """
        super(TabularModel, self).__init__(observation_space, action_space, device)

        self.num_envs = num_envs

    def _get_tensor_names(self) -> Tuple[str]:
        """Get the names of the tensors that the model is using

        :return: Tensor names
        :rtype: tuple of str
        """
        tensors = []
        for attr in dir(self):
            if not attr.startswith("__") and issubclass(type(getattr(self, attr)), torch.Tensor):
                tensors.append(attr)
        return sorted(tensors)

    def act(self, 
            states: torch.Tensor, 
            taken_actions: Union[torch.Tensor, None] = None, 
            inference=False) -> Tuple[torch.Tensor]:
        """Act in response to the state of the environment

        :param states: Observation/state of the environment used to make the decision
        :type states: torch.Tensor
        :param taken_actions: Actions taken by a policy to the given states (default: None)
        :type taken_actions: torch.Tensor or None, optional
        :param inference: Flag to indicate whether the model is making inference (default: False).
                          If True, the returned tensors will be detached from the current graph
        :type inference: bool, optional

        :return: Action to be taken by the agent given the state of the environment.
                 The tuple's components are the computed actions and None for the last two components
        :rtype: tuple of torch.Tensor
        """
        actions = self.compute(states.to(self.device), 
                               taken_actions.to(self.device) if taken_actions is not None else taken_actions)
        return actions, None, None
        
    def table(self) -> torch.Tensor:
        """Return the Q-table

        :return: Q-table
        :rtype: torch.Tensor
        """
        return self.q_table

    def to(self, *args, **kwargs) -> Model:
        """Move the model to a different device

        :param args: Arguments to pass to the method
        :type args: tuple
        :param kwargs: Keyword arguments to pass to the method
        :type kwargs: dict

        :return: Model moved to the specified device
        :rtype: Model
        """
        super(TabularModel, self).to(*args, **kwargs)
        for name in self._get_tensor_names():
            setattr(self, name, getattr(self, name).to(*args, **kwargs))
        return self

    def save(self, path: str, state_dict: Union[dict, None] = None) -> None:
        """Save the model to the specified path
            
        :param path: Path to save the model to
        :type path: str
        :param state_dict: State dictionary to save (default: None).
                           If None, the model's state_dict will be saved
        :type state_dict: dict, optional
        """
        torch.save({name: getattr(self, name) for name in self._get_tensor_names()}, path)

    def load(self, path: str) -> None:
        """Load the model from the specified path
        
        :raises ValueError: If the models are not compatible

        :param path: Path to load the model from
        :type path: str
        """
        tensors = torch.load(path)
        for name, tensor in tensors.items():
            if hasattr(self, name) and isinstance(getattr(self, name), torch.Tensor):
                _tensor = getattr(self, name)
                if isinstance(_tensor, torch.Tensor):
                    if _tensor.shape == tensor.shape and _tensor.dtype == tensor.dtype:
                        setattr(self, name, tensor)
                    else:
                        raise ValueError("Tensor shape ({} vs {}) or dtype ({} vs {}) mismatch"\
                            .format(_tensor.shape, tensor.shape, _tensor.dtype, tensor.dtype))
            else:
                raise ValueError("{} is not a tensor of {}".format(name, self.__class__.__name__))
