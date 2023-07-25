from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import torch

from skrl.models.torch import Model


class TabularMixin:
    def __init__(self, num_envs: int = 1, role: str = "") -> None:
        """Tabular mixin model

        :param num_envs: Number of environments (default: 1)
        :type num_envs: int, optional
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        Example::

            # define the model
            >>> import torch
            >>> from skrl.models.torch import Model, TabularMixin
            >>>
            >>> class GreedyPolicy(TabularMixin, Model):
            ...     def __init__(self, observation_space, action_space, device="cuda:0", num_envs=1):
            ...         Model.__init__(self, observation_space, action_space, device)
            ...         TabularMixin.__init__(self, num_envs)
            ...
            ...         self.table = torch.ones((num_envs, self.num_observations, self.num_actions),
            ...                                 dtype=torch.float32, device=self.device)
            ...
            ...     def compute(self, inputs, role):
            ...         actions = torch.argmax(self.table[torch.arange(self.num_envs).view(-1, 1), inputs["states"]],
            ...                                dim=-1, keepdim=True).view(-1,1)
            ...         return actions, {}
            ...
            >>> # given an observation_space: gym.spaces.Discrete with n=100
            >>> # and an action_space: gym.spaces.Discrete with n=5
            >>> model = GreedyPolicy(observation_space, action_space, num_envs=1)
            >>>
            >>> print(model)
            GreedyPolicy(
              (table): Tensor(shape=[1, 100, 5])
            )
        """
        self.num_envs = num_envs

    def __repr__(self) -> str:
        """String representation of an object as torch.nn.Module
        """
        lines = []
        for name in self._get_tensor_names():
            tensor = getattr(self, name)
            lines.append(f"({name}): {tensor.__class__.__name__}(shape={list(tensor.shape)})")

        main_str = self.__class__.__name__ + '('
        if lines:
            main_str += "\n  {}\n".format("\n  ".join(lines))
        main_str += ')'
        return main_str

    def _get_tensor_names(self) -> Sequence[str]:
        """Get the names of the tensors that the model is using

        :return: Tensor names
        :rtype: sequence of str
        """
        tensors = []
        for attr in dir(self):
            if not attr.startswith("__") and issubclass(type(getattr(self, attr)), torch.Tensor):
                tensors.append(attr)
        return sorted(tensors)

    def act(self,
            inputs: Mapping[str, Union[torch.Tensor, Any]],
            role: str = "") -> Tuple[torch.Tensor, Union[torch.Tensor, None], Mapping[str, Union[torch.Tensor, Any]]]:
        """Act in response to the state of the environment

        :param inputs: Model inputs. The most common keys are:

                       - ``"states"``: state of the environment used to make the decision
                       - ``"taken_actions"``: actions taken by the policy for the given states
        :type inputs: dict where the values are typically torch.Tensor
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        :return: Model output. The first component is the action to be taken by the agent.
                 The second component is ``None``. The third component is a dictionary containing extra output values
        :rtype: tuple of torch.Tensor, torch.Tensor or None, and dict

        Example::

            >>> # given a batch of sample states with shape (1, 100)
            >>> actions, _, outputs = model.act({"states": states})
            >>> print(actions[0], outputs)
            tensor([[3]], device='cuda:0') {}
        """
        actions, outputs = self.compute(inputs, role)
        return actions, None, outputs

    def table(self) -> torch.Tensor:
        """Return the Q-table

        :return: Q-table
        :rtype: torch.Tensor

        Example::

            >>> output = model.table()
            >>> print(output.shape)
            torch.Size([1, 100, 5])
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
        Model.to(self, *args, **kwargs)
        for name in self._get_tensor_names():
            setattr(self, name, getattr(self, name).to(*args, **kwargs))
        return self

    def state_dict(self, *args, **kwargs) -> Mapping:
        """Returns a dictionary containing a whole state of the module

        :return: A dictionary containing a whole state of the module
        :rtype: dict
        """
        _state_dict = {name: getattr(self, name) for name in self._get_tensor_names()}
        Model.state_dict(self, destination=_state_dict)
        return _state_dict

    def load_state_dict(self, state_dict: Mapping, strict: bool = True) -> None:
        """Copies parameters and buffers from state_dict into this module and its descendants

        :param state_dict: A dict containing parameters and persistent buffers
        :type state_dict: dict
        :param strict: Whether to strictly enforce that the keys in state_dict match the keys
                       returned by this module's state_dict() function (default: ``True``)
        :type strict: bool, optional
        """
        Model.load_state_dict(self, state_dict, strict=False)

        for name, tensor in state_dict.items():
            if hasattr(self, name) and isinstance(getattr(self, name), torch.Tensor):
                _tensor = getattr(self, name)
                if isinstance(_tensor, torch.Tensor):
                    if _tensor.shape == tensor.shape and _tensor.dtype == tensor.dtype:
                        setattr(self, name, tensor)
                    else:
                        raise ValueError(f"Tensor shape ({_tensor.shape} vs {tensor.shape}) or dtype ({_tensor.dtype} vs {tensor.dtype}) mismatch")
            else:
                raise ValueError(f"{name} is not a tensor of {self.__class__.__name__}")

    def save(self, path: str, state_dict: Optional[dict] = None) -> None:
        """Save the model to the specified path

        :param path: Path to save the model to
        :type path: str
        :param state_dict: State dictionary to save (default: ``None``).
                           If None, the model's state_dict will be saved
        :type state_dict: dict, optional

        Example::

            # save the current model to the specified path
            >>> model.save("/tmp/model.pt")
        """
        # TODO: save state_dict
        torch.save({name: getattr(self, name) for name in self._get_tensor_names()}, path)

    def load(self, path: str) -> None:
        """Load the model from the specified path

        The final storage device is determined by the constructor of the model

        :param path: Path to load the model from
        :type path: str

        :raises ValueError: If the models are not compatible

        Example::

            # load the model onto the CPU
            >>> model = Model(observation_space, action_space, device="cpu")
            >>> model.load("model.pt")

            # load the model onto the GPU 1
            >>> model = Model(observation_space, action_space, device="cuda:1")
            >>> model.load("model.pt")
        """
        tensors = torch.load(path)
        for name, tensor in tensors.items():
            if hasattr(self, name) and isinstance(getattr(self, name), torch.Tensor):
                _tensor = getattr(self, name)
                if isinstance(_tensor, torch.Tensor):
                    if _tensor.shape == tensor.shape and _tensor.dtype == tensor.dtype:
                        setattr(self, name, tensor)
                    else:
                        raise ValueError(f"Tensor shape ({_tensor.shape} vs {tensor.shape}) or dtype ({_tensor.dtype} vs {tensor.dtype}) mismatch")
            else:
                raise ValueError(f"{name} is not a tensor of {self.__class__.__name__}")
