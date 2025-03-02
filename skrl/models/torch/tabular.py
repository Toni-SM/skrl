from typing import Any, Mapping, Optional, Sequence, Tuple, Union

from packaging import version

import torch

from skrl.models.torch import Model


class TabularMixin:
    def __init__(self, *, num_envs: int = 1, role: str = "") -> None:
        """Tabular mixin model.

        :param num_envs: Number of environments.
        :param role: Role played by the model.
        """
        self.num_envs = num_envs

    def __repr__(self) -> str:
        """String representation of the object as torch.nn.Module.

        :return: String representation of the object.
        """
        lines = []
        for name in self._get_tensor_names():
            tensor = getattr(self, name)
            lines.append(f"({name}): {tensor.__class__.__name__}(shape={list(tensor.shape)})")

        main_str = self.__class__.__name__ + "("
        if lines:
            main_str += "\n  {}\n".format("\n  ".join(lines))
        main_str += ")"
        return main_str

    def _get_tensor_names(self) -> Sequence[str]:
        """Get the names of the tensors that the model is using.

        :return: Tensor names.
        """
        tensors = []
        for attr in dir(self):
            if not attr.startswith("__") and issubclass(type(getattr(self, attr)), torch.Tensor):
                tensors.append(attr)
        return sorted(tensors)

    def act(
        self, inputs: Mapping[str, Union[torch.Tensor, Any]], *, role: str = ""
    ) -> Tuple[torch.Tensor, Mapping[str, Union[torch.Tensor, Any]]]:
        """Act in response to the observations/states of the environment.

        :param inputs: Model inputs. The most common keys are:

            - ``"observations"``: observation of the environment used to make the decision.
            - ``"states"``: state of the environment used to make the decision.
            - ``"taken_actions"``: actions taken by the policy for the given observations/states.
        :param role: Role played by the model.

        :return: Model output. The first component is the expected action/value returned by the model.
            The second component is a dictionary containing extra output values according to the model.
        """
        actions, outputs = self.compute(inputs, role)
        return actions, outputs

    def table(self, *, role: str = "") -> torch.Tensor:
        """Return the *table* defined by the model.

        :param role: Role played by the model.

        :return: Table.

        :raises NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("The method to get the model's table (.table()) is not implemented")

    def to(self, *args, **kwargs) -> Model:
        """Move the model to a different device.

        :param args: Arguments to pass to the method.
        :param kwargs: Keyword arguments to pass to the method.

        :return: Model moved to the specified device.
        """
        Model.to(self, *args, **kwargs)
        for name in self._get_tensor_names():
            setattr(self, name, getattr(self, name).to(*args, **kwargs))
        return self

    def state_dict(self, *args, **kwargs) -> Mapping:
        """Returns a dictionary containing a whole state of the module.

        :return: A dictionary containing a whole state of the module.
        """
        _state_dict = {name: getattr(self, name) for name in self._get_tensor_names()}
        Model.state_dict(self, destination=_state_dict)
        return _state_dict

    def load_state_dict(self, state_dict: Mapping, strict: bool = True) -> None:
        """Copies parameters and buffers from state_dict into this module and its descendants.

        :param state_dict: A dict containing parameters and persistent buffers.
        :param strict: Whether to strictly enforce that the keys in state_dict match the keys
            returned by this module's state_dict() function.
        """
        Model.load_state_dict(self, state_dict, strict=False)

        for name, tensor in state_dict.items():
            if hasattr(self, name) and isinstance(getattr(self, name), torch.Tensor):
                _tensor = getattr(self, name)
                if isinstance(_tensor, torch.Tensor):
                    if _tensor.shape == tensor.shape and _tensor.dtype == tensor.dtype:
                        setattr(self, name, tensor)
                    else:
                        raise ValueError(
                            f"Tensor shape ({_tensor.shape} vs {tensor.shape}) or dtype ({_tensor.dtype} vs {tensor.dtype}) mismatch"
                        )
            else:
                raise ValueError(f"{name} is not a tensor of {self.__class__.__name__}")

    def save(self, path: str, state_dict: Optional[dict] = None) -> None:
        """Save the model to the specified path.

        :param path: Path to save the model to.
        :param state_dict: State dictionary to save. If None, the model's state_dict will be saved.

        Example::

            # save the current model to the specified path
            >>> model.save("/tmp/model.pt")
        """
        # TODO: save state_dict
        torch.save({name: getattr(self, name) for name in self._get_tensor_names()}, path)

    def load(self, path: str) -> None:
        """Load the model from the specified path.

        The final storage device is determined by the constructor of the model.

        :param path: Path to load the model from.

        :raises ValueError: If the models are not compatible.

        Example::

            # load the model onto the CPU
            >>> model = Model(device="cpu")
            >>> model.load("model.pt")

            # load the model onto the GPU 1
            >>> model = Model(device="cuda:1")
            >>> model.load("model.pt")
        """
        if version.parse(torch.__version__) >= version.parse("1.13"):
            tensors = torch.load(path, weights_only=False)  # prevent torch:FutureWarning
        else:
            tensors = torch.load(path)
        for name, tensor in tensors.items():
            if hasattr(self, name) and isinstance(getattr(self, name), torch.Tensor):
                _tensor = getattr(self, name)
                if isinstance(_tensor, torch.Tensor):
                    if _tensor.shape == tensor.shape and _tensor.dtype == tensor.dtype:
                        setattr(self, name, tensor)
                    else:
                        raise ValueError(
                            f"Tensor shape ({_tensor.shape} vs {tensor.shape}) or dtype ({_tensor.dtype} vs {tensor.dtype}) mismatch"
                        )
            else:
                raise ValueError(f"{name} is not a tensor of {self.__class__.__name__}")
