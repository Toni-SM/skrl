from typing import Any, Mapping, Tuple, Union

import torch


class TabularMixin:
    def __init__(self, *, role: str = "") -> None:
        """Tabular mixin model.

        :param role: Role played by the model.
        """
        pass

    def __repr__(self) -> str:
        """String representation of the object as torch.nn.Module.

        :return: String representation of the object.
        """
        lines = []
        for name, parameter in self.named_parameters():
            lines.append(f"({name}): {parameter.__class__.__name__}(shape={list(parameter.shape)})")

        string = self.__class__.__name__ + "("
        if lines:
            string += "\n  {}\n".format("\n  ".join(lines))
        string += ")"
        return string

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

    def tables(self, *, role: str = "") -> Mapping[str, torch.Tensor]:
        """Return the *tables* defined by the model.

        :param role: Role played by the model.

        :return: Tables.
        """
        return {name: param for name, param in self.named_parameters()}
