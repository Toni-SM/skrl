from __future__ import annotations

from gymnasium import spaces

import torch

from skrl.models.torch import DeterministicMixin, Model


class _NestedPolicy(DeterministicMixin, Model):
    def __init__(self) -> None:
        Model.__init__(
            self,
            observation_space=spaces.Box(-1, 1, shape=(2,)),
            action_space=spaces.Box(-1, 1, shape=(2,)),
            device="cpu",
        )
        DeterministicMixin.__init__(self)
        self.input_layer = torch.nn.Linear(2, 3, bias=False)
        self.body = torch.nn.ModuleDict(
            {"layers": torch.nn.ModuleList([torch.nn.Linear(3, 3), torch.nn.Sequential(torch.nn.Linear(3, 2))])}
        )

    def compute(self, inputs: dict, role: str = "") -> tuple[torch.Tensor, dict]:
        x = self.input_layer(inputs["observations"])
        for layer in self.body["layers"]:
            x = layer(x)
        return x, {}


def test_initialize_weights_in_nested_module_containers() -> None:
    model = _NestedPolicy()
    biases = {name: parameter.clone() for name, parameter in model.named_parameters() if name.endswith("bias")}

    model.init_weights("constant_", val=0.25)

    for name, parameter in model.named_parameters():
        expected = biases[name] if name.endswith("bias") else torch.full_like(parameter, 0.25)
        torch.testing.assert_close(parameter, expected)
    actions, _ = model.act({"observations": torch.ones(2, 2)})
    assert torch.isfinite(actions).all()


def test_initialize_nested_biases_and_skip_biasless_layers() -> None:
    model = _NestedPolicy()
    weights = {name: parameter.clone() for name, parameter in model.named_parameters() if name.endswith("weight")}

    model.init_biases("constant_", val=3.0)

    assert model.input_layer.bias is None
    for name, parameter in model.named_parameters():
        expected = weights[name] if name.endswith("weight") else torch.full_like(parameter, 3.0)
        torch.testing.assert_close(parameter, expected)
