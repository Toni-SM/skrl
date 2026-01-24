from __future__ import annotations

from typing import TYPE_CHECKING

import os

import torch


if TYPE_CHECKING:
    from skrl.models.torch import Model


def export_policy_as_jit(
    policy: Model,
    observation_preprocessor: torch.nn.Module | None,
    state_preprocessor: torch.nn.Module | None,
    path: str,
    filename: str = "policy.pt",
    example_inputs: dict[str, torch.Tensor] | None = None,
    optimize: bool = True,
    device: str | torch.device = "cpu",
) -> None:
    """Export a policy to a Torch JIT file.

    This exporter is designed for skrl models during evaluation. It wraps the given
    policy together with an optional observation preprocessor and produces a single
    module with a simple `forward(obs)` -> `actions` interface.

    Limitations:
    - Torch-only base
    - Non-recurrent policies (RNN/LSTM/GRU export is out of scope here)

    Args:
        policy: A skrl policy model (torch.nn.Module) implementing `act(inputs)`.
        observation_preprocessor: Optional module to preprocess observations.
        path: Directory to save the file to.
        filename: Output file name, defaults to "policy.pt".
        example_inputs: Example inputs for tracing. If None, dummy inputs with batch size 1 will be used.
        optimize: Whether to optimize the traced model for inference, defaults to True.
        device: Device to use for export, defaults to "cpu".
    """

    exporter = _TorchPolicyExporter(policy, observation_preprocessor, state_preprocessor)
    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, filename)

    exporter.to(device)
    exporter.eval()

    # Use tracing for broader TorchScript compatibility with dict-based models
    if example_inputs is not None:
        example_inputs = {k: v.to(device) for k, v in example_inputs.items()}
    else:
        example_inputs = {
            "observations": torch.zeros(1, exporter._num_observations, device=device),
            "states": torch.zeros(1, exporter._num_states, device=device),
        }

    traced = torch.jit.trace(exporter, tuple(example_inputs.values()))
    if optimize:
        traced = torch.jit.optimize_for_inference(traced)
    torch.jit.save(traced, full_path)


def export_policy_as_onnx(
    policy: Model,
    observation_preprocessor: torch.nn.Module | None,
    state_preprocessor: torch.nn.Module | None,
    path: str,
    filename: str = "policy.onnx",
    example_inputs: dict[str, torch.Tensor] | None = None,
    optimize: bool = True,
    dynamo: bool = True,
    opset_version: int = 18,
    verbose: bool = False,
    device: str | torch.device = "cpu",
) -> None:
    """Export a policy to an ONNX file.

    This exporter is designed for skrl models during evaluation. It wraps the given
    policy together with an optional observation preprocessor and produces a single
    ONNX graph with `obs` input and `actions` output.

    Limitations:
    - Torch-only base
    - Non-recurrent policies (RNN/LSTM/GRU export is out of scope here)

    Args:
        policy: A skrl policy model (torch.nn.Module) implementing `act(inputs)`.
        observation_preprocessor: Optional module to preprocess observations.
        path: Directory to save the file to.
        filename: Output file name, defaults to "policy.onnx".
        example_inputs: Example inputs for tracing. If None, dummy inputs with batch size 1 will be used.
        optimize: Whether to optimize the model for inference, defaults to True.
        dynamo: Whether to use Torch Dynamo for export, defaults to True.
        opset_version: ONNX opset version to use, defaults to 18.
        verbose: Whether to print the model export graph summary.
        device: Device to use for export, defaults to "cpu".
    """

    exporter = _TorchPolicyExporter(policy, observation_preprocessor, state_preprocessor)
    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, filename)

    exporter.to(device)
    exporter.eval()

    if example_inputs is not None:
        example_inputs = {k: v.to(device) for k, v in example_inputs.items()}
    else:
        example_inputs = {
            "observations": torch.zeros(1, exporter._num_observations, device=device),
            "states": torch.zeros(1, exporter._num_states, device=device),
        }

    torch.onnx.export(
        exporter,
        tuple(example_inputs.values()),
        full_path,
        artifacts_dir=path,
        opset_version=opset_version,
        verbose=verbose,
        report=verbose,
        input_names=["observations", "states"],
        output_names=["actions"],
        optimize=optimize,
        verify=True,
        dynamo=dynamo,
    )


class _TorchPolicyExporter(torch.nn.Module):
    """Wrap a skrl policy and optional observation preprocessor for export.

    The wrapper exposes a minimal `forward(obs)` that returns actions, handling the
    internal policy call and dict construction expected by skrl models.
    """

    def __init__(
        self,
        policy: Model,
        observation_preprocessor: torch.nn.Module | None = None,
        state_preprocessor: torch.nn.Module | None = None,
    ) -> None:
        super().__init__()
        # keep given instances to preserve any registered buffers/state; move to CPU on export
        self.policy = policy
        self._observation_preprocessor = (
            observation_preprocessor if observation_preprocessor is not None else torch.nn.Identity()
        )
        self._state_preprocessor = state_preprocessor if state_preprocessor is not None else torch.nn.Identity()

        # skrl `Model` exposes `num_observations` (0 if `observation_space` is None)
        # fall back to attempting to infer input size from first linear layer if necessary
        self._num_observations = getattr(self.policy, "num_observations", 0)
        self._num_states = getattr(self.policy, "num_states", 0)

    @torch.no_grad()
    def forward(self, observations: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        actions, _ = self.policy.act(
            {
                "observations": self._observation_preprocessor(observations),
                "states": self._state_preprocessor(states),
            },
            role="policy",
        )
        return actions
