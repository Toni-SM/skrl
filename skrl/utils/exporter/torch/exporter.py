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
    stochastic_evaluation: bool = False,
    device: str | torch.device = "cpu",
) -> None:
    """Export a policy to a TorchScript (JIT) file.

    The exporter wraps the given policy together with optional observation and
    state preprocessors into a single module exposing
    ``forward(observations, states) -> actions`` for inference.

    :param policy: Policy model to be exported.
    :param observation_preprocessor: Module to preprocess observations, applied before the policy.
    :param state_preprocessor: Module to preprocess states, applied before the policy.
    :param path: Directory where the exported file will be saved.
    :param filename: Output file name. Defaults to ``"policy.pt"``.
    :param example_inputs: Example inputs for tracing. If ``None``, dummy inputs with batch size 1 are used.
    :param optimize: Whether to optimize the traced model for inference. Defaults to ``True``.
    :param stochastic_evaluation: Whether to use stochastic actions returned by ``act()`` during export.
        If ``False``, mean actions are used when available to match deterministic evaluation.
    :param device: Device used for export. Defaults to ``"cpu"``.
    """

    exporter = _TorchPolicyExporter(
        policy,
        observation_preprocessor,
        state_preprocessor,
        stochastic_evaluation=stochastic_evaluation,
    )
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
    stochastic_evaluation: bool = False,
    dynamo: bool = True,
    opset_version: int = 18,
    verbose: bool = False,
    device: str | torch.device = "cpu",
) -> None:
    """Export a policy to an ONNX file.

    The exporter wraps the given policy together with optional observation and
    state preprocessors into a single module exposing
    ``forward(observations, states) -> actions`` for inference.

    :param policy: Policy model to be exported.
    :param observation_preprocessor: Module to preprocess observations, applied before the policy.
    :param state_preprocessor: Module to preprocess states, applied before the policy.
    :param path: Directory where the exported file will be saved.
    :param filename: Output file name. Defaults to ``"policy.onnx"``.
    :param example_inputs: Example inputs for export. If ``None``, dummy inputs with batch size 1 are used.
    :param optimize: Whether to optimize the exported model for inference. Defaults to ``True``.
    :param stochastic_evaluation: Whether to use stochastic actions returned by ``act()`` during export.
        If ``False``, mean actions are used when available to match deterministic evaluation.
    :param dynamo: Whether to use Torch Dynamo for export. Defaults to ``True``.
    :param opset_version: ONNX opset version to use. Defaults to ``18``.
    :param verbose: Whether to print the export graph summary.
    :param device: Device used for export. Defaults to ``"cpu"``.
    """

    exporter = _TorchPolicyExporter(
        policy,
        observation_preprocessor,
        state_preprocessor,
        stochastic_evaluation=stochastic_evaluation,
    )
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
    """Wrapper that prepares a policy model for export.

    This module exposes a minimal ``forward(observations, states)`` that returns
    actions, handling the internal policy call and input dictionary construction
    expected by policy's ``act()`` method.

    :param policy: A policy model to be wrapped.
    :param observation_preprocessor: Optional preprocessor applied to observations.
    :param state_preprocessor: Optional preprocessor applied to states.
    :param stochastic_evaluation: Whether to keep stochastic actions from policy ``act()``.
    """

    def __init__(
        self,
        policy: Model,
        observation_preprocessor: torch.nn.Module | None = None,
        state_preprocessor: torch.nn.Module | None = None,
        stochastic_evaluation: bool = False,
    ) -> None:
        super().__init__()

        self.policy = policy
        self._observation_preprocessor = (
            observation_preprocessor if observation_preprocessor is not None else torch.nn.Identity()
        )
        self._state_preprocessor = state_preprocessor if state_preprocessor is not None else torch.nn.Identity()
        self._stochastic_evaluation = stochastic_evaluation

        self._num_observations = getattr(self.policy, "num_observations", 0)
        self._num_states = getattr(self.policy, "num_states", 0)

    @torch.no_grad()
    def forward(self, observations: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """Compute actions from observations and states.

        :param observations: Batch of environment observations.
        :param states: Batch of agent states (or zeros if unused).
        :returns: Batch of actions produced by the policy.
        """
        actions, outputs = self.policy.act(
            {
                "observations": self._observation_preprocessor(observations),
                "states": self._state_preprocessor(states),
            },
            role="policy",
        )

        if self._stochastic_evaluation:
            return actions
        return outputs.get("mean_actions", actions)
