from __future__ import annotations

from typing import Any, Literal

import math

import warp as wp

from skrl import config
from skrl.utils.spaces.warp import compute_space_limits


LOG_SQRT_2_PI = wp.constant(math.log(math.sqrt(2 * math.pi)))
HALF_LOG_2_PI_PLUS = wp.constant(0.5 + 0.5 * math.log(2 * math.pi))


@wp.func
def _log_prob(
    action: float,
    loc: float,
    scale: float,
):
    return -wp.pow(action - loc, 2.0) / (2.0 * wp.pow(scale, 2.0)) - wp.log(scale) - LOG_SQRT_2_PI


@wp.kernel
def _gaussian(
    # inputs
    loc: wp.array2d(dtype=float),
    log_std: wp.array1d(dtype=float),
    log_std_min: float,
    log_std_max: float,
    clip_actions_min: wp.array1d(dtype=float),
    clip_actions_max: wp.array1d(dtype=float),
    taken_actions: wp.array2d(dtype=float),
    reduction: int,
    m: float,
    key: int,
    # outputs
    actions: wp.array2d(dtype=float),
    log_prob: wp.array2d(dtype=float),
    scale: wp.array1d(dtype=float),
):
    i, j = wp.tid()
    subkey = wp.rand_init(key + i, j)
    # clamp log standard deviations and compute distribution parameters
    scale[j] = wp.exp(wp.clamp(log_std[j], log_std_min, log_std_max))
    # sample actions
    if clip_actions_min:
        actions[i, j] = wp.clamp(wp.randn(subkey) * scale[j] + loc[i, j], clip_actions_min[j], clip_actions_max[j])
    else:
        actions[i, j] = wp.randn(subkey) * scale[j] + loc[i, j]
    # log of the probability density function
    if taken_actions:
        # mean
        if reduction == 0:
            wp.atomic_add(log_prob[i], 0, _log_prob(taken_actions[i, j], loc[i, j], scale[j]) / m)
        # sum
        elif reduction == 1:
            wp.atomic_add(log_prob[i], 0, _log_prob(taken_actions[i, j], loc[i, j], scale[j]))
        # prod
        elif reduction == 2:
            pass  # TODO: implement prod
        # none
        else:
            log_prob[i, j] = _log_prob(taken_actions[i, j], loc[i, j], scale[j])
    else:
        # mean
        if reduction == 0:
            wp.atomic_add(log_prob[i], 0, _log_prob(actions[i, j], loc[i, j], scale[j]) / m)
        # sum
        elif reduction == 1:
            wp.atomic_add(log_prob[i], 0, _log_prob(actions[i, j], loc[i, j], scale[j]))
        # prod
        elif reduction == 2:
            pass  # TODO: implement prod
        # none
        else:
            log_prob[i, j] = _log_prob(actions[i, j], loc[i, j], scale[j])


@wp.kernel
def _entropy(stddev: wp.array1d(dtype=float), entropy: wp.array1d(dtype=float)):
    i = wp.tid()
    entropy[i] = HALF_LOG_2_PI_PLUS + wp.log(stddev[i])


class GaussianMixin:
    def __init__(
        self,
        *,
        clip_actions: bool = False,
        clip_log_std: bool = True,
        min_log_std: float = -20,
        max_log_std: float = 2,
        reduction: Literal["mean", "sum", "prod", "none"] = "sum",
        role: str = "",
    ) -> None:
        """Gaussian mixin model (stochastic model).

        :param clip_actions: Flag to indicate whether the actions should be clipped to the action space.
        :param clip_log_std: Flag to indicate whether the log standard deviations should be clipped.
        :param min_log_std: Minimum value of the log standard deviation if ``clip_log_std`` is True.
        :param max_log_std: Maximum value of the log standard deviation if ``clip_log_std`` is True.
        :param reduction: Reduction method for returning the log probability density function.
            If ``"none"``, the log probability density function is returned as a tensor of shape
            ``(num_samples, num_actions)`` instead of ``(num_samples, 1)``.
        :param role: Role played by the model.

        :raises ValueError: If the reduction method is not valid.
        """
        self._g_clip_actions = clip_actions
        self._g_clip_actions_min, self._g_clip_actions_max = compute_space_limits(
            self.action_space if self._g_clip_actions else None, device=self.device, none_if_unbounded="both"
        )

        self._g_clip_log_std = clip_log_std
        self._g_log_std_min = min_log_std if self._g_clip_log_std else -wp.inf
        self._g_log_std_max = max_log_std if self._g_clip_log_std else wp.inf

        self._g_distribution = None

        if reduction not in ["mean", "sum", "prod", "none"]:
            raise ValueError("Reduction must be one of 'mean', 'sum', 'prod' or 'none'")
        self._g_reduction = ["mean", "sum", "prod", "none"].index(reduction)

        self._g_key = config.warp.key

    def act(self, inputs: dict[str, Any], *, role: str = "") -> tuple[wp.array, dict[str, Any]]:
        """Act stochastically in response to the observations/states of the environment.

        :param inputs: Model inputs. The most common keys are:

            - ``"observations"``: observation of the environment used to make the decision.
            - ``"states"``: state of the environment used to make the decision.
            - ``"taken_actions"``: actions taken by the policy for the given observations/states.
        :param role: Role played by the model.

        :return: Model output. The first component is the expected action/value returned by the model.
            The second component is a dictionary containing the following extra output values:

            - ``"log_std"``: log of the standard deviation.
            - ``"log_prob"``: log of the probability density function.
            - ``"mean_actions"``: mean actions (network output).
        """
        # map from observations/states to mean actions and log standard deviations
        mean_actions, outputs = self.compute(inputs, role)
        log_std = outputs["log_std"]

        self._g_key += 1
        shape = mean_actions.shape
        actions = wp.empty(shape=shape, dtype=wp.float32, device=self.device, requires_grad=True)
        if self._g_reduction == "none":
            log_prob = wp.zeros(shape=shape, dtype=wp.float32, device=self.device, requires_grad=True)
        else:
            log_prob = wp.zeros(shape=(shape[0], 1), dtype=wp.float32, device=self.device, requires_grad=True)
        scale = wp.empty(shape=log_std.shape, dtype=wp.float32, device=self.device, requires_grad=True)

        wp.launch(
            _gaussian,
            dim=shape,
            inputs=[
                mean_actions,
                log_std,
                self._g_log_std_min,
                self._g_log_std_max,
                self._g_clip_actions_min,
                self._g_clip_actions_max,
                inputs.get("taken_actions"),
                self._g_reduction,
                shape[1],
                self._g_key,
            ],
            outputs=[actions, log_prob, scale],
            device=self.device,
        )

        outputs["log_prob"] = log_prob
        outputs["mean_actions"] = mean_actions
        outputs["stddev"] = scale
        return actions, outputs

    def get_entropy(self, stddev: wp.array, *, role: str = "") -> wp.array:
        """Compute and return the entropy of the model.

        :param stddev: Model standard deviation.
        :param role: Role played by the model.

        :return: Entropy of the model.
        """
        entropy = wp.empty(shape=stddev.shape, dtype=wp.float32, device=self.device, requires_grad=True)
        wp.launch(
            _entropy,
            dim=stddev.shape,
            inputs=[stddev],
            outputs=[entropy],
            device=self.device,
        )
        return entropy
