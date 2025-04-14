from typing import Any, Mapping, Optional, Sequence, Union

import textwrap
import gymnasium

import flax.linen as nn  # noqa
import jax
import jax.numpy as jnp  # noqa

from skrl.models.jax import GaussianMixin  # noqa
from skrl.models.jax import Model  # noqa
from skrl.utils.model_instantiators.jax.common import one_hot_encoding  # noqa
from skrl.utils.model_instantiators.jax.common import convert_deprecated_parameters, generate_containers
from skrl.utils.spaces.jax import unflatten_tensorized_space  # noqa


def gaussian_model(
    *,
    observation_space: Optional[gymnasium.Space] = None,
    state_space: Optional[gymnasium.Space] = None,
    action_space: Optional[gymnasium.Space] = None,
    device: Optional[Union[str, jax.Device]] = None,
    clip_actions: bool = False,
    clip_log_std: bool = True,
    min_log_std: float = -20,
    max_log_std: float = 2,
    reduction: str = "sum",
    initial_log_std: float = 0,
    fixed_log_std: bool = False,
    network: Sequence[Mapping[str, Any]] = [],
    output: Union[str, Sequence[str]] = "",
    return_source: bool = False,
) -> Union[Model, str]:
    """Instantiate a :class:`~skrl.models.jax.gaussian.GaussianMixin`-based model.

    :param observation_space: Observation space. The ``num_observations`` property will contain the size of the space.
    :param state_space: State space. The ``num_states`` property will contain the size of the space.
    :param action_space: Action space. The ``num_actions`` property will contain the size of the space.
    :param device: Data allocation and computation device. If not specified, the default device will be used.
    :param clip_actions: Flag to indicate whether the actions should be clipped to the action space.
    :param clip_log_std: Flag to indicate whether the log standard deviations should be clipped.
    :param min_log_std: Minimum value of the log standard deviation if ``clip_log_std`` is True.
    :param max_log_std: Maximum value of the log standard deviation if ``clip_log_std`` is True.
    :param reduction: Reduction method for returning the log probability density function.
        Supported values are ``"mean"``, ``"sum"``, ``"prod"`` and ``"none"``.
        If ``"none"``, the log probability density function is returned as a tensor of shape
        ``(num_samples, num_actions)`` instead of ``(num_samples, 1)``.
    :param initial_log_std: Initial value for the log standard deviation.
    :param fixed_log_std: Whether the log standard deviation parameter should be fixed.
        Fixed parameters have the gradient computation deactivated.
    :param network: Network definition.
    :param output: Output expression.
    :param return_source: Whether to return the source string containing the model class used to
        instantiate the model rather than the model instance.

    :return: Gaussian model instance or definition source (if ``return_source`` is True).
    """
    # parse model definition
    containers, output = generate_containers(network, output, embed_output=True, indent=1)

    # network definitions
    networks = []
    forward: list[str] = []
    for container in containers:
        networks.append(f'self.{container["name"]}_container = {container["sequential"]}')
        forward.append(f'{container["name"]} = self.{container["name"]}_container({container["input"]})')
    # process output
    if output["modules"]:
        networks.append(f'self.output_layer = {output["modules"][0]}')
        forward.append(f'output = self.output_layer({container["name"]})')
    if output["output"]:
        forward.append(f'output = {output["output"]}')
    else:
        forward[-1] = forward[-1].replace(f'{container["name"]} =', "output =", 1)

    # build substitutions and indent content
    networks = textwrap.indent("\n".join(networks), prefix=" " * 8)[8:]
    forward = textwrap.indent("\n".join(forward), prefix=" " * 8)[8:]
    if fixed_log_std:
        log_std_parameter = f'jnp.full(shape={output["size"]}, fill_value={float(initial_log_std)})'
    else:
        log_std_parameter = f'self.param("log_std_parameter", lambda _: jnp.full(shape={output["size"]}, fill_value={float(initial_log_std)}))'

    template = f"""class GaussianModel(GaussianMixin, Model):
    def __init__(
        self,
        observation_space,
        state_space,
        action_space,
        device=None,
        clip_actions=False,
        clip_log_std=True,
        min_log_std=-20,
        max_log_std=2,
        reduction="sum",
        role="",
        **kwargs,
    ):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
            **kwargs,
        )
        GaussianMixin.__init__(
            self,
            clip_actions=clip_actions,
            clip_log_std=clip_log_std,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
            reduction=reduction,
            role=role,
        )

    def setup(self):
        {networks}
        self.log_std_parameter = {log_std_parameter}

    def __call__(self, inputs, role=""):
        observations = unflatten_tensorized_space(self.observation_space, inputs.get("observations"))
        states = unflatten_tensorized_space(self.state_space, inputs.get("states"))
        taken_actions = unflatten_tensorized_space(self.action_space, inputs.get("taken_actions"))
        {forward}
        return output, {{"log_std": self.log_std_parameter}}
    """
    # return source
    if return_source:
        return template

    # instantiate model
    _locals = {}
    exec(template, globals(), _locals)
    return _locals["GaussianModel"](
        observation_space=observation_space,
        state_space=state_space,
        action_space=action_space,
        device=device,
        clip_actions=clip_actions,
        clip_log_std=clip_log_std,
        min_log_std=min_log_std,
        max_log_std=max_log_std,
        reduction=reduction,
    )
