from typing import Any, Mapping, Optional, Sequence, Union

import textwrap
import gymnasium

import flax.linen as nn  # noqa
import jax
import jax.numpy as jnp  # noqa

from skrl.models.jax import MultiCategoricalMixin  # noqa
from skrl.models.jax import Model
from skrl.utils.model_instantiators.jax.common import generate_containers
from skrl.utils.spaces.jax import unflatten_tensorized_space  # noqa


def multicategorical_model(
    *,
    observation_space: Optional[gymnasium.Space] = None,
    state_space: Optional[gymnasium.Space] = None,
    action_space: Optional[gymnasium.Space] = None,
    device: Optional[Union[str, jax.Device]] = None,
    unnormalized_log_prob: bool = True,
    reduction: str = "sum",
    network: Sequence[Mapping[str, Any]] = [],
    output: Union[str, Sequence[str]] = "",
    return_source: bool = False,
) -> Union[Model, str]:
    """Instantiate a :class:`~skrl.models.jax.multicategorical.MultiCategoricalMixin`-based model.

    :param observation_space: Observation space. The ``num_observations`` property will contain the size of the space.
    :param state_space: State space. The ``num_states`` property will contain the size of the space.
    :param action_space: Action space. The ``num_actions`` property will contain the size of the space.
    :param device: Data allocation and computation device. If not specified, the default device will be used.
    :param unnormalized_log_prob: Flag to indicate how to the model's output will be interpreted.
        If True, the model's output is interpreted as unnormalized log probabilities (it can be any real number),
        otherwise as normalized probabilities (the output must be non-negative, finite and have a non-zero sum).
    :param reduction: Reduction method for returning the log probability density function.
        Supported values are ``"mean"``, ``"sum"``, ``"prod"`` and ``"none"``.
        If ``"none"``, the log probability density function is returned as a tensor of shape
        ``(num_samples, num_actions)`` instead of ``(num_samples, 1)``.
    :param network: Network definition.
    :param output: Output expression.
    :param return_source: Whether to return the source string containing the model class used to
        instantiate the model rather than the model instance.

    :return: MultiCategorical model instance or definition source (if ``return_source`` is True).
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

    template = f"""class MultiCategoricalModel(MultiCategoricalMixin, Model):
    def __init__(
        self,
        observation_space,
        state_space,
        action_space,
        device=None,
        unnormalized_log_prob=True,
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
        MultiCategoricalMixin.__init__(self, unnormalized_log_prob=unnormalized_log_prob, reduction=reduction, role=role)

    def setup(self):
        {networks}

    def __call__(self, inputs, role=""):
        observations = unflatten_tensorized_space(self.observation_space, inputs.get("observations"))
        states = unflatten_tensorized_space(self.state_space, inputs.get("states"))
        taken_actions = unflatten_tensorized_space(self.action_space, inputs.get("taken_actions"))
        {forward}
        return output, {{}}
    """
    # return source
    if return_source:
        return template

    # instantiate model
    _locals = {}
    exec(template, globals(), _locals)
    return _locals["MultiCategoricalModel"](
        observation_space=observation_space,
        state_space=state_space,
        action_space=action_space,
        device=device,
        unnormalized_log_prob=unnormalized_log_prob,
        reduction=reduction,
    )
