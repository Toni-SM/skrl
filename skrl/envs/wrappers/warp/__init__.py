from typing import Any, Literal, Union

import re

from skrl import logger
from skrl.envs.wrappers.warp.base import MultiAgentEnvWrapper, Wrapper
from skrl.envs.wrappers.warp.gymnasium_envs import GymnasiumWrapper
from skrl.envs.wrappers.warp.isaaclab_envs import IsaacLabMultiAgentWrapper, IsaacLabWrapper


__all__ = ["wrap_env", "Wrapper", "MultiAgentEnvWrapper"]


def wrap_env(
    env: Any,
    wrapper: Literal[
        "auto",
        "gym",
        "gymnasium",
        "dm",
        "brax",
        "isaaclab",
        "isaaclab-single-agent",
        "isaacgym-preview4",
        "pettingzoo",
        "isaaclab-multi-agent",
        "bidexhands",
    ] = "auto",
    verbose: bool = True,
) -> Union[Wrapper, MultiAgentEnvWrapper]:
    """Wrap an environment to use a common interface.

    Example::

        >>> from skrl.envs.wrappers.warp import wrap_env
        >>>
        >>> # assuming that there is an environment called "env"
        >>> env = wrap_env(env)

    :param env: The environment instance to be wrapped.
    :param wrapper: The type of wrapper to use.
        If ``"auto"``, the wrapper will be automatically selected based on the environment class.
        The supported wrappers are described in the following table:

        .. list-table:: Single-agent environments |br|
            :header-rows: 1

            * - Environment
                - Wrapper tag
            * - OpenAI Gym
                - ``"gym"``
            * - Gymnasium
                - ``"gymnasium"``
            * - Brax
                - ``"brax"``
            * - Isaac Lab
                - ``"isaaclab"`` (``"isaaclab-single-agent"``)
            * - Isaac Gym preview 4
                - ``"isaacgym-preview4"``

        .. list-table:: Multi-agent environments |br|
            :header-rows: 1

            * - Environment
                - Wrapper tag
            * - Petting Zoo
                - ``"pettingzoo"``
            * - Isaac Lab
                - ``"isaaclab"`` (``"isaaclab-multi-agent"``)
            * - Bi-DexHands
                - ``"bidexhands"``
    :param verbose: Whether to print verbose information about the environment and the wrapper.

    :return: Wrapped environment instance.

    :raises ValueError: Unknown wrapper type.
    """

    def _get_wrapper_name(env, verbose):
        def _in(values, container):
            if type(values) == str:
                values = [values]
            for item in container:
                for value in values:
                    if value in item or re.match(value, item):
                        return True
            return False

        base_classes = [str(base).replace("<class '", "").replace("'>", "") for base in env.__class__.__bases__]
        try:
            base_classes += [
                str(base).replace("<class '", "").replace("'>", "") for base in env.unwrapped.__class__.__bases__
            ]
        except:
            pass
        base_classes = sorted(list(set(base_classes)))
        if verbose:
            logger.info(f"Environment wrapper: 'auto' (class: {', '.join(base_classes)})")

        if _in(["omni.isaac.lab.*", "isaaclab.*"], base_classes):
            return "isaaclab-*"
        elif _in(["isaacgymenvs..*", "tasks..*.VecTask"], base_classes):
            return "isaacgym-preview4"
        elif _in("brax.envs..*", base_classes):
            return "brax"
        elif _in("dm_env..*", base_classes):
            return "dm"
        elif _in("pettingzoo.utils.env", base_classes) or _in("pettingzoo.utils.wrappers", base_classes):
            return "pettingzoo"
        elif _in("gymnasium..*", base_classes):
            return "gymnasium"
        elif _in("gym..*", base_classes):
            return "gym"
        return base_classes

    if wrapper == "auto":
        wrapper = _get_wrapper_name(env, verbose)

    if wrapper == "gym":
        if verbose:
            logger.info("Environment wrapper: Gym")
        return GymWrapper(env)
    elif wrapper == "gymnasium":
        if verbose:
            logger.info("Environment wrapper: gymnasium")
        return GymnasiumWrapper(env)
    elif wrapper == "pettingzoo":
        if verbose:
            logger.info("Environment wrapper: Petting Zoo")
        return PettingZooWrapper(env)
    elif wrapper == "bidexhands":
        if verbose:
            logger.info("Environment wrapper: Bi-DexHands")
        return BiDexHandsWrapper(env)
    elif wrapper == "brax":
        if verbose:
            logger.info("Environment wrapper: Brax")
        return BraxWrapper(env)
    elif wrapper == "isaacgym-preview4":
        if verbose:
            logger.info("Environment wrapper: Isaac Gym (preview 4)")
        return IsaacGymPreview4Wrapper(env)
    elif type(wrapper) is str and wrapper.startswith("isaaclab"):
        # use specified wrapper
        if wrapper == "isaaclab-single-agent":
            env_type = "single-agent"
            env_wrapper = IsaacLabWrapper
        elif wrapper == "isaaclab-multi-agent":
            env_type = "multi-agent"
            env_wrapper = IsaacLabMultiAgentWrapper
        # detect the wrapper
        else:
            env_type = "single-agent"
            env_wrapper = IsaacLabWrapper
            if hasattr(env.unwrapped, "possible_agents"):
                env_type = "multi-agent"
                env_wrapper = IsaacLabMultiAgentWrapper
        # wrap the environment
        if verbose:
            logger.info(f"Environment wrapper: Isaac Lab ({env_type})")
        return env_wrapper(env)
    else:
        raise ValueError(f"Unknown wrapper type: {wrapper}")
