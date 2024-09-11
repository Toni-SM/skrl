from typing import Any, Union

import re

from skrl import logger
from skrl.envs.wrappers.jax.base import MultiAgentEnvWrapper, Wrapper
from skrl.envs.wrappers.jax.bidexhands_envs import BiDexHandsWrapper
from skrl.envs.wrappers.jax.brax_envs import BraxWrapper
from skrl.envs.wrappers.jax.gym_envs import GymWrapper
from skrl.envs.wrappers.jax.gymnasium_envs import GymnasiumWrapper
from skrl.envs.wrappers.jax.isaacgym_envs import IsaacGymPreview2Wrapper, IsaacGymPreview3Wrapper
from skrl.envs.wrappers.jax.isaaclab_envs import IsaacLabMultiAgentWrapper, IsaacLabWrapper
from skrl.envs.wrappers.jax.omniverse_isaacgym_envs import OmniverseIsaacGymWrapper
from skrl.envs.wrappers.jax.pettingzoo_envs import PettingZooWrapper


__all__ = ["wrap_env", "Wrapper", "MultiAgentEnvWrapper"]


def wrap_env(env: Any, wrapper: str = "auto", verbose: bool = True) -> Union[Wrapper, MultiAgentEnvWrapper]:
    """Wrap an environment to use a common interface

    Example::

        >>> from skrl.envs.wrappers.jax import wrap_env
        >>>
        >>> # assuming that there is an environment called "env"
        >>> env = wrap_env(env)

    :param env: The environment to be wrapped
    :type env: gym.Env, gymnasium.Env, dm_env.Environment or VecTask
    :param wrapper: The type of wrapper to use (default: ``"auto"``).
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
                        * - Isaac Gym preview 2
                          - ``"isaacgym-preview2"``
                        * - Isaac Gym preview 3
                          - ``"isaacgym-preview3"``
                        * - Isaac Gym preview 4
                          - ``"isaacgym-preview4"``
                        * - Omniverse Isaac Gym
                          - ``"omniverse-isaacgym"``

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
    :type wrapper: str, optional
    :param verbose: Whether to print the wrapper type (default: ``True``)
    :type verbose: bool, optional

    :raises ValueError: Unknown wrapper type

    :return: Wrapped environment
    :rtype: Wrapper or MultiAgentEnvWrapper
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
            base_classes += [str(base).replace("<class '", "").replace("'>", "") for base in env.unwrapped.__class__.__bases__]
        except:
            pass
        base_classes = sorted(list(set(base_classes)))
        if verbose:
            logger.info(f"Environment wrapper: 'auto' (class: {', '.join(base_classes)})")

        if _in("omni.isaac.lab.envs..*", base_classes):
            return "isaaclab-*"
        elif _in("omni.isaac.gym..*", base_classes):
            return "omniverse-isaacgym"
        elif _in(["isaacgymenvs..*", "tasks..*.VecTask"], base_classes):
            return "isaacgym-preview4"  # preview 4 is the same as 3
        elif _in("rlgpu.tasks..*.VecTask", base_classes):
            return "isaacgym-preview2"
        elif _in("brax.envs..*", base_classes):
            return "brax"
        elif _in("robosuite.environments.", base_classes):
            return "robosuite"
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
    elif wrapper == "isaacgym-preview2":
        if verbose:
            logger.info("Environment wrapper: Isaac Gym (preview 2)")
        return IsaacGymPreview2Wrapper(env)
    elif wrapper == "isaacgym-preview3":
        if verbose:
            logger.info("Environment wrapper: Isaac Gym (preview 3)")
        return IsaacGymPreview3Wrapper(env)
    elif wrapper == "isaacgym-preview4":
        if verbose:
            logger.info("Environment wrapper: Isaac Gym (preview 4)")
        return IsaacGymPreview3Wrapper(env)  # preview 4 is the same as 3
    elif wrapper == "omniverse-isaacgym":
        if verbose:
            logger.info("Environment wrapper: Omniverse Isaac Gym")
        return OmniverseIsaacGymWrapper(env)
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
