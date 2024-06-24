from typing import Any, Union

import gym
import gymnasium

from skrl import logger
from skrl.envs.wrappers.jax.base import MultiAgentEnvWrapper, Wrapper
from skrl.envs.wrappers.jax.bidexhands_envs import BiDexHandsWrapper
from skrl.envs.wrappers.jax.gym_envs import GymWrapper
from skrl.envs.wrappers.jax.gymnasium_envs import GymnasiumWrapper
from skrl.envs.wrappers.jax.isaacgym_envs import IsaacGymPreview2Wrapper, IsaacGymPreview3Wrapper
from skrl.envs.wrappers.jax.isaaclab_envs import IsaacLabWrapper
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

                    +--------------------+-------------------------+
                    |Environment         |Wrapper tag              |
                    +====================+=========================+
                    |OpenAI Gym          |``"gym"``                |
                    +--------------------+-------------------------+
                    |Gymnasium           |``"gymnasium"``          |
                    +--------------------+-------------------------+
                    |Petting Zoo         |``"pettingzoo"``         |
                    +--------------------+-------------------------+
                    |Bi-DexHands         |``"bidexhands"``         |
                    +--------------------+-------------------------+
                    |Isaac Gym preview 2 |``"isaacgym-preview2"``  |
                    +--------------------+-------------------------+
                    |Isaac Gym preview 3 |``"isaacgym-preview3"``  |
                    +--------------------+-------------------------+
                    |Isaac Gym preview 4 |``"isaacgym-preview4"``  |
                    +--------------------+-------------------------+
                    |Omniverse Isaac Gym |``"omniverse-isaacgym"`` |
                    +--------------------+-------------------------+
                    |Isaac Lab           |``"isaaclab"``           |
                    +--------------------+-------------------------+
    :type wrapper: str, optional
    :param verbose: Whether to print the wrapper type (default: ``True``)
    :type verbose: bool, optional

    :raises ValueError: Unknown wrapper type

    :return: Wrapped environment
    :rtype: Wrapper or MultiAgentEnvWrapper
    """
    def _get_wrapper_name(env, verbose):
        def _in(value, container):
            for item in container:
                if value in item:
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

        if _in("omni.isaac.lab.envs.manager_based_env.ManagerBasedEnv", base_classes) or _in("omni.isaac.lab.envs.direct_rl_env.DirectRLEnv", base_classes):
            return "isaaclab"
        elif _in("omni.isaac.gym.vec_env.vec_env_base.VecEnvBase", base_classes) or _in("omni.isaac.gym.vec_env.vec_env_mt.VecEnvMT", base_classes):
            return "omniverse-isaacgym"
        elif _in("rlgpu.tasks.base.vec_task.VecTask", base_classes):
            return "isaacgym-preview2"
        elif _in("robosuite.environments.", base_classes):
            return "robosuite"
        elif _in("dm_env._environment.Environment.", base_classes):
            return "dm"
        elif _in("pettingzoo.utils.env", base_classes) or _in("pettingzoo.utils.wrappers", base_classes):
            return "pettingzoo"
        elif _in("gymnasium.core.Env", base_classes) or _in("gymnasium.core.Wrapper", base_classes):
            return "gymnasium"
        elif _in("gym.core.Env", base_classes) or _in("gym.core.Wrapper", base_classes):
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
    elif wrapper == "dm":
        if verbose:
            logger.info("Environment wrapper: DeepMind")
        return DeepMindWrapper(env)
    elif wrapper == "robosuite":
        if verbose:
            logger.info("Environment wrapper: Robosuite")
        return RobosuiteWrapper(env)
    elif wrapper == "bidexhands":
        if verbose:
            logger.info("Environment wrapper: Bi-DexHands")
        return BiDexHandsWrapper(env)
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
    elif wrapper == "isaaclab" or wrapper == "isaac-orbit":
        if verbose:
            logger.info("Environment wrapper: Isaac Lab")
        return IsaacLabWrapper(env)
    else:
        raise ValueError(f"Unknown wrapper type: {wrapper}")
